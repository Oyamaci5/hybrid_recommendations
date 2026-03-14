// ============================================================
// CONFIGURATION
// ============================================================
const CONFIG = {
  PARENT_FOLDER_ID: '1uTwopnvg2L2ru_K75Ais8hHsXvKoUhPf',   // Google Drive parent folder ID
  SPREADSHEET_ID: '1zcmqJRX3vlc66bJU5TbS29Yh-1eeXg8vJINSG_4rT-E',        // Google Sheet ID (leave empty to auto-create)
  GEMINI_API_KEY: 'AIzaSyD9owkx9r1sF42J_obwv7RDY_nTdWNPFNo',         // Gemini API key
  GEMINI_MODEL: 'gemini-2.5-flash',                   // Gemini model
  CACHE_PROPERTY_KEY: 'processed_files_cache',        // Script property key for cache
  MAX_PDF_BYTES: 20 * 1024 * 1024,                    // Max PDF size to send to Gemini (20MB)
};

// ============================================================
// MAIN ENTRY POINT
// ============================================================
function updateArticleSheets() {
  const parentFolder = DriveApp.getFolderById(CONFIG.PARENT_FOLDER_ID);
  const subfolders = getSubfolders(parentFolder);
  const ss = getOrCreateSpreadsheet();
  const cache = loadCache();

  let cacheUpdated = false;

  for (const folder of subfolders) {
    const folderName = folder.getName();
    const folderId = folder.getId();

    // Create sheet page only if it doesn't exist
    let sheet = ss.getSheetByName(folderName);
    if (!sheet) {
      sheet = ss.insertSheet(folderName);
      setupSheetHeaders(sheet);
      Logger.log(`Created new sheet page: "${folderName}"`);
    }

    // Get all PDFs in this folder
    const pdfFiles = getPdfFiles(folder);

    // Determine which files are new (not in cache)
    const folderCache = cache[folderId] || {};
    let newEntries = [];

    for (const file of pdfFiles) {
      const fileId = file.getId();
      if (folderCache[fileId]) {
        // Already processed — skip
        continue;
      }

      // New file: extract year via Gemini
      const fileName = file.getName();
      const articleName = cleanFileName(fileName);
      let releaseYear = '';

      try {
        releaseYear = extractYearFromPdf(file);
      } catch (e) {
        Logger.log(`Error extracting year from "${fileName}": ${e.message}`);
        releaseYear = 'N/A';
      }

      // Save to cache
      folderCache[fileId] = {
        articleName: articleName,
        releaseYear: releaseYear,
        processedAt: new Date().toISOString(),
      };
      cacheUpdated = true;

      Logger.log(`Processed: "${articleName}" → Year: ${releaseYear}`);
    }

    // Update cache for this folder
    cache[folderId] = folderCache;

    // Rebuild sheet from cache (keeps order consistent)
    rebuildSheet(sheet, folderCache);
  }

  // Remove default "Sheet1" if it exists and is empty
  cleanupDefaultSheet(ss);

  if (cacheUpdated) {
    saveCache(cache);
  }

  Logger.log('Update complete.');
}

// ============================================================
// SPREADSHEET HELPERS
// ============================================================
function getOrCreateSpreadsheet() {
  if (CONFIG.SPREADSHEET_ID && CONFIG.SPREADSHEET_ID !== 'YOUR_SPREADSHEET_ID_HERE') {
    return SpreadsheetApp.openById(CONFIG.SPREADSHEET_ID);
  }

  // Auto-create spreadsheet in the parent folder
  const ss = SpreadsheetApp.create('Articles Index');
  const file = DriveApp.getFileById(ss.getId());
  const parentFolder = DriveApp.getFolderById(CONFIG.PARENT_FOLDER_ID);
  parentFolder.addFile(file);
  DriveApp.getRootFolder().removeFile(file);

  // Store the ID so we reuse it next time
  Logger.log(`Created spreadsheet: ${ss.getUrl()}`);
  Logger.log(`Set CONFIG.SPREADSHEET_ID to: ${ss.getId()}`);

  return ss;
}

function setupSheetHeaders(sheet) {
  sheet.getRange('A1').setValue('Article Name').setFontWeight('bold');
  sheet.getRange('B1').setValue('Release Year').setFontWeight('bold');
  sheet.setColumnWidth(1, 500);
  sheet.setColumnWidth(2, 120);
  sheet.getRange('1:1').setBackground('#4285f4').setFontColor('#ffffff');
  sheet.setFrozenRows(1);
}

function rebuildSheet(sheet, folderCache) {
  // Clear existing data (keep headers)
  const lastRow = sheet.getLastRow();
  if (lastRow > 1) {
    sheet.getRange(2, 1, lastRow - 1, 2).clearContent();
  }

  // Write all cached entries
  const entries = Object.values(folderCache);
  if (entries.length === 0) return;

  // Sort by article name
  entries.sort((a, b) => a.articleName.localeCompare(b.articleName));

  const data = entries.map(e => [e.articleName, e.releaseYear]);
  sheet.getRange(2, 1, data.length, 2).setValues(data);
}

function cleanupDefaultSheet(ss) {
  const defaultSheet = ss.getSheetByName('Sheet1');
  if (defaultSheet && ss.getSheets().length > 1) {
    const lastRow = defaultSheet.getLastRow();
    if (lastRow <= 1) {
      ss.deleteSheet(defaultSheet);
    }
  }
}

// ============================================================
// DRIVE HELPERS
// ============================================================
function getSubfolders(parentFolder) {
  const folders = [];
  const iter = parentFolder.getFolders();
  while (iter.hasNext()) {
    folders.push(iter.next());
  }
  // Sort alphabetically for consistent ordering
  folders.sort((a, b) => a.getName().localeCompare(b.getName()));
  return folders;
}

function getPdfFiles(folder) {
  const files = [];
  const iter = folder.getFilesByType('application/pdf');
  while (iter.hasNext()) {
    files.push(iter.next());
  }
  return files;
}

function cleanFileName(fileName) {
  // Remove .pdf extension and clean up
  return fileName.replace(/\.pdf$/i, '').trim();
}

// ============================================================
// GEMINI API — EXTRACT YEAR FROM PDF
// ============================================================
function extractYearFromPdf(file) {
  const fileSize = file.getSize();

  if (fileSize > CONFIG.MAX_PDF_BYTES) {
    Logger.log(`File too large (${fileSize} bytes), trying text extraction fallback.`);
    return extractYearFromFileName(file.getName());
  }

  const pdfBlob = file.getBlob();
  const pdfBytes = pdfBlob.getBytes();
  const base64Pdf = Utilities.base64Encode(pdfBytes);

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${CONFIG.GEMINI_MODEL}:generateContent?key=${CONFIG.GEMINI_API_KEY}`;

  const payload = {
    contents: [
      {
        parts: [
          {
            inlineData: {
              mimeType: 'application/pdf',
              data: base64Pdf,
            },
          },
          {
            text: 'Extract ONLY the publication/release year of this academic article/paper. Return ONLY the 4-digit year number (e.g., 2023). If you cannot determine the year, return "N/A". Do not include any other text.',
          },
        ],
      },
    ],
    generationConfig: {
      temperature: 0,
      maxOutputTokens: 100,
      thinkingConfig: {
        thinkingBudget: 0,
      },
    },
  };

  const options = {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify(payload),
    muteHttpExceptions: true,
  };

  const response = UrlFetchApp.fetch(url, options);
  const status = response.getResponseCode();

  if (status !== 200) {
    Logger.log(`Gemini API error (${status}): ${response.getContentText()}`);
    // Fallback: try to get year from filename
    return extractYearFromFileName(file.getName());
  }

  const json = JSON.parse(response.getContentText());
  
  // Gemini 2.5 thinking models return multiple parts:
  // parts with thought=true are internal reasoning, the actual answer is in non-thought parts
  const parts = json.candidates?.[0]?.content?.parts || [];
  let answerText = '';
  for (const part of parts) {
    if (!part.thought && part.text) {
      answerText = part.text.trim();
    }
  }

  // Fallback: try first part if no non-thought part found
  if (!answerText) {
    answerText = parts[0]?.text?.trim() || '';
  }

  Logger.log(`Gemini raw answer for "${file.getName()}": "${answerText}"`);

  // Validate it's a 4-digit year
  const yearMatch = answerText.match(/\b(19|20)\d{2}\b/);
  if (yearMatch) {
    return yearMatch[0];
  }

  return answerText || 'N/A';
}

function extractYearFromFileName(fileName) {
  const match = fileName.match(/(19|20)\d{2}/);
  return match ? match[0] : 'N/A';
}

// ============================================================
// CACHE MANAGEMENT (Script Properties)
// ============================================================
function loadCache() {
  const props = PropertiesService.getScriptProperties();
  const raw = props.getProperty(CONFIG.CACHE_PROPERTY_KEY);
  if (!raw) return {};

  try {
    return JSON.parse(raw);
  } catch (e) {
    Logger.log('Cache corrupted, resetting.');
    return {};
  }
}

function saveCache(cache) {
  const props = PropertiesService.getScriptProperties();
  const json = JSON.stringify(cache);

  // Script Properties has a 9KB per property limit
  // If cache is too large, split across multiple properties
  if (json.length > 8000) {
    saveLargeCache(cache);
  } else {
    props.setProperty(CONFIG.CACHE_PROPERTY_KEY, json);
  }
}

function saveLargeCache(cache) {
  const props = PropertiesService.getScriptProperties();
  const json = JSON.stringify(cache);
  const chunkSize = 8000;
  const chunks = Math.ceil(json.length / chunkSize);

  // Clear old chunks
  const oldChunkCount = parseInt(props.getProperty(CONFIG.CACHE_PROPERTY_KEY + '_chunks') || '0');
  for (let i = 0; i < oldChunkCount; i++) {
    props.deleteProperty(CONFIG.CACHE_PROPERTY_KEY + '_' + i);
  }

  // Save new chunks
  for (let i = 0; i < chunks; i++) {
    props.setProperty(
      CONFIG.CACHE_PROPERTY_KEY + '_' + i,
      json.substring(i * chunkSize, (i + 1) * chunkSize)
    );
  }
  props.setProperty(CONFIG.CACHE_PROPERTY_KEY + '_chunks', chunks.toString());
  props.deleteProperty(CONFIG.CACHE_PROPERTY_KEY); // remove single-key version
}

function loadCache() {
  const props = PropertiesService.getScriptProperties();

  // Check for chunked cache first
  const chunkCount = parseInt(props.getProperty(CONFIG.CACHE_PROPERTY_KEY + '_chunks') || '0');
  if (chunkCount > 0) {
    let json = '';
    for (let i = 0; i < chunkCount; i++) {
      json += props.getProperty(CONFIG.CACHE_PROPERTY_KEY + '_' + i) || '';
    }
    try {
      return JSON.parse(json);
    } catch (e) {
      Logger.log('Chunked cache corrupted, resetting.');
      return {};
    }
  }

  // Single property cache
  const raw = props.getProperty(CONFIG.CACHE_PROPERTY_KEY);
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch (e) {
    Logger.log('Cache corrupted, resetting.');
    return {};
  }
}

// ============================================================
// TRIGGER MANAGEMENT
// ============================================================

/**
 * Run once to set up a time-based trigger that checks for new files periodically.
 * This is the recommended approach since Drive folder change triggers are limited.
 */
function setupTimeTrigger() {
  // Remove existing triggers for this function
  const triggers = ScriptApp.getProjectTriggers();
  for (const trigger of triggers) {
    if (trigger.getHandlerFunction() === 'updateArticleSheets') {
      ScriptApp.deleteTrigger(trigger);
    }
  }

  // Create a trigger that runs every 10 minutes
  ScriptApp.newTrigger('updateArticleSheets')
    .timeDriven()
    .everyMinutes(10)
    .create();

  Logger.log('Time trigger set: updateArticleSheets will run every 10 minutes.');
}

/**
 * Remove all triggers.
 */
function removeTriggers() {
  const triggers = ScriptApp.getProjectTriggers();
  for (const trigger of triggers) {
    ScriptApp.deleteTrigger(trigger);
  }
  Logger.log('All triggers removed.');
}

/**
 * Clear the file cache. Use this if you want to force re-processing of all files.
 */
function clearCache() {
  const props = PropertiesService.getScriptProperties();

  // Clear chunked cache
  const chunkCount = parseInt(props.getProperty(CONFIG.CACHE_PROPERTY_KEY + '_chunks') || '0');
  for (let i = 0; i < chunkCount; i++) {
    props.deleteProperty(CONFIG.CACHE_PROPERTY_KEY + '_' + i);
  }
  props.deleteProperty(CONFIG.CACHE_PROPERTY_KEY + '_chunks');
  props.deleteProperty(CONFIG.CACHE_PROPERTY_KEY);

  Logger.log('Cache cleared. Next run will re-process all files.');
}

// ============================================================
// MENU (for manual runs from the Sheet UI)
// ============================================================
function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('📄 Articles')
    .addItem('🔄 Update Now', 'updateArticleSheets')
    .addItem('⏰ Setup Auto-Update (10 min)', 'setupTimeTrigger')
    .addItem('🗑️ Clear Cache & Reprocess', 'clearCacheAndRerun')
    .addItem('❌ Remove Auto-Update', 'removeTriggers')
    .addToUi();
}

function clearCacheAndRerun() {
  clearCache();
  updateArticleSheets();
}
