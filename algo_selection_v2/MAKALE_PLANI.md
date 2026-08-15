# Makale / Tez Yazım Planı

Anlatı omurgası tek cümle: **"Meta-sezgisel kümelemeli CF'de kazanç, algoritma
seçiminden değil hedef fonksiyonun tahmine hizalanmasından gelir; WCSS hattında
güçlü baseline'ı kimse geçemez, tahmine hizalı fitness %2.7 MAE iyileştirir."**

Her iddia bir CSV'ye dayanır — metin uzadıkça kural: *bir deney = bir tablo/şekil
+ en fazla iki paragraf yorum; gerisi ek (appendix).*

## Bölüm iskeleti ve kaynak eşlemesi

| # | Bölüm | Hedef | İçerik → kaynak dosya |
|---|---|---|---|
| 1 | Giriş | 1.5–2 sf | Problem (CF + seyreklik), boşluk: (a) film önerisinde AVOA yok, (b) meta-kümeleme makaleleri zayıf baseline kullanıyor. Katkı listesi: 3 madde (aşağıda) |
| 2 | İlgili Çalışmalar | 2–3 sf | `docs/literature.csv`'den tablo: Katarya 2016 (PSO), Katarya 2018 (GWO), GOA 2024, Thakrar, AVOA orijinal. Her satır: yöntem/veri/baseline/iyileşme. "Baseline'ları zayıf" gözlemi burada kurulur |
| 3 | Yöntem | 3–4 sf | Pipeline şeması (veri→NMF→meta merkez→[Lloyd?]→küme-CF→metrik). İki fitness tanımı (WCSS formülü; pred_mae + iç-val protokolü). Sızıntı önlemleri kutusu |
| 4 | Deney Düzeneği | 1.5 sf | ML-100K resmi 5-fold; NFE-eşit bütçe gerekçesi; seed/istatistik protokolü (Friedman+Holm, kullanıcı-bazlı Wilcoxon); `KONTROL_LISTESI.md` bölüm E özeti |
| 5.1 | Uzay seçimi | 1 sf | Hopkins + K-sweep tablosu → `results/space_check.csv`. "Eski uzayda sil=−0.03" dersi |
| 5.2 | Algoritma eleme | 1.5 sf | 22 algo × 5 seed; rastgele-arama eşiği; dejenere çözüm/silhouette tuzağı → `pure_meta_summary.csv`, `PURE_META_RAPORU.md` |
| 5.3 | Final tur | 1.5 sf | NFE-eşit, 30 seed; iki track tablosu; Lloyd iterasyon kanıtı → `final_run.csv`, `FINAL_TUR_RAPORU.md` |
| 5.4 | CF sonuçları (ANA) | 2 sf | Ana tablo + Lloyd paradoksu grafiği → `cf_stage.csv`, `CF_ASAMASI_RAPORU.md` |
| 6 | Tartışma | 2 sf | (i) kritik replikasyon: baseline seçimi iyileşme iddialarını belirler; (ii) Lloyd paradoksu: WCSS≠MAE optimumu; (iii) NFL bağlamında AVOA/GWO seçimi |
| 7 | Sonuç + Gelecek | 0.5–1 sf | ML-1M genelleme, kNN tahminci, çok amaçlı fitness |

Toplam gövde ~16–18 sf; kalan her şey Ek'e (ham tablolar, tüm K sweep'leri,
22-algoritma tam listesi, koşu komutları).

## Katkı listesi (Giriş'in son paragrafı — aynen bu üçlü)

1. Film önerisi alanında AVOA ile kümeleme ilk kez denenmiş, adil (NFE-eşit,
   güçlü baseline'lı, Holm-düzeltmeli) protokolle 22 algoritma içinden seçilmiştir.
2. Kritik replikasyon: literatürdeki meta-kümeleme iyileşmeleri zayıf baseline'a
   (random-init k-means) karşı yeniden üretilmiş (+%6.5–8.5), güçlü baseline'a
   (KMeans++ n_init=10) karşı kaybolduğu gösterilmiştir.
3. Tahmine hizalı fitness (iç-val MAE) önerilmiş; Lloyd'un optimize edemediği bu
   hedefte KMeans++'a karşı %2.7 MAE iyileşmesi elde edilmiştir (7.5 kat az bütçeyle).

## Uzunluk kontrol kuralları

- Tablolar CSV'den script'le üretilir (elle sayı taşınmaz — tutarlılık + emek).
- Her bulgu bir kez anlatılır: sonuçta veri, tartışmada yorum; tekrar yasak.
- "Doğru yapılanlar/hatalar" iç mutfağı (KONTROL_LISTESI B bölümü) makaleye girmez;
  sadece 4. bölümde protokol gerekçesi olarak 2-3 cümle.
- Şekil bütçesi: 5 (pipeline, K-sweep, yakınsama, iki-track bar, Lloyd paradoksu).
- Yazım sırası: 5→3→4→6→2→1→7 (sonuçlar hazırken deneyden başla; giriş en son).

## Kalan doğrulama işleri (yazıma başlamadan)

- [ ] pred_mae hattı: 30 seed + CV5 + kullanıcı-bazlı Wilcoxon (tahminleri kaydet)
- [ ] pred_mae NFE doyum eğrisi (2k→5k→10k)
- [ ] K duyarlılığı (5/7/10/14) pred_mae hattında
- [ ] Küme-içi kNN tahminci ile ana tablo tekrarı
- [ ] (Ops.) WNMF uzayı vs NMF-20 karşılaştırması, ML-1M genelleme
