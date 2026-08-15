# CF Aşaması Raporu — Fitness Hipotezi Doğrulandı (fold 1 pilot)

> **SONRADAN DÜZELTME (bkz. MAKALE_KATKILARI.md K1):** Bu rapordaki kısıtsız
> pred_mae sonuçlarının bir kısmı dejenere çözümden (dev tek küme = kümesiz kNN
> taklidi) besleniyor. %2.7 iyileşme iddiası, denge kısıtlı fitness rakamlarıyla
> (K=10'da +%1.6) değiştirildi. Bu rapor tarihsel kayıt olarak duruyor.

Protokol: ML-100K fold 1; iç-train %90 / iç-val %10 (test'e sızıntı yok);
NMF-20 (iç-train'den), K=7; tahminci = küme-film ortalaması + fallback zinciri;
skor u1.test MAE/RMSE. 3 seed pilot. Veri: `results/cf_stage.csv`.

## Ana tablo (test MAE, 3 seed ortalaması)

| Fitness | Yöntem | Lloyd | MAE | WCSS | Fallback% |
|---|---|---|---|---|---|
| **pred_mae** | **GWO** | **yok** | **0.8259** | 2066 | 0.84 |
| **pred_mae** | **AVOA** | **yok** | **0.8295** | 1254 | 0.67 |
| **pred_mae** | **NGO** | **yok** | **0.8300** | 1802 | 1.07 |
| pred_mae | NGO/GWO/AVOA | var | 0.8438–0.8459 | ~225–233 | 1.7–2.1 |
| wcss | GWO (Lloyd'suz) | yok | 0.8460 | 248 | 2.04 |
| wcss | **B0_KMEANS++** | var | 0.8489 | 216 | 2.86 |
| wcss | AVOA/GWO/NGO | var | 0.8493–0.8509 | ~217–220 | 2.3–2.5 |
| wcss | B0_RANDOM_LLOYD | var | 0.8534 | 250 | 2.85 |
| wcss | B0_RANDOM_CENT | yok | 0.8553 | 385 | 2.80 |

## Dört bulgu

1. **Fitness fonksiyonu belirleyici — cevap: evet, en önemli şey.**
   pred_mae fitness'lı GWO: MAE 0.8259 vs KMeans++ 0.8489 → **%2.7 iyileşme**
   (eski WCSS yaklaşımında %0.05'ti; ~50 kat büyük fark). Üstelik pred_mae kolu
   yalnızca 2.000 NFE kullandı (WCSS kolu 15.000) — 7.5 kat az bütçeyle kazandı.

2. **Lloyd, MAE kazancını GERİ ALIYOR (Lloyd paradoksu).**
   pred_mae merkezleri Lloyd'a verilince MAE 0.826→0.845'e bozuluyor; çünkü Lloyd
   merkezleri WCSS optimumuna geri çekiyor (WCSS 1254-2066→225'e düşüyor ama MAE
   yükseliyor). **WCSS optimumu ile MAE optimumu farklı yerlerde** — kanıt bu tablo.
   Sonuç: pred_mae hattında kmref KAPALI olmalı; kmref yalnız WCSS hattında anlamlı.

3. **WCSS hattı eski bulguları birebir tekrarlıyor:** herkes 0.846–0.851 bandında,
   KMeans++ ortada, fark yok. Eski %0.05'lik sonuçlarınız hatalı değilmiş —
   yanlış hedefin doğru ölçümüymüş.

4. **pred_mae kümeleri daha kapsayıcı:** fallback %0.7–1.1 vs %2.4–2.9. MAE'ye göre
   şekillenen kümeler test filmlerini daha iyi örtüyor.

## Cevaplar

- **Ek preprocess gerekti mi?** Hayır. NMF-20 tek başına yeterli (uzay kontrolü bunu
  göstermişti). Kullanıcı z-score / LOF gray-sheep artık zorunlu değil, ablasyon
  bölümü malzemesi. Tek zorunlu kural: her şey iç-train'den fit edilir.
- **Merkez arama neyi minimize ediyor?** İç-validasyon MAE'sini (fold1 train'in
  %10'u). Test hiçbir aşamada fitness'a girmiyor — hakem sorusu buradan gelir,
  cevap hazır.

## Pilotun sınırları → doğrulama planı

- n=3 seed → 30 seed'e çıkar; CV5 (u1–u5) tekrarı; kullanıcı-bazlı Wilcoxon için
  tahminlerin diske yazılması (`--save-preds` eklenecek).
- pred_mae NFE'si 5.000'e çıkarılıp doyum eğrisi çizilmeli.
- K duyarlılığı (K=5,7,10,14) pred_mae hattında.
- Tahminci şu an küme-ortalaması; küme-içi kNN ile tekrar (muhtemelen MAE daha da
  düşer — literatür tablosuna o girer).
