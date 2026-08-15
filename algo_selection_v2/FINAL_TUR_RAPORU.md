# Final Tur Raporu — NFE-Eşit Bütçe (15.000), 30 Seed, nmf20, K=7

Veri: `results/final_run.csv` (270 satır: 7 meta × 30 + 2 baseline × 30).
Friedman χ²=124.5, p=3.9e-23. Tüm ikili testler Holm düzeltmeli Wilcoxon (n=30).

## Ana tablo (kmref sonrası WCSS)

| Yöntem | WCSS | std | Lloyd iter. | KMeans++'a karşı | RANDOM'a karşı |
|---|---|---|---|---|---|
| **B0_KMEANS++** | **224.23** | 1.27 | 17.7 | — | — |
| GWO | 226.43 | 2.59 | 7.6 | −2.2 kötü (p=.003) | **+21.0 iyi (p<.0001)** |
| AVOA | 226.75 | 5.10 | 9.2 | −2.5 kötü (p=.026) | **+20.6 iyi (p<.0001)** |
| NGO | 227.54 | 3.96 | 9.9 | −3.3 kötü (p=.0005) | **+19.9 iyi (p<.0001)** |
| HHO | 231.06 | 6.86 | 15.1 | −6.8 kötü | +16.3 iyi |
| AGTO | 231.24 | 5.74 | 12.5 | −7.0 kötü | +16.2 iyi |
| HGS | 231.50 | 7.52 | 14.5 | −7.3 kötü | +15.9 iyi |
| PSO | 245.66 | 14.24 | 23.6 | −21.4 kötü | fark yok (p=.47) |
| B0_RANDOM→Lloyd | 247.39 | 7.24 | 23.9 | — | — |

## Saf arama gücü (wcss_meta, Lloyd'suz, eşit NFE)

**AVOA 1. (rank 1.57, 245.5)** — NGO 2. (2.00) — GWO 3. (2.63) — AGTO — HGS — HHO — PSO son.
Dikkat: AVOA'nın ham çıktısı (245.5), random-init+Lloyd'un toplamından (247.4) bile iyi.

## Üç kesin sonuç

1. **KMeans++ kendi hedefinde yenilmez (artık kanıtlı).** 7 metanın 7'si de anlamlı
   şekilde geride (en yakın GWO −2.2, p=.003). WCSS, Lloyd'un kapalı-form çözdüğü
   hedef; meta orada ancak "eşitleyebilir". Bu soru kapandı — tezde net yazılacak.

2. **Literatür iddiası zayıf baseline'a karşı doğru, güçlüye karşı değil.**
   Random-init k-means'e karşı tüm metalar +%6.5–8.5 iyileşme (p<.0001) — makalelerin
   raporladığı türden fark birebir üretildi. Ama güçlü baseline (KMeans++ n_init=10)
   gelince fark tersine dönüyor. Bu "kritik replikasyon" tezin özgün katkılarından
   biri: *iyileşme iddiaları baseline seçimine bağlıdır.*

3. **Meta-init'in ölçülebilir gerçek faydası: Lloyd yükünü yarıya indiriyor.**
   İterasyon: meta-init 7.6–9.9 / KMeans++ 17.7 / random 23.9. "Daha iyi başlangıç"
   iddiasının dürüst kanıtı bu sütun.

## Algoritma seçimi — SONUÇ

**AVOA seçimi artık veriyle savunulabilir:** eşit bütçede en iyi saf arayıcı (rank
1.57), kmref sonrası ilk üçte (GWO ile istatistiksel olarak başabaş), üstüne
özgünlük cümlesi ("film önerisinde AVOA'lı kümeleme ilk"). Yanında GWO
(Katarya 2018 kıyası) ve NGO (tutarlı 2.) CF aşamasına taşınır. HHO yalnız
literatür karşılaştırma tablosunda kalır; PSO elendi (rastgele aramadan farksız).

## Sıradaki aşama — CF (hedef değişiyor)

WCSS yarışı bitti; asıl ayrışma Lloyd'un giremediği hedefte olacak:
1. AVOA / GWO / NGO merkezleri → Thakrar pipeline init (kmref açık) → CV5
   MAE/RMSE/NDCG@10 + kullanıcı-bazlı Wilcoxon (literatür tablosu).
2. Fitness = knn_mae pilotu (fold 1): merkez ararken doğrudan validation MAE
   minimize edilir — KMeans++'ın yarışamadığı alan; makale düzeyi fark buradan
   beklenir.
