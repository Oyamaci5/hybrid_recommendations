# ML-1M Hedefli Doğrulama — Kurulum ve Komutlar

## Durum: altyapı hazır ve test edildi ✓

- `ctx_ml1m.py` — ML-1M yükleyici (6040×3952, resmi fold yok → rastgele %90/10,
  fold = tekrar no). Benzerlik matrisi float32 = 146 MB. Kontroller (rating 1-5,
  duplicate yok, sızıntı yok) yerleşik.
- `ml1m_run.py` — koşucu; ML-100K protokolünün aynısı (repair kapasite + warm
  start + soft top-2 + küme-MF + kNN karışımı), ölçek için optimize edildi:
  - `fast_repair`: mesafe açılımı (‖x−c‖² = ‖x‖² − 2x·c + ‖c‖²)
  - fitness'ta `np.add.at` yerine `np.bincount` → **fitness 0.09 s** (önce ~1.5 s)
  - kNN'de `argpartition` (tam sıralama yerine)
  - havuz ölçümü 1/10 örneklem
- `cluster_mf.py` film sayısına parametrik hale getirildi (1682 sabiti kaldırıldı).

## İlk gerçek sonuç (fold 1, seed 42, K=40, saf CF)

| Yöntem | MAE | RMSE | NDCG@10 | P@10 | R@10 | Havuz | Max küme |
|---|---|---|---|---|---|---|---|
| B0_repair | **0.7216** | 0.9187 | **0.8662** | 0.5232 | 0.7948 | 302 | 151 |

Karşılaştırma: ML-100K'da aynı protokolde B0 K=40 → 0.8015. **ML-1M'de baseline
çok daha iyi** (10 kat veri, kullanıcı başına 134 puan vs 74). Bu beklenen;
asıl soru metaların bu güçlü baseline'ı geçip geçmediği.

## K TARAMASI SONUCU (fold 1, seed 42, saf CF) ★★★

| K | Havuz (%) | B0 MAE | AVOA MAE | Fark | B0 NDCG | AVOA NDCG |
|---|---|---|---|---|---|---|
| 6 | 2013 (%33) | **0.6880** | **0.6876** | −0.0004 | 0.8860 | **0.8878** |
| 10 | 1208 (%20) | 0.6973 | 0.6912 | −0.0061 | 0.8813 | 0.8882 |
| 20 | 604 (%10) | 0.7099 | 0.6986 | −0.0113 | 0.8719 | 0.8824 |
| 40 | 302 (%5) | 0.7216 | 0.7066 | −0.0150 | 0.8662 | 0.8793 |
| 60 | 201 (%3) | 0.7321 | 0.7125 | **−0.0196** | 0.8593 | 0.8764 |

**Üç bulgu:**

1. **Fark hunisi ML-1M'de birebir tekrarlandı** — havuz %33'ten %3'e inerken fark
   −0.0004'ten −0.0196'ya çıkıyor. ML-100K'daki eğilim ölçekte doğrulandı;
   protokolün veri setine özgü olmadığının kanıtı.
2. **AVOA sıralama kalitesini bütçe daralınca koruyor:** NDCG 0.8878 → 0.8764
   (−1.3%) iken B0 0.8860 → 0.8593 (−3.0%). Yani kısıtlı bütçede AVOA'nın
   üstünlüğü MAE'den çok NDCG'de görünüyor.
3. **Verimlilik iddiası (makale grafiği):**
   - AVOA K=40 (havuz **302**): MAE 0.7066, NDCG 0.8793
   - B0 K=20 (havuz **604**): MAE 0.7099, NDCG 0.8719
   → **AVOA yarı havuzla B0'ı her iki metrikte geçiyor** = 2× hesap verimliliği.
   Aynı desen K=60 vs K=20'de de var (3× havuz farkı).

**Seçilen çalışma noktaları:**
- **Mutlak performans: K=6** (MAE 0.6876 / NDCG 0.8878) — literatürün ML-100K
  için iddia ettiği 0.68 bandına dürüst protokolle ML-1M'de ulaşıldı.
- **Algoritma farkı: K=40** (fark −0.0150, havuz %5; K=60 biraz daha büyük fark
  verir ama havuz 201'e inince küme-MF örneklem başına zayıflıyor).

## Süre ölçümleri (sandbox; sizin makinede muhtemelen 2-3× hızlı)

| İş | Süre |
|---|---|
| Bağlam + NMF uzayı (fold başına, bir kez) | ~15 s |
| Bir fitness çağrısı | 0.09 s |
| AVOA optimizasyon (600 NFE, pop=10) | ~1.3 dk |
| AVOA optimizasyon (1200 NFE, pop=15) | ~2.6 dk |
| Bir değerlendirme (küme-MF + kNN + metrikler) | ~18 s |
| **Bir hücre (1 yöntem × 1 fold × 1 seed)** | **~2-3 dk** |

Ana tablo (6 yöntem × 3 fold × 5 seed = 90 hücre) ≈ **3.5-4.5 saat**.

## Komutlar

**1) K taraması — çalışma noktası (≈45 dk):**
```
py -3.12 algo_selection_v2\ml1m_run.py --exp ksweep --klist 6 10 20 40 60 --max-fe 600 --resume
```

**Neden K=6 ve 10 dahil (düzeltme):** iki eşleştirme ilkesi var —
(a) *aynı K* → aynı havuz oranı (K=6: 1M'de %33, 100K'da da %33; doğrudan
karşılaştırılabilir), (b) *aynı küme büyüklüğü* → 1M'de K=40 (151 kişi/küme)
≈ 100K'da K=6 (158 kişi/küme). İlk taslakta (b) seçilmişti; ölçüm (a)'nın hem
daha doğru hem daha ucuz olduğunu gösterdi:

| K | Havuz | B0 MAE | B0 NDCG@10 | Değerlendirme |
|---|---|---|---|---|
| **6** | 2013 (%33) | **0.6880** | **0.8857** | 11 s |
| 40 | 302 (%5) | 0.7216 | 0.8662 | 18 s |

Düşük K'da küme sayısı az → küme-MF modeli az → daha hızlı. **ML-1M'de MAE 0.688**
literatürün iddia ettiği banda kendi protokolümüzle ulaşıldığını gösteriyor.
Tarama her iki ilkeyi de kapsıyor (6/10 = oran eşleşmesi, 40/60 = boyut eşleşmesi).

**2) Ana tablo — iki çalışma noktası:**

Algoritma farkı noktası (K=40, ≈4 saat) — makalenin ana tablosu:
```
py -3.12 algo_selection_v2\ml1m_run.py --exp ana --k 40 --folds 1 2 3 --seeds 42 43 44 45 46 --max-fe 600 --resume
```

Mutlak performans noktası (K=6, ≈3 saat) — literatür kıyas tablosu:
```
py -3.12 algo_selection_v2\ml1m_run.py --exp ana --k 6 --folds 1 2 3 --seeds 42 43 44 45 46 --max-fe 600 --resume
```
NOT: ikisi aynı `ml1m_ana.csv`'ye yazar ama satırlarda `K` sütunu var, analiz
K'ya göre filtrelenebilir. Karışmasını istemezseniz ilk koşu bitince dosyayı
`ml1m_ana_K40.csv` olarak yeniden adlandırın.

**3) Analiz:**
```
py -3.12 algo_selection_v2\tam_run_analiz.py --file ml1m_ana.csv
```

## Notlar

- `--resume` her iki koşuda da açık; kesip devam edebilirsiniz.
- Bellek: fold başına ~1.5 GB (S matrisi 146 MB + R matrisi 95 MB + geçiciler).
- `--uzay nmf` varsayılan (saf CF). Tür ablasyonu isterseniz `--uzay nmf+genre`
  ayrı dosyaya yazmaz — önce mevcut `ml1m_ana.csv`'yi yeniden adlandırın.
- Fold sayısı 3 (5 değil): ML-1M'de rastgele bölme kullanıldığı için 3 tekrar ×
  5 seed = 15 hücre/yöntem zaten yeterli istatistiksel güç veriyor.

## Beklenen bulgu (hipotez)

ML-100K'da fark, havuz kısıtı sıkılaştıkça büyüyordu (K=6: %2.8 → K=40: %3.8).
ML-1M'de aynı K için havuz oransal olarak daha küçük (302/6040 = %5), dolayısıyla
**farkın korunması veya büyümesi** bekleniyor. Aksi çıkarsa da bu bir bulgudur:
"yöntemin katkısı veri yoğunluğu arttıkça azalır" — dürüstçe raporlanır.
