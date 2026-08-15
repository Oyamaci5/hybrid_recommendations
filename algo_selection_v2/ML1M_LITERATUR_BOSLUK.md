# ML-1M Literatürüne Göre Eksiklerimiz — Odak Dağıtmadan

Amaç: ML-1M ile çalışan makalelerin standart beklentilerinden bizde olmayanları
listelemek, **yapımıza uyanları** seçmek, sırayla denemek.

---

## 1. Konum kontrolü — ML-1M'de bizim yerimiz

| Kaynak / yöntem | MAE | RMSE | Not |
|---|---|---|---|
| Standart kullanıcı-kNN (literatür) | — | ~0.99 | zayıf referans |
| SVD tabanlı MBCF | 0.6909 | 0.8761 | yaygın referans |
| **Bizim (K=6, HHO)** | **0.6859** | ~0.93 | havuz %33 |
| **Bizim (K=6, AVOA)** | **0.6876** | ~0.93 | havuz %33 |
| **Bizim (K=40, AVOA)** | 0.7066 | ~0.95 | havuz **%5** |

**MAE'de SVD referansının önündeyiz** (0.686 vs 0.691), RMSE'de gerideyiz
(0.93 vs 0.876). Bu fark açıklanabilir: bizim tahminci kNN ağırlıklı, RMSE
büyük hatalara duyarlı; MF ağırlığını (β) artırmak RMSE'yi düşürür.
→ **Eksik 1: β seçimi MAE'ye göre yapılıyor; RMSE hedefli varyant raporlanmalı.**

## 2. ML-1M makalelerinin standart beklentileri vs bizim durum

| Beklenti | Bizde | Yapılacak |
|---|---|---|
| %90/10 rastgele bölme, çok tekrar | ✓ (fold=tekrar) | — |
| MAE **ve** RMSE birlikte | ✓ | — |
| Top-N metrikleri (P/R/NDCG/F1) | ✓ | — |
| Kullanıcı başına 20+ puan filtresi | ✓ (ML-1M zaten böyle) | — |
| Ölçeklenebilirlik/süre raporu | kısmi (süre var, analiz yok) | **E2** |
| Seyreklik/kapsama (coverage) | ✗ | **E1** |
| Cold-start alt analizi | ML-100K'da var, 1M'de yok | **E3** |
| İstatistiksel test | ML-100K'da var, 1M'de yok | **E4** |
| Bellek/karmaşıklık tartışması | ✗ | E5 (yazıda, deney yok) |
| 100K–1M karşılaştırmalı tablo | kısmi | **E6** |

## 3. Yapımıza uyan, denemeye değer 6 madde (öncelik sırasıyla)

**E1 — Coverage (katalog kapsama) + kullanıcı kapsaması.** Kümeleme havuzu
daralttığı için "kaç farklı film önerilebiliyor" sorusu doğal. Ucuz: mevcut
tahminlerden hesaplanır, yeni koşu gerekmez. Literatürde standart, bizde yok.
*Beklenti:* AVOA'nın dengeli kümeleri daha yüksek coverage vermeli → ek katkı.

**E2 — Ölçeklenebilirlik tablosu.** Zaten `sure_s` kaydediyoruz. Havuz→süre
ilişkisini tabloya dökmek yeter: "havuz %33→%5 iken tahmin süresi X→Y".
Makalenin "verimlilik" iddiasını sayısallaştırır. Deney gerekmez.

**E3 — ML-1M cold-start katmanı.** ML-100K'daki analizin aynısı (kullanıcıyı
puan sayısına göre katmanlara ayır). 1M'de cold kullanıcı sayısı çok daha fazla
→ istatistiksel olarak daha güçlü. Tek koşudan türetilebilir (tahminler elde).

**E4 — ML-1M istatistiksel mühür.** `--exp ana` (3 fold × 5 seed) + K'yı
bloklayan Friedman + kullanıcı-bazlı Wilcoxon. **En kritik eksik** — ana iddia
şu an tek fold/seed.

**E5 — RMSE-hedefli β varyantı.** İç-val'de β'yı RMSE'ye göre seçen ikinci satır.
Tek satır kod, SVD referansıyla RMSE kıyasını adil yapar.

**E6 — 100K vs 1M karşılaştırmalı tablo.** "Fark hunisi iki veri setinde de
aynı" mesajını tek şekilde gösterir. Deney yok, mevcut CSV'lerden derlenir.

## 4. Denemeye DEĞMEYENLER (odak dağıtmamak için, gerekçeli)

- Derin öğrenme baseline'ları (NCF, AutoRec): farklı model ailesi, tezin sorusu
  "kümeleme algoritması seçimi" — kapsam dışı, gelecek çalışma notu yeter.
- ML-10M/20M: veri zaten var ama 1M genellemesi iddiayı taşıyor; maliyet 10 kat.
- Zaman-farkındalıklı bölme (temporal split): protokol değişikliği tüm
  sonuçların tekrarını gerektirir.
- Yeni metasezgiseller: 22 algoritma zaten elendi, kapı kapalı.

## 5. Sıra (önerilen)

1. **E4** (ana koşu, mühür) — arka planda başlat, en uzun iş
2. E1 + E2 + E6 (hesap/derleme, koşu gerektirmez) — E4 koşarken yapılır
3. E3 (cold-start katmanı) — E4 bitince tahminlerden
4. E5 (RMSE-β) — tek ek koşu
5. Varyant deneyleri (ölçekli bütçe / memetik / hibrit) — `ml1m_varyant.py`

Komutlar:
```
py -3.12 algo_selection_v2\ml1m_run.py --exp ana --k 40 --folds 1 2 3 --seeds 42 43 44 45 46 --max-fe 600 --resume
py -3.12 algo_selection_v2\ml1m_varyant.py --klist 20 40 60 --max-fe 600 --resume
```
