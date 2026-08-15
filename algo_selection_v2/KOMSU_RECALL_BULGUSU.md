
# Kümelemenin Gerçek İşlevi: Yaklaşık Komşu Arama (ANN) ★★★★

## Soru
"Küme-içi kNN zaten en yakın 20 kişiyi seçiyorsa, kümelemenin ne anlamı var?"

## Ölçüm: komşu geri çağırma (neighbor recall@20)

Her kullanıcı için: **tüm veri setindeki gerçek en-benzer 20 komşunun** kaçı
kümeleme havuzunda kalıyor? (ML-100K, fold 1, seed 42, saf CF, repair'li)

| K | Havuz | B0 recall@20 | **AVOA recall@20** | Oran |
|---|---|---|---|---|
| 6 | 315 (%33) | %28.0 | **%60.6** | **2.2×** |
| 20 | 95 (%10) | %13.3 | **%28.3** | **2.1×** |
| 40 | 47 (%5) | %8.9 | **%16.9** | **1.9×** |

## Yorum — sezginin neden yanıldığı

**Küme-içi kNN, "en yakın 20 kişiyi" seçmiyor; havuzdaki en yakın 20 kişiyi
seçiyor.** Ve havuz, gerçek en yakın 20 komşunun yalnızca %9–61'ini içeriyor.
Yani kümeleme kararı, kNN'in *hangi adayları görebileceğini* belirliyor —
kümeleme olmasaydı kNN gerçek komşuları bulurdu ama 6040 kişiyle karşılaştırma
yapmak zorunda kalırdı.

**Doğru çerçeve: kümeleme bir yaklaşık en yakın komşu (ANN) indeksidir.**
- Amacı daha iyi komşu bulmak değil, **aynı komşuları daha ucuza bulmak**.
- Meta-sezgiselin işi de bu indeksi iyileştirmek: aynı bütçede daha çok gerçek
  komşuyu havuzda tutmak.
- **AVOA, K-means++'a göre aynı maliyette 2 kat fazla gerçek komşu koruyor.**
  Yöntemin neden kazandığının mekanizma açıklaması bu.

## Bu bulgu neyi açıklıyor

1. **Neden kümesiz kNN çoğu noktada en iyi:** recall %100 olduğu tek durum o.
   (Tablo A'da KNN_ALL 0.7216 ile önde — beklenen sonuç, artık nedeni de belli.)
2. **Neden fark bütçe daraldıkça büyüyor:** havuz küçüldükçe her iki yöntemin
   recall'ü düşüyor, ama oran ~2× sabit kalıyor; mutlak kayıp B0'da daha fazla.
3. **Neden tahmin hatası recall kadar bozulmuyor:** %28 recall ile bile MAE
   0.7654 (AVOA %60.6 ile 0.7450). Çünkü kaybedilen komşuların yerine geçen
   "ikinci sınıf" komşular hâlâ bilgi taşıyor + MF bileşeni açığı kapatıyor.
   Sıralama metrikleri (NDCG) recall'e daha duyarlı.
4. **Neden K=6'da MAE farkı küçük ama NDCG farkı var:** o noktada recall zaten
   yüksek (%61), MAE doygunlaşıyor; sıralama hâlâ ayrışıyor.

## Makaleye katkısı

- **Yeni değerlendirme ekseni:** literatürdeki hiçbir CF-kümeleme çalışması
  komşu recall'ü raporlamıyor; yalnız MAE/precision bakıyorlar. Oysa kümelemenin
  *asıl işi* budur ve doğrudan ölçülebilir.
- **Mekanizma kanıtı:** "AVOA daha iyi" demek yerine "AVOA aynı maliyette 2 kat
  fazla gerçek komşu koruyor, MAE farkı bunun sonucu" demek çok daha güçlü.
- **Tasarım rehberi:** recall–maliyet eğrisi, uygulamacının bütçesine göre
  K seçmesini sağlar (MAE eğrisinden daha yorumlanabilir).

## Sıradaki (öneri)
1. ML-1M'de aynı ölçüm (6040 kullanıcı; recall'ün daha da düşmesi beklenir).
2. Diğer metalarla recall karşılaştırması (HGS/HHO/NGO/GWO) — sıralamanın
   MAE sıralamasıyla uyuşup uyuşmadığı.
3. Fitness'a recall terimi eklemek: merkezleri doğrudan "gerçek komşuları koru"
   hedefiyle aramak (yeni varyant adayı; şu an dolaylı olarak MAE üzerinden oluyor).
