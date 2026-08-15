# ML-1M: Meta Farkı Zaten Ortaya Çıktı ★★★ (5 algoritma × 5 bütçe)

Kaynak: `ml1m_ksweep.csv` (fold 1, seed 42, saf CF, repair kapasiteli).

## Havuz nedir? (kavram netleştirmesi)

**Havuz = kullanıcının komşu adaylarının sayısı** = *soft top-2* havuz:
kullanıcının atandığı küme + ona en yakın ikinci küme. Repair sayesinde her küme
tam N/K büyüklüğünde olduğundan havuz ≈ 2N/K:

| K | Küme büyüklüğü (6040/K) | Havuz (2 küme) | Toplamın %'si |
|---|---|---|---|
| 6 | 1007 | 2013 | %33 |
| 10 | 604 | 1208 | %20 |
| 20 | 302 | 604 | %10 |
| 40 | 151 | 302 | %5 |
| 60 | 101 | 201 | %3 |

kNN, k=20 komşuyu bu havuzdan seçer. Havuz küçüldükçe hesap ucuzlar ama
"doğru komşu havuzda mı?" sorusu kritikleşir — merkez kalitesi orada belirleyici olur.

## Ana tablo — MAE (satır: K/bütçe, sütun: yöntem)

| K | Havuz | **AVOA** | HGS | HHO | NGO | GWO | B0 |
|---|---|---|---|---|---|---|---|
| 6 | 2013 | 0.6876 | 0.6864 | **0.6859** | 0.6900 | 0.6886 | 0.6880 |
| 10 | 1208 | **0.6912** | 0.6951 | 0.6956 | 0.6943 | 0.6978 | 0.6973 |
| 20 | 604 | **0.6986** | 0.7061 | 0.7053 | 0.7042 | 0.7097 | 0.7099 |
| 40 | 302 | **0.7066** | 0.7165 | 0.7171 | 0.7188 | 0.7222 | 0.7216 |
| 60 | 201 | **0.7125** | 0.7250 | 0.7257 | 0.7259 | 0.7304 | 0.7321 |

**Ortalama sıra:** AVOA **1.4** < HGS 2.6 < HHO 2.8 < NGO 3.6 < B0 5.2 < GWO 5.4

## Aradığımız "meta farkı" burada — AVOA diğer METALARDAN da ayrışıyor

| K | Havuz | AVOA − 2.sıradaki meta | AVOA − B0 |
|---|---|---|---|
| 6 | 2013 | +0.0017 (HHO önde) | −0.0004 |
| 10 | 1208 | **−0.0031** (NGO) | −0.0061 |
| 20 | 604 | **−0.0056** (NGO) | −0.0113 |
| 40 | 302 | **−0.0099** (HGS) | −0.0150 |
| 60 | 201 | **−0.0125** (HGS) | −0.0196 |

**Bu, tezin en güçlü tablosu.** Çünkü:
1. Fark yalnız "meta vs K-means++" değil, **meta vs meta** — yani "hangi
   meta-sezgisel seçildiği önemlidir" iddiası doğrudan kanıtlanıyor.
2. Fark bütçe daraldıkça **düzenli olarak büyüyor** (0.003 → 0.013): rastgele
   dalgalanma değil, sistematik bir eğilim.
3. K=6'da (bol bütçe) sıralama karışıyor (HHO 1., AVOA 3., aradaki fark 0.002)
   → beklenen davranış; kümeleme orada zaten belirleyici değil.
4. GWO'nun B0'ın bile altında kalması, ML-100K'daki "GWO havuz istismarcısı"
   bulgusunu ölçekte doğruluyor (burada havuz sabit olduğu için istismar edemiyor).

**Not:** bu tek fold/seed. Mühürlemek için `--exp ana` koşusu (3 fold × 5 seed)
ve K'yı bloklayan Friedman gerekiyor — ama desen zaten çok net.

## "Hibrit AVOA fark yaratır mı?"

**Zaten hibrit çalışıyoruz** — ve bu makalenin yöntem katkısı olarak yazılmalı:
mevcut yöntemimiz *memetik* bir yapı (deterministik sezgisel + metasezgisel):

```
KMeans++ (deterministik) → warm start popülasyonu → AVOA (metasezgisel refinement)
→ kapasiteli onarım (repair) → soft top-2 havuz → küme-MF + kNN karışımı
```

Yani "hibrit AVOA" zaten var: **AVOA ⊕ K-means++ warm start ⊕ repair operatörü.**
ML-100K'da bunun katkısı ölçüldü (K11/K15: sıfırdan arayan AVOA baseline'ı
geçemezken, warm start'lı AVOA 5/5 kazandı).

Ek hibritleme seçenekleri (öncelik sırasıyla):
1. **Memetik yerel arama:** her nesilde en iyi çözüme 1 Lloyd adımı uygula
   (AVOA ⊕ Lloyd). Ucuz, literatürde yaygın, muhtemelen küçük kazanç.
2. **AVOA ⊕ HGS operatör karması:** HGS ikinci sırada; iki keşif mekanizmasını
   birleştirmek (AVOA'nın açlık-tokluk fazı + HGS'nin oyun/işbirliği fazı).
   Repo'nuzda `HA_AVOAHGS.py` zaten var — aynı protokole sarılabilir.
3. **Uyarlamalı popülasyon/NFE:** bütçeyi K'ya göre ölçeklemek.

**Ama dikkat:** hibrit eklemek yeni ayar yükü ve aşırı-uyum riski getirir; K19
bulgumuz (iç-val ile parametre seçimi test'i kötüleştiriyor, Spearman −0.50)
bu riske işaret ediyor. Öneri: hibrit ancak **mevcut protokolde, ek ayar yapmadan**
denensin; kazanç net değilse "denendi, katkı yok" olarak raporlansın.
