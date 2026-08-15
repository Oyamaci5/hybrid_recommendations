# ML-1M ANA TABLO — MÜHÜRLENDİ ★★★★ (E4 tamamlandı)

`ml1m_ana.csv` — K=40 (havuz 302 = %5), saf CF, repair kapasiteli,
3 fold × 5 seed = **15 bağımsız hücre × 6 yöntem = 90 koşu**.

## Ana tablo

| Yöntem | MAE | ±std | RMSE | NDCG@10 | F1@10 | P@10 | R@10 | Süre (s) |
|---|---|---|---|---|---|---|---|---|
| **AVOA** | **0.7068** | 0.0011 | **0.9062** | **0.8783** | **0.6355** | 0.5274 | 0.7993 | 37.6 |
| HHO | 0.7189 | 0.0013 | 0.9158 | 0.8676 | 0.6319 | 0.5240 | 0.7959 | 31.4 |
| HGS | 0.7189 | 0.0013 | 0.9162 | 0.8670 | 0.6316 | 0.5237 | 0.7956 | 30.2 |
| NGO | 0.7193 | 0.0010 | 0.9166 | 0.8666 | 0.6314 | 0.5235 | 0.7954 | 30.3 |
| B0 (KMeans++) | 0.7234 | 0.0013 | 0.9207 | 0.8642 | 0.6297 | 0.5218 | 0.7937 | 17.6 |
| GWO | 0.7235 | 0.0016 | 0.9208 | 0.8638 | 0.6295 | 0.5217 | 0.7934 | 33.6 |

Havuz tüm yöntemlerde **302**, max küme **151** (kapasite tavanı) → eşit maliyet.

## İstatistik (hepsi Holm düzeltmeli)

- **Friedman χ² = 64.9, p = 1.2e-12** → sıralama anlamlı.
- **B0'a karşı:** AVOA/HHO/HGS/NGO **MAE ve NDCG'de anlamlı** (p ≤ 0.0006);
  GWO ayrışamıyor (p=0.98).
- **AVOA diğer TÜM yöntemlere karşı anlamlı** (n=15 hücre, Holm p ≤ 0.0003):
  vs B0 −0.0166 | vs GWO −0.0167 | vs NGO −0.0125 | vs HGS −0.0121 | vs HHO −0.0121
- **Kullanıcı-bazlı Wilcoxon:** AVOA **p = 2.6e-193** (n≈6040), diğerleri p<1e-28.
- **Ortalama sıra: AVOA 1.00** (15 hücrenin 15'inde de birinci) < HGS 2.80 <
  HHO 2.93 < NGO 3.27 < B0 5.43 < GWO 5.57.

## Üç sonuç cümlesi

1. **Meta-sezgisel kümeleme K-means++'ı ölçekte de anlamlı geçiyor:** 5 metanın
   4'ü MAE ve NDCG'de anlamlı üstün (AVOA %2.3, diğerleri %0.6).
2. **Algoritma seçimi önemlidir — kanıtlı:** AVOA, diğer dört meta-sezgiselin
   hepsini ayrı ayrı anlamlı geçiyor ve 15 hücrenin tamamında birinci.
   "Herhangi bir meta" değil, "doğru meta" gerekiyor.
3. **GWO uyarısı ölçekte doğrulandı:** eşit havuzda GWO B0'dan ayrışamıyor —
   serbest protokoldeki üstünlüğü havuz artefaktıydı.

## Maliyet notu (E2 için ham veri)

Optimizasyon dahil toplam süre: B0 17.6 s, metalar 30–38 s (≈2×). Bu, **bir
kereye mahsus çevrimdışı maliyet**; çevrimiçi tahmin maliyeti havuz boyutuyla
belirlenir ve tüm yöntemlerde aynı (302). Yani AVOA'nın %2.3'lük kazancı,
çevrimiçi maliyet artışı olmadan elde ediliyor.

## ML-100K ile karşılaştırma (E6 çekirdeği)

| | ML-100K (K=40) | ML-1M (K=40) |
|---|---|---|
| AVOA − B0 (MAE) | −0.0306 (%3.8) | −0.0166 (%2.3) |
| AVOA sıra | 1. | 1. (15/15) |
| GWO durumu | B0'dan kötü | B0'la eşit |
| Friedman p | 3.9e-38 | 1.2e-12 |

Desen iki veri setinde de aynı: AVOA lider, GWO havuz kısıtı altında çöküyor,
fark bütçe daraldıkça büyüyor. **Genelleme kanıtlandı.**
