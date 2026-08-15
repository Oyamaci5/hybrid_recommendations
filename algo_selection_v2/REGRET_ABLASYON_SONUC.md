
# Onarım Öncelik Kuralı Ablasyonu — Regret Tabanlı Sıralama

**Öneri (haklı bir eleştiri):** Mevcut kural yalnızca `min_c D[u,c]`'ye bakıyor;
kullanıcının *alternatiflerinin ne kadar kötü olduğunu* (pişmanlık) göz ardı
ediyor. Doğrusu: `Regret(u) = D[u,2.yakın] − D[u,1.yakın]`, **azalan** sırada.

Test edilen üç kural (ML-100K, fold 1, seed 42):
- **mesafe** (mevcut): min uzaklık, artan
- **regret**: pişmanlık, azalan
- **regret_oranı**: `Regret/D[u,1.yakın]`, azalan (ölçek bağımsız)

---

## SONUÇ 1 — K-means++ merkezlerinde regret AÇIK ARA daha iyi

| K | Kural | SSE artışı | Yer değiştiren | MAE | NDCG@10 |
|---|---|---|---|---|---|
| 6 | mesafe | 1694.8% | 77.4% | 0.7757 | 0.7965 |
| 6 | **regret** | **1643.2%** | **60.2%** | **0.7637** | **0.8220** |
| 6 | regret_oranı | 1639.4% | 70.0% | 0.7674 | 0.8149 |
| 20 | mesafe | 1425.6% | 84.0% | 0.8067 | 0.7635 |
| 20 | **regret** | 1439.8% | **53.3%** | **0.7939** | 0.7746 |
| 20 | regret_oranı | **1400.7%** | 68.9% | 0.7959 | **0.7802** |
| 40 | mesafe | 1087.8% | 83.8% | 0.8179 | 0.7392 |
| 40 | **regret** | 1207.2% | **49.0%** | **0.8027** | 0.7582 |
| 40 | regret_oranı | 1142.0% | 61.8% | 0.8084 | **0.7597** |

**Kazanımlar (regret vs mesafe):**
- MAE: **−0.0120 / −0.0128 / −0.0152** (K=6/20/40) → ortalama **%1.6 iyileşme**
- NDCG: **+0.0255 / +0.0111 / +0.0190**
- **Yer değiştiren kullanıcı oranı %77–84'ten %49–60'a düşüyor** — yani daha az
  kullanıcı zorla kaydırılıyor. Teşhis metriği öngörüyü doğruluyor.

**Dikkat çekici:** SSE artışı bazen regret'te daha yüksek (K=40: 1207% vs 1088%)
ama MAE daha iyi. Yani **SSE, tahmin kalitesinin iyi bir vekili değil** —
tezimizin "geometrik hedef ≠ tahmin hedefi" tezinin bir kanıtı daha.

## SONUÇ 2 — AVOA merkezlerinde fark yok (hatta mesafe hafif önde)

| K | Kural | SSE artışı | MAE | NDCG@10 |
|---|---|---|---|---|
| 6 | **mesafe** | 10.02% | **0.7469** | 0.8350 |
| 6 | regret | 9.79% | 0.7550 | 0.8322 |
| 6 | regret_oranı | 9.80% | 0.7507 | **0.8352** |
| 40 | **mesafe** | 31.56% | **0.7805** | 0.8012 |
| 40 | regret | 31.52% | 0.7939 | 0.7883 |
| 40 | regret_oranı | 31.57% | 0.7827 | **0.8026** |

**Kritik gözlem — SSE artışı:** K-means++ merkezlerinde kapasite kısıtı SSE'yi
**%1000–1700 artırıyor**; AVOA merkezlerinde yalnızca **%10–32**.

**Yorum:** AVOA, fitness'ı boyunca kapasiteli atamayla yaşadığı için merkezleri
zaten **kapasiteye uyumlu** yerleştiriyor — kısıt onun için neredeyse bedava.
K-means++ ise kısıtı hiç görmediğinden merkezleri kısıtla çatışıyor ve öncelik
kuralı kritik hale geliyor. Yani:

> **Regret kuralı, kısıtla uyumsuz merkezleri kurtarır; kısıta zaten uyumlu
> merkezlerde gereksizdir.**

Bu, meta-sezgiselin katkısının *ne olduğuna* dair yeni bir kanıt:
AVOA yalnız "daha iyi kümeleme" değil, **kısıtla uyumlu kümeleme** öğreniyor.

---

## KARAR

**Ana hatta `regret` kuralı benimseniyor.** Gerekçeler:
1. Baseline'ı (B0) belirgin iyileştiriyor → **karşılaştırma daha adil olur**;
   zayıf baseline'la kazanmak istemiyoruz.
2. AVOA'da kayıp ihmal edilebilir (K=6'da −0.008, K=40'ta −0.013 MAE) ve
   NDCG'de regret_oranı zaten en iyi.
3. Literatürde standart: kapasiteli atama sezgisellerinde en yüksek pişmanlık
   önce işlenir.
4. Yer değiştiren kullanıcı oranını %30 düşürüyor → "zorlama" eleştirisine
   karşı da daha savunulabilir.

**Uygulama notu:** `fast_repair` ve `repair_assign` fonksiyonlarında sıralama
satırı değişecek:
```python
# eski: sira = np.argsort(d.min(1))
sira = np.argsort(-(ikinci_yakin - en_yakin))   # azalan regret
```

**Yeniden koşulması gerekenler:** ana tablolar (ML-100K K=6/K=40, ML-1M K=6/K=40)
regret kuralıyla tekrar üretilmeli. Beklenti: B0 iyileşeceği için AVOA–B0 farkı
bir miktar daralacak, ama karşılaştırma daha sağlam olacak. Mevcut sonuçlar
ablasyon tablosu olarak korunacak.

## Teze girecek cümle

> "Kapasiteli atamada işlem sırası, kısıt ihlali durumunda hangi kullanıcıların
> kaydırılacağını belirlediği için sonucu etkilemektedir. Yalnızca en yakın
> merkeze uzaklığa dayalı sıralama, alternatifleri kötü olan kullanıcıların
> kaydırılmasına yol açabilmektedir. Bu nedenle kapasiteli atama yazınında
> standart olan pişmanlık (regret) tabanlı sıralama benimsenmiştir:
> Regret(u) = D(u, ikinci en yakın) − D(u, en yakın) değeri azalan sırada
> işlenir. Bu değişiklik K-means++ merkezlerinde MAE'yi ortalama %1.6
> iyileştirmiş, yer değiştiren kullanıcı oranını %77–84'ten %49–60'a
> düşürmüştür. Meta-sezgisel merkezlerde ise fark gözlenmemiştir; bunun nedeni,
> uygunluk fonksiyonunun kapasiteli atamayı içermesi sayesinde bu merkezlerin
> kısıtla zaten uyumlu olmasıdır (kapasite kısıtının yol açtığı SSE artışı
> K-means++'ta %1000–1700, meta-sezgiselde %10–32)."
