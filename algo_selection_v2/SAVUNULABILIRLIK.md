
# "Bu Kadar Yöntem Kullanmak Zorlama mı?" — Dürüst Değerlendirme

Üç soru: (1) Katarya eski değil mi? (2) Yeni makaleler neden kNN kullanmıyor?
(3) ALS-MF + kNN birlikte kullanmak savunulabilir mi?

---

## 1. kNN eski mi? — Hayır, hâlâ standart referans

Katarya 2016/2017 eski ama **kNN'in kendisi güncelliğini korudukça** bu bir
sorun değil. Kritik kaynak:

**Dacrema, Cremonesi & Jannach (2019), "Are We Really Making Much Progress?
A Worrying Analysis of Recent Neural Recommendation Approaches", RecSys'19**
(+ 2021 TOIS takip çalışması):
- 18 yeni sinir ağı tabanlı öneri yöntemini yeniden üretmeye çalışmışlar;
  **çoğu, iyi ayarlanmış kNN ve basit MF referanslarını geçemiyor.**
- Sonuç: kNN "eski" değil, **güçlü ve dürüst referans**; asıl sorun onu
  düzgün ayarlamadan atlayan çalışmalar.

Ayrıca `Surprise` kütüphanesinin resmî benchmark'ında KNNWithMeans hâlâ standart
(ML-100K MAE 0.750) — bizim kümesiz kNN'imiz 0.7467 ile onunla uyumlu çıkmıştı;
yani ölçüm hattımızın doğruluğunun bağımsız kanıtı.

## 2. Yeni meta-sezgisel makaleler neden küme-ortalaması kullanıyor?

Üç sebep — üçü de **bizim lehimize**:

1. **Kolaylık:** küme-ortalaması tek satır; küme-içi kNN benzerlik matrisi +
   komşu seçimi gerektirir. Bu makalelerin odağı optimizasyon algoritması,
   tahmin katmanı ihmal edilmiş.
2. **Ölçek kaygısı:** benzerlik matrisi O(N²) bellek ister. Ama bizim
   protokolümüz tam da bunu çözüyor — havuz kısıtı ile arama O(2N/K)'ya iniyor.
3. **Farkında olmama:** tahmin katmanının sonucu ne kadar belirlediğini
   ölçmemişler. Biz ölçtük: **küme-ortalaması %9–11 daha kötü** (ML-1M: 0.807 vs
   0.734; ML-100K: 0.849 vs 0.788) ve **algoritma farkını da eziyor**.

→ Bu, "zayıflık" değil, tezin **bulgusu**: literatürün zayıf sonuçlarının bir
kısmı kümeleme yönteminden değil, tahmin katmanının ihmalinden geliyor.

## 3. ALS-MF + kNN karışımı zorlama mı? — Hayır, ama çerçeveleme önemli

### Neden zorlama DEĞİL

- **Ensemble/karışım, öneri sistemlerinde standarttır.** Netflix Prize'ı kazanan
  çözüm onlarca modelin karışımıydı; iki bileşenli karışım son derece mütevazı.
- **Ağırlık ayarı bile gerekmiyor:** ablasyonumuz ayarsız 0.5/0.5'in ayarlıya
  eşit olduğunu gösterdi (0.7201 vs 0.7194). Yani "ayar sömürüsü" yok.
- **Her bileşenin ablasyonu var:** cmean, bias, cknn, cmf ve tüm ikili/üçlü
  kombinasyonları test edildi; gereksizler (cmean, bias) elendi.
- **Bileşen sayısı aslında 2** — pipeline'ın geri kalanı (NMF uzayı, kapasite
  kısıtı, soft havuz) tahminci değil, **kümeleme protokolünün** parçası.

### Asıl risk: katkının dağılması

Dürüst uyarı: sistemin çok parçalı görünmesi, "asıl katkı ne?" sorusunu
zorlaştırabilir. Çözüm **çerçeveleme**:

> Tezin katkısı yeni bir tahminci önermek DEĞİL. Katkı:
> (a) meta-sezgisel kümeleme için **adil karşılaştırma protokolü** (eşit havuz
> bütçesi, kapasite kısıtı, NFE eşitliği, çok-seed istatistik),
> (b) bu protokolde **algoritma seçiminin ölçülebilir etkisinin** gösterilmesi
> (AVOA, 4 rakibini ve K-means++'ı anlamlı geçiyor; komşu-recall'de 2 kat),
> (c) tahmin katmanının sonuçları nasıl belirlediğinin ortaya konması
> (küme-ortalaması → kNN geçişi %9–11).
>
> Tahminci (kNN + MF karışımı) **yeni değil, literatürden alınmış güçlü bir
> standart**; onu kullanma sebebimiz zayıf tahmincinin algoritma farkını
> gizlemesi.

Bu çerçevede "çok yöntem" eleştirisi kendiliğinden düşer: yöntemlerin çoğu
**kontrol** amaçlı (adil kıyas için gerekli), yenilik iddiası taşımıyor.

## 4. Savunmada kullanılacak üç cümle

1. *"kNN'i seçtik çünkü Dacrema ve ark. (2019) iyi ayarlanmış kNN'in birçok
   modern yöntemi geçtiğini göstermiştir; ayrıca kümesiz kNN sonucumuz (0.7467)
   Surprise kütüphanesinin yayınlanmış değeriyle (0.750) uyuşarak ölçüm
   hattımızı doğrulamaktadır."*
2. *"Karışım ağırlığı ayarlanmadan (0.5/0.5) da aynı sonuç alınmaktadır;
   dolayısıyla performans hiperparametre sömürüsünden gelmemektedir."*
3. *"Tahminci katmanı katkı iddiamız değildir; literatürdeki standart bileşenlerden
   kurulmuştur. Katkımız, algoritma karşılaştırmasının adil yapılabildiği
   protokol ve bu protokolde elde edilen bulgulardır."*

## 5. Sadeleştirme önerisi (isterseniz)

Eğer yine de sadeleştirmek isterseniz, **tek bileşene inmek mümkün**:
küme-içi kNN tek başına (ML-1M K=40: 0.7335) hâlâ küme-ortalamasından (0.807)
çok iyi ve AVOA–B0 farkı orada da korunuyor. MF'i "opsiyonel iyileştirme"
olarak ek bölüme alabilirsiniz.

**Ama önermiyorum:** MF karışımı 0.7335 → 0.7201 kazandırıyor (%1.8) ve
literatür kıyasında (SVD 0.6909) bizi rekabetçi tutan şey bu.
