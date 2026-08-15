
# Ablasyon Merdiveni — "Etrafından Dolaşma" Eleştirisinin Cevabı

## Sorun

Şimdiye kadarki karşılaştırma şöyleydi: *"literatür 0.685 diyor, bizim sistem
0.707 veriyor."* Ama biz aynı anda **beş şeyi** değiştirmiştik (uzay, fitness,
atama, havuz, tahminci). Bu haliyle bir eleştirmen haklı olarak şunu söyler:

> "Siz algoritma karşılaştırması yapmıyorsunuz, pipeline karşılaştırması
> yapıyorsunuz — ve pipeline'ı kendi sonucunuzu destekleyecek şekilde kurdunuz."

## Çözüm: her basamakta TEK bir şey değiştir, algoritma kıyasını TEKRARLA

Eğer meta avantajı **her basamakta** görülüyorsa, bulgu pipeline'a bağlı değildir.
Eğer yalnız son basamakta çıkıyorsa, eleştiri haklıdır. Ölçtük.

---

## Sonuç tablosu (ML-100K, K=40, fold 1, seed 42, NFE=400)

| Basamak | Ne eklendi | B0 MAE | AVOA MAE | **Fark** | Havuz | Küme max/min |
|---|---|---|---|---|---|---|
| **S0** literatür | ham matris + WCSS + serbest + küme-ort. | 0.8670 | 0.8670 | **0.0000** | 259 | 220/1 |
| **S1** | + NMF-20 uzayı | 0.9009 | 0.9009 | **0.0000** | 133 | 186/1 |
| **S2** | + tahmine hizalı fitness | 0.9009 | 0.8287 | −0.0722 ⚠ | **937** | **941/0** |
| **S3** | + kapasiteli onarım | 0.9122 | 0.9129 | +0.0007 | 47 | 24/11 |
| **S4** | + soft top-2 havuz | 0.8904 | 0.8769 | **−0.0135** | 47 | 24/11 |
| **S5** | + küme-içi kNN | 0.8409 | 0.8175 | **−0.0234** | 47 | 24/11 |
| **S6** | + küme-MF karışımı (tam) | 0.8027 | 0.7919 | **−0.0108** | 47 | 24/11 |

---

## Okunuş — beş bulgu

### 1. S0 ve S1'de meta HİÇ fark yaratmıyor (fark tam 0.0000)

Sebebi iki katlı:
- **WCSS fitness'ında K-means++ zaten optimaldir** (Lloyd kapalı-form çözüyor);
  meta-sezgisel sıcak başlangıçtan iyileştirme bulamıyor.
- **S0'da arama uzayı 40×1682 = 67.280 boyut**; 400 NFE ile hiç hareket
  edilemiyor. (Literatürün kurulumu tam olarak bu.)

→ **Literatürdeki "algoritmalar birbirine yakın çıkıyor" gözlemi bir bulgu değil,
kurulumun sonucu.** Aynı kurulumda biz de sıfır fark ölçüyoruz.

### 2. S2'de "büyük iyileşme" var ama SAHTE

AVOA 0.8287'ye iniyor (−0.072) — ama küme dağılımına bakın: **max 941, min 0**,
havuz 937 (%99). Yani kümelemeyi iptal etti. Bu, kısıtsız doğruluk hedefinin
dejenerasyon davranışı.

→ **Küme dağılımı raporlanmasa bu satır "yöntemimiz %8 iyileştirdi" diye
sunulabilirdi.** Literatür eleştirimizin en somut örneği bu satırdır.

### 3. S3'te kısıt dejenerasyonu kesiyor — ve fark sıfırlanıyor

Kapasite eklenince AVOA'nın sahte avantajı kayboluyor (+0.0007, B0 hafif önde).
Bu **dürüstlüğümüzün kanıtı**: kısıt bizim lehimize çalışan bir hile değil;
tam tersine, sahte kazancı yok ediyor.

### 4. S4–S6: gerçek ve tutarlı avantaj

Kısıt varken, tahmin katmanı iyileştikçe **meta avantajı ortaya çıkıyor ve
kalıcı oluyor**: −0.0135 → −0.0234 → −0.0108. Havuz sabit (47), küme dağılımı
sabit (24/11) — yani **fark yalnızca merkez kalitesinden geliyor.**

### 5. Mutlak MAE'nin S1/S3'te kötüleşmesi bilgi veriyor

S0 → S1: 0.8670 → 0.9009 (NMF sıkıştırması küme-ortalaması için bilgi kaybı)
S2 → S3: 0.8287 → 0.9122 (dejenerasyon kaybı = kısıtın maliyeti)

Bu iki artış, **bizim eklediğimiz bileşenlerin bedava olmadığını** gösteriyor.
Kazanç yalnız S4–S6'da, yani tahmin katmanında geliyor. Şeffaflık açısından
değerli: neyi neye ödediğimiz açık.

---

## Bu tablo eleştiriyi nasıl kapatıyor?

| Eleştiri | Cevap |
|---|---|
| "Pipeline'ı kendinize göre kurdunuz" | Aynı pipeline'da B0 da koşuyor; fark her basamakta B0'a karşı ölçülüyor |
| "Sonucu iyileştirmek için etrafından dolaştınız" | S3'te eklediğimiz kısıt kendi avantajımızı **yok ediyor** (−0.072 → +0.0007) |
| "Literatürden iyi olmanız pipeline'dan" | Doğru — ve bunu kendimiz gösteriyoruz (S0→S6 mutlak fark). Ama **algoritma iddiası** her basamakta ayrı ölçülüyor |
| "Meta avantajı sadece son kurulumda çıkıyor" | S4, S5, S6'da ayrı ayrı çıkıyor; üç farklı tahminci ile |

**Ana mesaj cümlesi:**

> "Bileşenlerin katkısı kademeli ablasyonla ayrıştırılmıştır. Meta-sezgisel
> merkez optimizasyonunun avantajı, literatürün kurulumunda (WCSS uygunluk
> fonksiyonu, ham puan uzayı) ölçülemez düzeydedir; tahmine hizalı uygunluk
> fonksiyonu eklendiğinde ortaya çıkan iyileşme ise kümelemenin dejenerasyonundan
> kaynaklanmakta (bir kümede 941 kullanıcı, 39 küme boş), kapasite kısıtı
> eklendiğinde kaybolmaktadır. Gerçek ve tutarlı avantaj yalnızca kısıt altında
> ve keskinleştirilmiş tahmin katmanıyla birlikte gözlenmektedir (S4–S6:
> −0.0108 ile −0.0234 arası)."

---

## Kalan iş: merdiveni çok-seed ve çok-algoritma yapmak

Şu an tek seed, tek meta (AVOA). Tam hali:

```
py -3.12 algo_selection_v2\ablasyon_merdiveni.py --k 40 --tum --seeds 42 43 44 --max-fe 600 --resume
py -3.12 algo_selection_v2\ablasyon_merdiveni.py --k 6  --tum --seeds 42 43 44 --max-fe 600 --resume
```

Beklenti: S0/S1'de tüm metalar B0 ile aynı, S4–S6'da hepsi B0'ın üstünde,
AVOA sıralamada birinci. Bu, hem "pipeline değil algoritma" hem "AVOA seçimi"
iddialarını aynı tabloda mühürler.

**Bu tablo makalenin en önemli tablosu olacak** — yöntem bölümünün de gerekçesi
buradan türetilecek (her bileşen neden var, ne kazandırıyor, neye mal oluyor).
