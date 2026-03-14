# Matrix Factorization: Örnek Üzerinden Baştan Sona Anlatım

Bu dokümanda, matrix factorization kavramları **küçük sayılar ve matrixler** üzerinden adım adım anlatılıyor. Türev, loss, lokal/global minimum ve optimizasyon hep bu örnek üzerinden açıklanıyor.

---

## 1. Örnek veri: Rating matrix R

Diyelim ki **3 kullanıcı** ve **4 film** var. Kullanıcılar bazı filmlere 1–5 puan vermiş; vermedikleri yerler boş (bilinmiyor).

**Rating matrix R** (satır = kullanıcı, sütun = film):

```
         Film1  Film2  Film3  Film4
Kullanıcı1   5      ?      4      ?
Kullanıcı2   ?      3      ?      2
Kullanıcı3   4      ?      ?      3
```

`?` = o kullanıcı o filmi puanlamamış. Amacımız: bu boş hücreleri **tahmin etmek** (ör. Kullanıcı1–Film2 için ne puan verirdi?).

Sayısal olarak, sadece **gözlenen** rating’leri kullanıyoruz. Örnekte 6 tane var:

| Kullanıcı | Film | Gerçek rating (r) |
|-----------+------+--------------------|
| 0         | 0    | 5                  |
| 0         | 2    | 4                  |
| 1         | 1    | 3                  |
| 1         | 3    | 2                  |
| 2         | 0    | 4                  |
| 2         | 3    | 3                  |

---

## 2. Model: R ≈ U · V^T

Matrix Factorization’da R’yi iki küçük matrisin çarpımı gibi düşünüyoruz:

- **U**: kullanıcı matrisi, boyut (3 kullanıcı × 2 faktör)  
- **V**: film matrisi, boyut (4 film × 2 faktör)  
- **Tahmin matrisi**: R_hat = U · V^T → boyut (3 × 4)

Burada **latent_dim = 2** seçiyoruz (faktör sayısı); örnek küçük kalsın diye.

### Örnek U ve V (rastgele başlangıç)

```
U (3×2):                    V (4×2):
        Faktör0  Faktör1           Faktör0  Faktör1
Kullanıcı0  0.5    0.3     Film0    0.4    0.2
Kullanıcı1  0.2   -0.1     Film1    0.3    0.5
Kullanıcı2 -0.2    0.4     Film2    0.1    0.3
                            Film3   -0.1    0.2
```

### Tahmin nasıl yapılıyor?

**r_ui (tahmin)** = Kullanıcı u’nun satırı ile Film i’nin satırının **iç çarpımı**.

Örnek:
- Kullanıcı0, Film0: (0.5)(0.4) + (0.3)(0.2) = 0.20 + 0.06 = **0.26**
- Kullanıcı0, Film2: (0.5)(0.1) + (0.3)(0.3) = 0.05 + 0.09 = **0.14**

Tüm matris:

**R_hat = U · V^T** (tahmin edilen rating’ler):

```
         Film0   Film1   Film2   Film3
Kullanıcı0  0.26   0.30    0.14    0.01
Kullanıcı1  0.05   0.01   -0.04   -0.07
Kullanıcı2  0.04   0.10    0.10    0.06
```

Gerçek R’deki gözlenen değerler: 5, 4, 3, 2, 4, 3. Tahminlerimiz (0.26, 0.14, 0.01, -0.07, 0.04, 0.06) çok uzak — henüz **kötü bir U,V** kullandık. Bu yüzden **optimizasyon** yapacağız: U ve V’yi değiştirerek tahminleri gerçeğe yaklaştıracağız.

---

## 3. Loss (kayıp) ne demek? — Matrix üzerinden

**Loss** = “Tahminlerimiz gerçek rating’lerden ne kadar uzak?”ın sayısal ölçüsü.

Sadece **gözlenen** (u, i, r) üçlülerinde hesaplıyoruz. Örnekte 6 tane var.

**MSE (Ortalama Kare Hata)** kullanıyoruz:

- Her gözlenen (u, i, r) için: **hata² = (r − tahmin)²**
- Loss = bu hataların ortalaması

Yukarıdaki U,V ile (sayıları yuvarlayarak):

| (u,i) | Gerçek r | Tahmin | Hata (r − tahmin) | Hata²   |
|-------|----------|--------|-------------------|---------|
| (0,0)  | 5        | 0.26   | 4.74              | 22.47   |
| (0,2)  | 4        | 0.14   | 3.86              | 14.90   |
| (1,1)  | 3        | 0.01   | 2.99              | 8.94    |
| (1,3)  | 2        | -0.07  | 2.07              | 4.28    |
| (2,0)  | 4        | 0.04   | 3.96              | 15.68   |
| (2,3)  | 3        | 0.06   | 2.94              | 8.64    |

**Loss (MSE)** ≈ (22.47 + 14.90 + 8.94 + 4.28 + 15.68 + 8.64) / 6 ≈ **12.49**.

- **Loss büyük** → tahminler kötü (matrix’teki sayılar gerçekten uzak).  
- **Loss küçük** → tahminler iyi (R_hat, R’ye yakın).  
- **Optimizasyonun amacı** = U ve V’yi öyle seçmek ki **loss küçülsün**.

---

## 4. Optimizasyon ne demek? — Matrix bağlamında

- **Parametreler:** U ve V’deki tüm sayılar (3×2 + 4×2 = 14 sayı).  
- **Hedef:** Bu 14 sayıyı değiştirerek **loss’u mümkün olduğunca küçültmek**.

Her farklı (U, V) çifti farklı bir R_hat ve dolayısıyla farklı bir loss verir.  
“İyi çözüm” = loss’u düşüren bir (U, V).

---

## 5. Türev (gradyan) neden hesaplanıyor? — Matrix örneğiyle

Türev bize şunu söyler: **“Şu anki U ve V’de, hangi parametreyi biraz artırırsak / azaltırsak loss artar veya azalır, ve ne kadar?”**

### Tek parametre örneği

Sadece **U[0,0]** (Kullanıcı0, Faktör0) değişsin, diğerleri sabit kalsın.

- U[0,0] = 0.5 iken loss = 12.49 (yukarıdaki gibi).  
- U[0,0] = 0.6 yaparsak tahminler (özellikle Kullanıcı0’ın tüm film tahminleri) değişir → loss başka bir değer alır.  
- U[0,0] = 0.4 yaparsak yine loss değişir.

**Türev (bu parametreye göre)** = “U[0,0]’ı çok az artırırsak loss yaklaşık ne kadar değişir?”  
- Türev **negatif** ise: U[0,0]’ı **artırmak** loss’u **azaltır** → o yönde gideriz.  
- Türev **pozitif** ise: U[0,0]’ı **azaltmak** loss’u azaltır → ters yönde gideriz.

Gradyan = **tüm** parametreler (U ve V’deki her sayı) için bu “hangi yönde değiştirmeliyim?” bilgisinin bir vektörü.  
Yani: “U ve V’deki her bir sayıyı **biraz** değiştirirken, loss’u **en hızlı azaltan** toplu yön” gradyanın verdiği yöndür.

### Matrix tarafı

- U ve V’yi **gradyan yönünün tersine** küçük adımlarla güncellersek → R_hat = U·V^T değişir → gözlenen hücrelerdeki hatalar (r − tahmin) azalır → **loss düşer**.  
- Türev **hesaplanmazsa** hangi sayıyı artırıp azaltacağımızı bilemeyiz; türev sayesinde **bilinçli** bir güncelleme yaparız.

**Özet (matrix diliyle):**  
Türev = “Mevcut U ve V ile R_hat’i nasıl değiştirirsek loss azalır?” bilgisi. Bu bilgiyle U ve V’yi adım adım güncelleyerek R_hat’i R’ye (gözlenen hücrelerde) yaklaştırıyoruz.

---

## 6. Lokal minimum ne demek? — Matrix üzerinden

**Minimum** = loss’un (neredeyse) daha da düşmediği bir (U, V) noktası.  
Yani küçük oynatmalarla U,V’yi değiştirince loss ya aynı kalıyor ya artıyor; “dibe” varmışız.

- **Lokal (yerel) minimum:** Sadece “etrafımızdaki” en iyi nokta. Yani mevcut (U,V)’den küçük adımlarla hareket edince loss düşmüyor; ama uzakta, başka bir (U,V) bölgesinde **daha düşük** loss olan başka bir “dip” olabilir.  
- **Global minimum:** Tüm mümkün (U,V) uzayında loss’un **en düşük** olduğu nokta.

### Örnek (kavramsal)

- **A noktası (U1, V1):** Loss = 0.8. Etrafında küçük oynatınca loss artıyor → **lokal minimum**.  
- **B noktası (U2, V2):** Loss = 0.3. Yine etrafında oynatınca loss artıyor; ve tüm uzayda en düşük loss bu → **global minimum**.

SGD gibi yöntemler “aşağı iner” ve genelde **ilk rastladıkları** bir lokal minimumda durur. Başlangıç (U,V) farklı olsaydı başka bir lokal minimuma (belki daha iyi, belki daha kötü) düşebilirdi.

**Matrix açısından:**  
- Lokal minimumdaki (U,V) ile R_hat = U·V^T, **o bölgeye göre** en iyi tahminleri verir; yani **işe yarar** bir öneri sistemi elde ederiz.  
- Daha iyi bir (U,V) (ör. global minimuma daha yakın) bulursak, aynı R matrisi için **daha düşük loss** ve dolayısıyla **daha doğru** R_hat elde ederiz.

---

## 7. Lokal / global minimum bize ne katkı sağlıyor? — Örnekle

- **Lokal minimum bulmak:**  
  O (U,V) ile R_hat’i hesaplıyoruz. Gözlenen hücrelerde hata makul seviyede; boş hücreler için de tahminlerimiz anlamlı. Yani **çalışan bir tahmin matrisi** ve öneri sistemi.

- **Daha iyi bir minimum (ör. global’e yakın) bulmak:**  
  Aynı R için daha düşük loss → tahminler gerçek rating’lere daha yakın → **daha isabetli** öneriler (daha iyi RMSE/MAE).

Sayısal örnek (kavramsal):

| (U,V) tipi      | Loss (örnek) | Anlamı (matrix)                          |
|------------------|-------------|-------------------------------------------|
| Rastgele başlangıç | 12.49       | R_hat çok uzak (yukarıdaki gibi)         |
| Bir lokal minimum  | 0.5        | R_hat gözlenen hücrelerde R’ye yakın     |
| Daha iyi minimum   | 0.2        | R_hat daha da yakın; daha iyi tahminler  |

Yani: Minimum bulmak = **loss’u düşürmek** = **R_hat’i R’ye (gözlenen yerlerde) yaklaştırmak** = daha iyi tahmin ve öneri kalitesi.

---

## 8. Özet: Örnek üzerinden zincir

1. **R** = gözlenen rating matrix’i (boşluklar var).  
2. **R_hat = U·V^T** = tahmin matrix’i; U ve V’yi değiştirerek R_hat’i değiştiriyoruz.  
3. **Loss** = sadece gözlenen (u,i,r) hücrelerinde (r − tahmin)² ortalaması; ne kadar büyükse tahminler o kadar kötü.  
4. **Optimizasyon** = U,V’yi seçerek loss’u küçültmek.  
5. **Türev (gradyan)** = “U ve V’deki hangi sayıları hangi yönde değiştirirsek loss azalır?” bilgisi; bu yüzden hesaplanıyor.  
6. **Lokal minimum** = “Bu (U,V) etrafında” loss’un dibe vurduğu nokta; bize **çalışan bir R_hat** verir.  
7. **Global (veya daha iyi) minimum** = daha düşük loss = **daha doğru R_hat** ve daha iyi öneriler.

Bu örnekteki 3×4 R ve 2 faktörlü U,V, gerçek projedeki büyük matrix’lerin aynı mantıkla çalışan küçük bir modelidir; formüller ve amaç aynı, sadece boyutlar büyür.
