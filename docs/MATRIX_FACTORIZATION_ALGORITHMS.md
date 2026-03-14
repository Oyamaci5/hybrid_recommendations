# Matrix Factorization ve Algoritmalar — Teknik Doküman

Bu doküman, projedeki **düz matrix factorization** yapısını, kullanılan **optimizasyon algoritmalarını**, **artı/eksilerini**, **nasıl kullanıldıklarını** ve **parametrelerin anlamlarını** açıklar.

---

## 1. Ortak Model: Matrix Factorization (MF)

Tüm yöntemler aynı **MF modelini** kullanır: `models/mf_model.py` — **MFModel**.

### Ne yapar?

- **R** (kullanıcı × ürün rating matrisi) çoğunlukla boştur.
- Amaç: **R ≈ U · V^T** ile düşük ranklı tahmin.
  - **U**: kullanıcı latent vektörleri, boyut `(n_users, latent_dim)`
  - **V**: ürün latent vektörleri, boyut `(n_items, latent_dim)`
- Tahmin: **r_ui = U[u] · V[i]** (iç çarpım).
- Kayıp: Gözlenen rating’ler üzerinde **MSE + L2 regularizasyon**.

Bu model collaborative filtering (işbirlikçi filtreleme) ile kullanıcı–ürün puanlarını tahmin etmek için kullanılır; değerlendirme **RMSE** ve **MAE** ile yapılır (`core/metrics.py`).

---

## 2. Başlangıç ve ince ayar ne demek?

Hibrit yöntemlerde (ör. **COA-SGD-MF**) iki aşama vardır: **başlangıç** ve **ince ayar**. Kısaca anlamları:

### Başlangıç (initialization)

- **Ne demek:** U ve V matrislerine **ilk değerleri vermek** — yani optimizasyonun **hangi noktadan** başlayacağını belirlemek.
- **MF-SGD’de:** Başlangıç **rastgele** (küçük rastgele sayılar). Bu yüzden bazen kötü bir bölgeye düşüp yerel minimumda kalabilir.
- **COA-SGD-MF / HHO-SGD-MF’te:** Başlangıç **COA veya HHO** ile aranır. Bu algoritmalar birçok aday (coati/şahin) deneyerek “loss’u düşüren” U,V değerlerini bulur. Yani **rastgele yerine, bilinçli bir ilk nokta** seçilir.
- **Özet:** Başlangıç = “Eğitime hangi U,V değerleriyle başlıyoruz?”

### İnce ayar (refinement)

- **Ne demek:** Bu **ilk noktadan** hareketle U ve V’yi adım adım güncelleyip **loss’u daha da küçültmek**. Yani bulunan çözümü **iyileştirmek**.
- **COA-SGD-MF’te:** İnce ayar **SGD** ile yapılır. Gradyanlara bakılarak U,V küçük adımlarla güncellenir; böylece tahmin hatası (MSE) azalır.
- **COA-KMeans-MF’te:** İnce ayar **K-Means** ile yapılır (embedding’ler küme merkezlerine doğru çekilir).
- **Özet:** İnce ayar = “Başlangıç noktasını alıp daha iyi bir çözüme getiriyoruz.”

### COA-SGD-MF özeti

1. **Başlangıç (COA):** Rastgele U,V yerine, COA ile loss’u düşüren bir U,V bulunur → model bu değerlerle başlatılır.  
2. **İnce ayar (SGD):** Bu U,V üzerinde SGD çalıştırılır → tahminler daha da iyileştirilir.

Böylece hem **iyi bir başlangıç** (COA) hem de **hızlı ve hedefli iyileştirme** (SGD) bir arada kullanılmış olur.

---

## 3. Sezgisel yöntemler ne demek?

Bu bölümde **sezgisel (metaheuristic) yöntemler** kavramı, hiç bilmeyen biri için adım adım açıklanır.

### Optimizasyon nedir?

- Elimizde bir **hedef** var: U ve V matrislerini öyle seçmek ki, tahminlerimiz gerçek rating’lere mümkün olduğunca yakın olsun.
- Bunu sayısal olarak **kayıp (loss)** ile ölçüyoruz: tahmin yanlışsa loss büyür, doğruysa küçülür.
- **Optimizasyon** = loss’u **küçültmek** için U ve V’yi değiştirme süreci. Yani “en iyi” veya “yeterince iyi” U,V’yi **aramak**.

Bu aramayı yapmanın iki büyük ailesi vardır: **gradyan tabanlı** yöntemler ve **sezgisel** yöntemler.

---

### Türev neden hesaplanıyor? Lokal / global minimum bize ne katkı sağlıyor?

#### Türev (gradyan) neden hesaplanıyor?

- U ve V binlerce sayıdan oluşuyor. Loss’u küçültmek için **hangi sayıyı artırıp azaltacağımızı** bilmemiz lazım.
- **Türev** tam da bunu söyler: “Şu anki U,V’de, her bir parametreyi biraz değiştirirsen loss **hangi yönde** ne kadar değişir?”  
  Yani **yön bilgisi**: “Bu yöne gidersen loss artar, ters yöne gidersen loss azalır.”
- **Gradyan** = tüm parametreler için bu yönlerin bir araya gelmesi. Matematiksel olarak loss’u **en hızlı artıran** yönü verir. SGD ise **gradyanın tersi** yönde adım atar → loss’u o adımda mümkün olduğunca hızlı **azaltır**.
- **Özet:** Türev hesaplanıyor çünkü “**Nereye doğru adım atmalıyım?**” sorusuna cevap veriyor. Türev olmadan yönü tahmin etmek gerekir (deneme-yanılma); türevle tek bir formülle en iyi yön bulunur ve adım atılır.

#### Lokal minimum bulmak bize ne katkı sağlıyor?

- **Minimum** = loss’un (neredeyse) artık düşmediği bir U,V noktası. O noktada tahminlerimiz **o bölgeye göre** en iyi hale gelmiş demektir.
- **Lokal (yerel) minimum** = etrafındaki her yönden daha derin bir “çukur”; yani küçük bir hareketle loss’u daha da düşüremiyoruz. SGD bu tür noktalarda durur.
- **Katkısı:** O lokal minimumdaki U,V ile **tahminler zaten rastgelelikten çok daha iyi**. Yani kullanıcı–ürün puanları makul bir doğrulukla tahmin edilir; öneri sistemi **işe yarar** hale gelir. Lokal minimum bulmak = “en azından iyi bir çözüm” bulmak.

#### Global minimum bulmak (veya daha iyi bir minimuma yaklaşmak) bize ne katkı sağlıyor?

- **Global minimum** = tüm uzayda loss’un **en düşük** olduğu nokta. O U,V ile teorik olarak bu model ve bu kayıp fonksiyonuyla **mümkün olan en iyi tahminler** yapılır.
- **Katkısı:** Daha düşük loss = tahminler gerçek rating’lere **daha yakın** = **daha doğru öneriler**. Kullanıcıya daha isabetli film/ürün önerisi, daha iyi RMSE/MAE.
- Pratikte global minimumu **garantiyle** bulmak çoğu zaman mümkün değil (özellikle büyük MF’te). Ama **daha iyi bir lokal minimum** bulmak (örneğin COA/HHO ile başlayıp SGD ile ince ayar) yine de bize **daha düşük loss** ve dolayısıyla **daha iyi tahmin kalitesi** kazandırır.

**Kısa özet:**

| Hedef | Ne anlama geliyor? | Bize katkısı |
|--------|---------------------|--------------|
| **Türev hesaplamak** | “Hangi yöne adım atarsam loss azalır?” bilgisini vermek | Doğru yöne adım atıp loss’u hızlı ve bilinçli düşürmek |
| **Lokal minimum bulmak** | O bölgede loss’u minimize etmek | İşe yarar, iyi tahminler; öneri sistemi çalışır |
| **Global (veya daha iyi) minimum** | Mümkün olan en düşük veya daha düşük loss | Daha doğru tahminler, daha iyi öneriler (daha düşük RMSE/MAE) |

Yani: Türev **yön** veriyor; minimum (lokal veya daha iyi) **düşük loss** veriyor; düşük loss da **daha iyi rating tahmini ve öneri kalitesi** demek.

---

### Gradyan tabanlı yöntem (SGD) nasıl çalışır?

- **Gradyan**, bir noktada “en dik yokuşun yönü” gibidir: loss’u **en hızlı artıran** yön. SGD ise tam tersine, **en hızlı azaltan** yöne küçük adımlar atar.
- Yani: “Şu anki U,V’den loss’u azaltmak için hangi yöne gitmeliyim?” sorusuna, matematiğin (türev) verdiği cevaba göre hareket eder.
- **Artısı:** Adım adım çok verimli; genelde hızlı iyileşir.
- **Eksisi:** Sadece **bulunduğu yere göre** hareket eder. Eğer loss yüzeyinde birçok “çukur” (yerel minimum) varsa, SGD **en yakın çukura** düşer ve orada kalır. Daha iyi bir çukur başka yerde olsa bile, oraya **kendi başına** gitmez; çünkü o noktadan “yukarı çıkıp” başka çukura geçmek gradyan mantığına aykırı.

Özet: SGD **yerel** bir arama yapar — “buradan en iyi nereye inerim?” diye bakar.

---

### Sezgisel (metaheuristic) yöntem ne demek?

- **Sezgisel** = türev (gradyan) kullanmadan, **deneme–yanılma ve kurallara** dayalı bir arama.
- **Metaheuristic** = genel bir “nasıl arayalım?” stratejisi; belirli bir probleme özel formül yerine, **keşif + sömürü** dengesi kuran, çoğu zaman doğadaki davranışlardan esinlenen algoritmalar.
- Bu yöntemler **birçok aday çözüm** (parçacık, coati, şahin vb.) tutar. Her aday bir U,V setidir. Adım adım bu adaylar hareket eder; “en iyi” bulunan çözüm güncellenir; diğerleri bazen rastgele, bazen en iyisine doğru, bazen birbirine göre konum alır.
- **Gradyan kullanmazlar:** “Şu yöne git” demek için türev hesaplamazlar. Sadece “bu U,V’nin loss’u ne?” diye **değer** bakarlar; hareket kuralları formüllere (ve bazen rastgeleliğe) dayanır.

Özet: Sezgisel yöntemler **geniş alan tarar**, birçok noktayı deneyerek “iyi bölgeler” bulmaya çalışır; tek bir noktadan türevle inen SGD’den farklıdır.

---

### Neden sezgisel yöntem kullanıyoruz?

- MF’te loss yüzeyi **çok boyutlu** ve genelde **birçok yerel minimum** (küçük çukur) içerir.
- **Sadece SGD** kullanırsak: Rastgele bir başlangıçtan ineriz; hangi çukura düştüysek orada kalırız. Bazen o çukur “kötü” bir çukur olur; yani daha iyi bir çukur (daha düşük loss) başka yerde vardır ama SGD oraya ulaşamaz.
- **Sezgisel yöntemler** (COA, HHO, PSO): Aynı anda birçok noktayı deneyerek uzayda **gezinir**. Böylece “daha iyi bir çukur” bulma şansı artar. Bu yüzden projede bunları **başlangıç** için kullanıyoruz: önce sezgisel yöntem “iyi bir bölge” bulsun, sonra SGD o bölgede **ince ayar** yapsın.

Basit benzetme:
- **SGD:** Tek kişi, bulunduğu yerden hep aşağı iniyor; en yakın vadiye düşüp orada kalıyor.
- **Sezgisel:** Birçok kişi dağda farklı yerlerde; en alçak noktayı bulanı takip edip oraya doğru da hareket ediyorlar; böylece daha iyi bir vadi bulma ihtimali artıyor.

---

### Projedeki sezgisel yöntemler

- **COA (Coati Optimization Algorithm):** Coati (rakun benzeri hayvan) sürüsünün avlanma ve predatörden kaçma davranışından esinlenir. Popülasyondaki her “coati” bir U,V adayıdır; kurallar keşif (farklı bölgeler) ve sömürü (en iyi etrafında daralma) dengesini kurar.
- **HHO (Harris Hawks Optimization):** Şahinlerin tavşan kovalamasından esinlenir. Her “şahin” bir U,V adayıdır; “kaçış enerjisi” ile keşif/sömürü geçişi kontrol edilir; bazen Levy uçuşu gibi rastgele büyük adımlar atılır.
- **PSO (Particle Swarm Optimization):** Parçacık sürüsü. Her parçacık bir U,V’dir; hem “kendi en iyi bulduğu” konuma hem de “sürünün en iyisi”ne doğru çekilir; atalet ile momentum korunur.

Hepsi **aynı amaca** hizmet eder: Gradyan kullanmadan, birçok adayla uzayı tarayıp **loss’u düşüren** U,V bölgeleri bulmak. Projede COA ve HHO **sadece başlangıç** için, PSO ise (PSO-MF’te) **tüm eğitim** için kullanılır.

---

### Sezgisel vs gradyan: kısa karşılaştırma

| | Gradyan tabanlı (SGD) | Sezgisel (COA, HHO, PSO) |
|--|------------------------|---------------------------|
| **Bilgi** | Türev (yön) kullanır | Sadece loss değeri; türev yok |
| **Arama** | Tek noktadan, yerel adımlar | Çok nokta (popülasyon), geniş tarama |
| **Yerel minimum** | Takılabilir | Farklı bölgeleri deneyerek kaçınma şansı |
| **İnce ayar** | Çok uygun (küçük adımlar) | Genelde yavaş veya hassas değil |
| **Maliyet** | Her adımda az hesaplama | Popülasyon × loss değerlendirme |

Bu yüzden **hibrit** mantığı kullanıyoruz: **Sezgisel = iyi başlangıç**, **SGD = hızlı ince ayar**.

---

## 4. Düz (Klasik) Matrix Factorization: MF-SGD

**Dosyalar:** `models/mf_model.py`, `optimizers/sgd.py`, `methods/mf_sgd.py`

### Nasıl kullanılır?

- **Tek faz:** Rastgele başlatılan U, V üzerinde **sadece SGD** ile eğitim.
- Baseline deneyleri: `experiments/run_baseline.py`.

### Parametreler

| Parametre | Varsayılan | Anlamı |
|-----------|------------|--------|
| `n_users` | — | Kullanıcı sayısı. |
| `n_items` | — | Ürün sayısı. |
| `latent_dim` | 10 | Latent faktör sayısı (k). Tahmin kalitesi vs. hesaplama dengesi. |
| `learning_rate` | 0.01 | SGD adım büyüklüğü. Büyük → hızlı ama kararsız; küçük → yavaş ama stabil. |
| `regularization` | 0.01 | L2 (lambda). Aşırı öğrenmeyi azaltır; büyük → daha güçlü ceza. |
| `random_seed` | 42 | Tekrarlanabilirlik için seed. |
| `n_iterations` | 100 | SGD epoch sayısı. |

### Artılar (+)

- Basit, anlaşılır, endüstride yaygın.
- Her (u,i,r) için sadece ilgili satır/sütun güncellenir; bellek ve hesaplama verimli.
- Hiperparametre sayısı az; ayarlaması nispeten kolay.
- Gradyan tabanlı; teorik temeli net.

### Eksiler (-)

- Rastgele başlangıç; kötü yerel minimuma takılabilir.
- Learning rate ve iterasyon sayısına duyarlı.
- Çok büyük veride tek tek örnek güncellemesi yavaş kalabilir (mini-batch kullanılmıyor).

---

## 5. SGD Optimizer (Stochastic Gradient Descent)

**Dosya:** `optimizers/sgd.py` — **SGDOptimizer**

### Nasıl kullanılır?

- **Tek başına:** MF-SGD’de tek optimizer (düz MF).
- **Hibrit:** COA-SGD-MF ve HHO-SGD-MF’te **2. faz** olarak; metaheuristik ile bulunan U,V üzerinde **ince ayar**.

### Parametreler

| Parametre | Varsayılan | Anlamı |
|-----------|------------|--------|
| `learning_rate` | 0.01 | Her güncellemede adım büyüklüğü. |
| `regularization` | 0.01 | L2 katsayısı (lambda). |

### Artılar (+)

- Yerel optimizasyon için hızlı ve etkili.
- Gradyan bilgisi kullandığı için ince ayar için doğal seçim.
- Hibrit yapılarda iyi başlangıçla çok iyi sonuç verebilir.

### Eksiler (-)

- Tek başına kullanıldığında başlangıca bağımlı; global arama yapmaz.
- Learning rate çok büyükse titreme, küçükse yavaş yakınsama.

---

## 6. COA (Coati Optimization Algorithm)

**Dosya:** `optimizers/coa.py` — **COAOptimizer**

### Nasıl kullanılır?

- **Sadece başlangıç (initializer)** olarak: U ve V için iyi bir başlangıç noktası arar; eğitimin tamamını COA yapmaz.
- **COA-SGD-MF:** 1) COA ile başlangıç, 2) SGD ile ince ayar.
- **COA-KMeans-MF:** 1) COA ile başlangıç, 2) K-Means ile ince ayar.

### Parametreler

| Parametre | Varsayılan | Anlamı |
|-----------|------------|--------|
| `n_coatis` | 30 | Popülasyondaki coati (birey) sayısı. Çok büyük → yavaş; küçük → çeşitlilik azalır. |
| `regularization` | 0.01 | Fitness’ta L2 cezası. |
| `boundary` | 1.0 | U,V elemanları [-boundary, boundary] aralığında sınırlanır. |
| `n_iterations` | 50 | COA iterasyon sayısı (fit sırasında coa_iterations). |

### Artılar (+)

- Global arama; rastgele başlangıçtan daha iyi bölge bulabilir.
- İki fazlı (iguana avlama + predatörden kaçma) keşif ve sömürü dengesi.
- SGD/K-Means ile birleştirildiğinde daha iyi final loss ve daha hızlı yakınsama potansiyeli.

### Eksiler (-)

- Her birey tam U,V vektörü taşır; boyut büyüdükçe maliyet artar.
- COA tek başına ince ayar için yavaş; bu yüzden sadece initializer olarak kullanılması mantıklı.

---

## 7. HHO (Harris Hawks Optimization)

**Dosya:** `optimizers/hho.py` — **HHOOptimizer**

### Nasıl kullanılır?

- **Sadece başlangıç (initializer)** olarak: COA gibi U,V için iyi başlangıç arar.
- **HHO-SGD-MF:** 1) HHO ile başlangıç, 2) SGD ile ince ayar.
- **HHO-KMeans-MF:** 1) HHO ile başlangıç, 2) K-Means ile ince ayar.

### Parametreler

| Parametre | Varsayılan | Anlamı |
|-----------|------------|--------|
| `n_hawks` | 30 | Şahin (birey) sayısı. Popülasyon büyüklüğü. |
| `escape_energy_initial` | 1.0 | Başlangıç kaçış enerjisi E0. E = 2*E0*(1-t/T) ile azalır; keşif → sömürü geçişini kontrol eder. |
| `regularization` | 0.01 | Fitness’ta L2. |
| `boundary` | 1.0 | Embedding sınırı [-boundary, boundary]. |
| `n_iterations` | 50 | HHO iterasyon sayısı (fit’te hho_iterations). |

### Artılar (+)

- Keşif/sömürü dengesi (E parametresi) ve Levy uçuşu ile çeşitlilik.
- Kötü yerel minimumlardan kaçınmada yardımcı olabilir.
- Deneylerde (convergence_interpretation.txt) HHO başlangıcı ile daha iyi başlangıç loss ve final performans gözlenmiş.

### Eksiler (-)

- COA’ya benzer şekilde yüksek boyutlu U,V için maliyetli.
- E0 ve popülasyon boyutu ayarı gerekebilir.

---

## 8. K-Means Optimizer

**Dosya:** `optimizers/kmeans.py` — **KMeansOptimizer**

### Nasıl kullanılır?

- **Refinement (ince ayar)** olarak; SGD’nin alternatifi.
- U ve V embedding’leri üzerinde K-Means kümeleme yapılır; embedding’ler küme merkezlerine doğru güncellenir.
- **COA-KMeans-MF:** COA başlangıç + K-Means refinement.
- **HHO-KMeans-MF:** HHO başlangıç + K-Means refinement.
- Gradyan kullanmaz; sadece küme merkezlerine çekme.

### Parametreler

| Parametre | Varsayılan | Anlamı |
|-----------|------------|--------|
| `n_clusters_users` | None | Kullanıcı embedding’leri için küme sayısı. None → auto: min(√n_users, n_users). |
| `n_clusters_items` | None | Ürün embedding’leri için küme sayısı. None → auto: min(√n_items, n_items). |
| `learning_rate` | 0.1 | Merkeze doğru hareket katsayısı. U[u] = (1-lr)*U[u] + lr*center. |
| `regularization` | 0.01 | Güncelleme sonrası L2 küçültme. |
| `max_iter` | 100 | Her K-Means çağrısında sklearn KMeans max iterasyonu. |
| `n_iterations` | 100 | Kaç kez “kümele → merkeze çek → loss hesapla” döngüsü yapılacağı. |

### Artılar (+)

- Gradyan gerektirmez; türevsiz refinement seçeneği sunar.
- Benzer kullanıcı/ürünleri gruplayarak yapıyı sadeleştirebilir.
- SGD’ye alternatif; farklı hibrit kombinasyonları araştırmak için uygun.

### Eksiler (-)

- Küme sayısı ve learning rate seçimi önemli; yanlış seçimde loss kötüleşebilir.
- Rating tahmin loss’una doğrudan gradyan inmiyor; SGD kadar hedefli ince ayar olmayabilir.

---

## 9. PSO (Particle Swarm Optimization)

**Dosyalar:** `optimizers/pso.py`, `methods/pso_mf.py` — **PSOOptimizer**, **PSOMF**

### Nasıl kullanılır?

- **Tek faz:** Eğitimin **tamamı** PSO ile yapılır; SGD veya K-Means ikinci faz yok.
- Her parçacık tam bir MF çözümü (U,V düzleştirilmiş) taşır; en iyi parçacık modele yazılır.
- Karşılaştırma ve araştırma amacıyla “tam metaheuristik MF” baseline’ı.

### Parametreler

| Parametre | Varsayılan | Anlamı |
|-----------|------------|--------|
| `swarm_size` | 30 | Parçacık sayısı. |
| `inertia_weight_max` | 0.9 | Başlangıç atalet ağırlığı (w). w yüksek → keşif. |
| `inertia_weight_min` | 0.2 | Son atalet ağırlığı. w düşük → sömürü. w iterasyonla lineer düşer. |
| `cognitive_coeff` (c1) | 2.0 | Bireyin kendi en iyi konumuna çekilmesi. |
| `social_coeff` (c2) | 2.0 | Sürünün global en iyisine çekilmesi. |
| `regularization` | 0.01 | Fitness’ta L2. |
| `boundary` | 1.0 | Konum sınırı. |
| `n_iterations` | 100 | PSO iterasyon sayısı. |

### Artılar (+)

- Global arama; gradyan kullanmadan optimizasyon.
- c1/c2 ve inertia ile keşif–sömürü ayarlanabilir.
- Hibrit olmayan metaheuristik baseline olarak karşılaştırma imkânı.

### Eksiler (-)

- Boyut (n_users*k + n_items*k) büyüdükçe çok yavaş ve bellek yoğun.
- İnce ayar için gradyan kullanmadığından SGD kadar hassas olmayabilir.
- Birçok parametre (swarm_size, w, c1, c2) ayarı gerekir.

---

## 10. Özet Tablo: Yöntemler ve Kullanım

| Yöntem | Dosya | 1. Faz | 2. Faz | Amaç |
|--------|--------|--------|--------|------|
| **MF-SGD** | `methods/mf_sgd.py` | — | SGD | Düz MF, baseline. |
| **COA-SGD-MF** | `methods/coa_sgd_mf.py` | COA | SGD | İyi başlangıç + SGD ince ayar. |
| **HHO-SGD-MF** | `methods/hho_sgd_mf.py` | HHO | SGD | İyi başlangıç + SGD ince ayar. |
| **COA-KMeans-MF** | `methods/coa_kmeans_mf.py` | COA | K-Means | İyi başlangıç + küme tabanlı refinement. |
| **HHO-KMeans-MF** | `methods/hho_kmeans_mf.py` | HHO | K-Means | İyi başlangıç + küme tabanlı refinement. |
| **PSO-MF** | `methods/pso_mf.py` | PSO | — | Tam metaheuristik MF (karşılaştırma). |

---

## 11. Neden Bu Algoritmalar?

- **SGD:** Endüstride standart; hızlı, ölçeklenebilir; **baseline** ve **refinement** için.
- **COA / HHO:** Rastgele başlangıçtan kaçınmak; **daha iyi başlangıç** → daha iyi yerel minimum ve yakınsama.
- **K-Means:** Gradyan kullanmadan **alternatif refinement**; küme yapısı ve farklı hibritleri denemek için.
- **PSO:** “Sadece metaheuristik ile MF” deneyi; hibrit iki fazlı yöntemlerle karşılaştırma için.

Tüm algoritmalar aynı **MFModel** (U, V, r_ui = U·V^T) ve aynı **loss (MSE + L2)** ile değerlendirilir; fark sadece U,V’nin **nasıl arandığı ve iyileştirildiği**dir.
