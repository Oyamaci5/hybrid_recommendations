
# Özgünlük Listesi — Tez/Makale Katkıları

Her katkı için: **iddia · kanıt · literatürdeki durum · risk**.
Sıralama, savunmadaki güçlerine göre (Ö1 en güçlü).

---

## Ö1 — Kapasite-farkındalıklı adil karşılaştırma protokolü ★★★★

**İddia.** Kümeleme tabanlı CF'de meta-sezgisel algoritmalar, hesaplama
maliyeti eşitlenmeden karşılaştırılamaz. Bunun için kapasiteli onarım (repair)
+ sabit komşu havuzu (≈2N/K) + NFE eşitliği içeren bir protokol öneriyoruz.

**Kanıt.**
- Kısıtsız protokolde yöntemler farklı maliyet sınıflarında yarışıyor:
  ML-1M K=6'da B0'ın havuzu %60, AVOA'nın %34 (`K_SECIMI_SAVUNMASI.md`).
- GWO serbest protokolde 1. sırada görünüyor (havuz 286), havuz eşitlenince
  B0 seviyesine düşüyor — üç ayrı veri noktasında (`tabloB_plus_K40.csv`,
  `ml1m_ana.csv`).
- Aynı epoch'ta algoritmalar 5 kata kadar farklı NFE harcıyor.

**Literatür.** Taranan 8 çalışmanın hiçbirinde havuz büyüklüğü, küme boyut
dağılımı veya NFE raporlanmıyor. Kapasiteli kümeleme (Bradley 2000, Malinen &
Fränti 2014) kümeleme literatüründe var ama **öneri sistemlerine
uygulanmamış**.

**Risk.** "Kısıt yapay" itirazı → cevap hazır: kısıtsızken sistem *daha kötü*
(ML-1M: 0.6880 vs 0.6897) ve kısıt kullanıcıyı komşusuz bırakmıyor (%100 telafi).

---

## Ö2 — Dejenerasyon analizi: kısıtsız doğruluk hedefi kümelemeyi yok eder ★★★★

**İddia.** Küme tabanlı CF'de doğruluk temelli uygunluk fonksiyonu kısıtsız
bırakılırsa, küresel optimum "kümelemeyi iptal etmek"tir. Bu, literatürdeki
bazı sonuçların yorumunu değiştirir.

**Kanıt.**
- Kısıtsız MAE-fitness: 943 kullanıcının 928'i tek kümede (`MAKALE_KATKILARI` K1).
- ML-1M K=120, serbest atama: **105 küme boş**, tek kümede 3893 kullanıcı (%64) —
  ve bu dejenere yapı en iyi küme-ortalaması MAE'sini veriyor (`ml1m_buyukK.csv`).
- Doğrudan atama araması: HHO 943'ün **942'sini** tek kümeye koydu; NFE'yi
  5 katına çıkarmak değiştirmedi (`PARAMETRIZASYON_BULGUSU.md`).

**Literatür.** Küme boyut dağılımı raporlanmadığı için bu tuzak görünmez;
hiçbir çalışma bu analizi yapmıyor.

**Risk.** Yok — kendi verimizle gösteriliyor, iddia değil ölçüm.

---

## Ö3 — Kümeleme = yaklaşık komşu arama (ANN); komşu-recall yeni değerlendirme ekseni ★★★★

**İddia.** Kümelemenin işlevi "daha iyi komşu bulmak" değil, **aynı komşuları
daha ucuza bulmak**tır. Bu nedenle asıl ölçüt komşu geri çağırma oranıdır.

**Kanıt** (ML-100K, `KOMSU_RECALL_BULGUSU.md`):

| K | Havuz | B0 recall@20 | AVOA recall@20 |
|---|---|---|---|
| 6 | %33 | %28.0 | **%60.6** |
| 20 | %10 | %13.3 | **%28.3** |
| 40 | %5 | %8.9 | **%16.9** |

AVOA aynı maliyette **2 kat fazla gerçek komşu** koruyor — MAE farkının
mekanizma açıklaması.

**Literatür.** Taranan hiçbir CF-kümeleme çalışması komşu recall'ü raporlamıyor.

**Risk.** ML-1M'de ölçülmedi (eksik iş listesinde).

---

## Ö4 — Tahmin katmanının belirleyiciliği: literatürün kör noktası ★★★

**İddia.** Küme tabanlı CF'de tahmin katmanı seçimi, kümeleme algoritması
seçiminden **5 kat büyük** bir kaldıraçtır. Literatür bu katmanı geçiştirdiği
için hem mutlak performansı hem algoritma farkını olduğundan düşük gösteriyor.

**Kanıt** (ML-1M, K=40, aynı merkezler):
- Tahmin katmanı: küme-ort. 0.807 → kNN 0.734 → kNN+MF 0.720 = **0.087**
- Kümeleme algoritması: B0 0.723 → AVOA 0.707 = **0.017**
- Kaba tahminciyle algoritma farkı eziliyor: ML-100K'da küme-ortalamasıyla tüm
  metalar 0.84–0.85'te sıkışıyordu (`predictor_upgrade.csv`).

**Literatür.** HSC/Firefly küme-ortalaması kullanıyor; Katarya (2016) küme-içi
kNN kullanan tek çalışma. kNN'in güncelliği: Dacrema ve ark. (RecSys'19) iyi
ayarlı kNN'in birçok modern yöntemi geçtiğini gösteriyor.

**Risk.** "Karışım kullanmak zorlama" itirazı → ayarsız 0.5/0.5 ile ayarlıya
eşit sonuç (hiperparametre sömürüsü yok) + her bileşenin ablasyonu var.

---

## Ö5 — Kritik replikasyon: iyileşme iddiaları baseline ve protokole bağlı ★★★

**İddia.** Literatürdeki büyük iyileşme oranları, zayıf baseline ve raporlanmayan
protokol tercihlerinden kaynaklanıyor.

**Kanıt.**
- Zayıf baseline'a (random-init k-means) karşı tüm metalar +%6.5–8.5 → makalelerin
  bildirdiği büyüklük yeniden üretildi. Güçlü baseline'a (KMeans++ n_init=10)
  karşı fark kayboluyor.
- HSC replikasyonu: bildirilen MAE 0.685, bizim replikasyon 0.858. Üç iç
  tutarsızlık belgelendi: standart dışı MAE tanımı (payda = film sayısı),
  RMSE/MAE oranı 1.78 vs diğer satırlarda 1.27, MAE'nin K ile monoton azalması
  (`HSC_REPLIKASYON_ANALIZI.md`).
- Klasik baseline'lar aynı hatta konulduğunda birbirine 0.757–0.765 aralığında
  sıkışıyor (`klasik_baselines.csv`) — literatürdeki büyük farklar hattın
  zayıflığından.

**Literatür.** Dacrema 2019/2021'in derin öğrenme için yaptığını,
meta-sezgisel öneri sistemleri için biz yapıyoruz.

**Risk.** Dil hassasiyeti — "hata" değil "yeniden üretilemedi + şu gözlemler"
formülasyonu kullanılacak.

---

## Ö6 — Fark hunisi: algoritma seçiminin etkisi sistem katmanlarınca sönümlenir ★★★

**İddia.** Meta-sezgisel farkı optimizasyon katmanında 8 kat, tahmin katmanında
%2–3, tam sistemde binde yarım. Fark, hesap bütçesi daraldıkça büyür.

**Kanıt** (iki veri setinde tutarlı):

| Ölçüm noktası | Fark |
|---|---|
| Saf arama (Lloyd'suz WCSS) | 258 → 2075 (8 kat) |
| CF, kNN-only, K=30 | %2.2 |
| CF, kNN+MF, K=40 | %1.7–2.3 |
| CF, kNN+MF, K=6 | %0.2 |
| Global MF + bol havuz | anlamsız (p=0.077) |

ML-1M K taraması: havuz %33→%3 iken fark −0.0004 → −0.0196.

**Literatür.** Farkı tek sayıya sıkıştırıyorlar; katmanlı analiz yok.

**Risk.** Yok; hem kısıtlılığımızı hem katkımızı dürüstçe konumlandırıyor.

---

## Ö7 — Film önerisinde AVOA ile kümeleme (ilk uygulama) ★★

**İddia.** African Vulture Optimization Algorithm, film öneri sistemlerinde
kümeleme amacıyla ilk kez kullanılmıştır ve adil protokolde 22 aday arasından
seçilmiştir.

**Kanıt.** ML-1M ana tablo: AVOA 15/15 hücrede birinci, diğer 4 metayı da
anlamlı geçiyor (Holm p ≤ 0.0003), kullanıcı-bazlı Wilcoxon p=2.6e-193.
ML-100K'da Friedman p=1.6e-35.

**Literatür.** AVOA+öneri sistemi kombinasyonu taramada bulunamadı.

**Risk.** *Önemli dürüstlük notu:* AVOA'nın üstünlüğü **tahminci-fitness
hizalamasına bağlı**; küme-ortalaması tahmincisiyle B0'ı geçemiyor
(`BUYUK_K_CMEAN_BULGUSU.md`). Bu, sınırlılıklar bölümünde açıkça yazılacak.
Ayrıca K=6 gibi bol bütçeli noktalarda HHO/NGO ile ayrışmıyor.

---

## Ö8 — Yöntemsel bileşenler (ikincil ama savunulabilir) ★★

| Bileşen | Katkı | Kanıt |
|---|---|---|
| **Memetik warm start** | K-means++ → meta rafinasyon; literatürde ters yön (meta → k-means init) var, bu yön yok | Sıfırdan arayan meta B0'ı geçemiyor; warm start ile 5/5 geçti |
| **Soft top-2 havuz** | Kapasite kısıtının zararını %100 telafi ediyor | Yer değiştiren kullanıcıların hepsi en yakın kümesini havuzda buluyor |
| **Küme-başına MF + kNN karışımı** | Tahminin iki bileşeni de kümelemeye bağlı → algoritma farkı tahmine akıyor | Global MF farkı 4 kat sönümlüyordu |
| **NMF ile arama uzayı indirgeme** | HSC'de 117.740 boyut, bizde 1.400 (84 kat) | Meta-sezgiselin çalışabilmesi için önkoşul |
| **Hiperparametre transfer bulgusu** | İç-val ile ayar test'i kötüleştiriyor (Spearman −0.50) | 24 konfigürasyon taraması |

---

## Tez giriş bölümü için katkı paragrafı (taslak)

> Bu çalışmanın katkıları şunlardır: (i) kümeleme tabanlı işbirlikçi filtrelemede
> meta-sezgisel algoritmaların hesaplama maliyeti eşitlenerek karşılaştırılmasını
> sağlayan, kapasite kısıtı ve sabit komşu havuzuna dayalı bir değerlendirme
> protokolü önerilmiştir; (ii) kısıtsız doğruluk hedefinin kümelemeyi dejenere
> ettiği ve bu durumun küme boyut dağılımı raporlanmadığında görünmez kaldığı
> deneysel olarak gösterilmiştir; (iii) kümelemenin yaklaşık komşu arama işlevi
> nicelleştirilerek komşu geri çağırma oranı yeni bir değerlendirme ekseni olarak
> önerilmiş, önerilen yöntemin aynı maliyette iki kat fazla gerçek komşu koruduğu
> ortaya konmuştur; (iv) tahmin katmanı seçiminin kümeleme algoritması seçiminden
> beş kat büyük bir etkiye sahip olduğu gösterilerek literatürdeki sonuçların
> yorumuna katkı sağlanmıştır; (v) African Vulture Optimization Algorithm film
> öneri sistemlerinde ilk kez kümeleme amacıyla kullanılmış ve 22 aday arasından
> adil bir eleme protokolüyle seçilmiştir.

---

## Sınırlılıklar (dürüstlük bölümü — savunmada güç kazandırır)

1. AVOA'nın üstünlüğü fitness–tahminci hizalamasına bağlı; küme-ortalaması
   tahmincisiyle B0 daha iyi.
2. Bol bütçe rejiminde (havuz ≥%20) algoritmalar ayrışmıyor.
3. Katalog kapsamasında %2.9 gerileme var (doğruluk–çeşitlilik ödünleşimi).
4. RMSE'de ayarlı SVD referansına yakınız ama önünde değiliz (0.879 vs 0.876).
5. Tek veri ailesi (MovieLens); zaman-farkındalıklı bölme denenmedi.
6. Komşu-recall analizi yalnız ML-100K'da yapıldı.
