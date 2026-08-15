# Danışman Görüşmesi Kılavuzu

Amaç: 20-25 dakikada (a) ne yaptığımızı, (b) neden böyle yaptığımızı, (c) hangi
katkıların tez/makale değerinde olduğunu anlatmak. Her karar bir **sorun → çözüm →
kanıt** zincirine bağlı. Tüm sayılar `algo_selection_v2/results/` altında.

---

## 0) Açılış — 60 saniyelik özet (ezberlenecek)

> "Hocam, meta-sezgisel algoritmalarla kümeleme tabanlı işbirlikçi filtreleme
> üzerine çalışıyorum. Başlangıçta literatürdeki gibi WCSS'i optimize ettim ama
> iyileşme binde 5'te kaldı. Nedenini araştırınca üç yapısal sorun buldum:
> hedef fonksiyon yanlış hedefe bakıyordu, karşılaştırmalar farklı hesap
> maliyetlerini gizliyordu ve kısıtsız optimizasyon kümelemeyi dejenere ediyordu.
> Bunları düzelten bir protokol kurdum. Sonuçta ML-100K'da MAE 0.745, NDCG 0.840
> elde ettim — bu, kümeleme yapmayan tam kNN'den (0.747/0.835) daha iyi ve
> komşuluk havuzunun sadece üçte biriyle. Ayrıca 5 fold × 10 seed = 300 koşuda
> beş meta-sezgiselin beşi de K-means++ baseline'ını istatistiksel olarak anlamlı
> geçti (Holm düzeltmeli p<1e-9)."

---

## 1) "Neden bu algoritmalar?" — seçim hikâyesi

**Anlatım sırası:**
1. mealpy kütüphanesindeki 20+ algoritmayla başladım (literatürde sık geçenler +
   güncel olanlar).
2. **Adil eleme protokolü:** aynı NFE bütçesi (fonksiyon çağrısı sayısı), 5 seed,
   eşit-bütçeli rastgele arama eşiği. → 22 algoritmadan 7'si rastgele aramayı
   geçebildi (`pure_meta_summary.csv`).
3. Final tur: NFE-eşit 15.000 bütçe, 30 seed → GWO, AVOA, NGO, HGS, HHO kısa listesi
   (`final_run.csv`, Friedman p=3.9e-23).
4. **Neden AVOA öne çıktı:** eşit maliyetli arenada (kapasite kısıtlı) hem K=6 hem
   K=40 çalışma noktasında birinci (`k_tarama_repair.csv`). Ayrıca özgünlük:
   film önerisinde AVOA tabanlı kümeleme literatürde yok.
5. HHO ve GWO neden duruyor: HHO'nun literatür karşılığı var (HHO-K-means makalesi),
   GWO'nun da (Katarya 2018 GWO+FCM) → doğrudan kıyas için.

**Hoca "neden AVOA?" derse tek cümle:** *"Adil protokolde iki çalışma noktasında da
birinci çıktı ve film önerisinde AVOA'lı kümeleme literatürde yok — hem performans
hem özgünlük gerekçesi var."*

---

## 2) "Neden bu kadar çok deney/birleşim var?" — savunma

Bu soruya **özür dileyerek değil, metodolojik güç olarak** cevap verilmeli:

> "Her ek bileşen bir soruna cevap olarak eklendi, keyfi denemeler değil.
> Elimde 20 numaralı katkı dosyası var; her biri 'şu sorunu gördüm → şunu denedim
> → şu kanıtla kaldı/elendi' şeklinde kayıtlı. Ayrıca elenenlerin listesi de var:
> significance weighting, IUF, kullanıcı-uyarlamalı β, LOF, NFE artırma,
> hiperparametre ayarı — hepsi denendi, katkı vermedi, kayda geçti."

**Zincir (tahtaya çizilecek):**

```
WCSS fitness → iyileşme yok  ──► tahmine hizalı fitness
                                    │
                            tek dev küme (dejenerasyon)
                                    ├──► kapasite kısıtı (repair)
                                    │        │
                                    │   sınır kullanıcılar komşusuz
                                    │        └──► soft top-2 havuz
                                    │
                            havuz farkı sonucu gizliyor
                                    └──► eşit bütçeli karşılaştırma
```

---

## 3) "Neden farklı seed'ler?" — istatistik gerekçesi

> "Meta-sezgiseller stokastik; tek koşuda çıkan fark şansa bağlı olabilir.
> İncelediğim 6 makalenin hiçbirinde çoklu koşu ya da anlamlılık testi yok.
> Ben 10 seed × 5 resmi fold = 50 bağımsız hücre kullandım. Ayrıca 5-fold
> ortalaması üzerinden Wilcoxon yapmak matematiksel olarak hatalı — n=5'te
> ulaşılabilecek en küçük p 0.0625, yani α=0.05'te anlamlılık imkânsız.
> Bu yüzden hem hücre bazında (n=50) hem kullanıcı bazında (n=943) test yaptım,
> çoklu karşılaştırma için Holm düzeltmesi uyguladım (Demšar 2006 protokolü)."

---

## 4) Sunulacak dört ana sonuç (sırayla)

**S1 — Fark hunisi (metodolojik katkı).**
Meta-sezgisel farkı: saf arama katmanında 8 kat (258→2075 WCSS), tahmin
katmanında %2-3, tam sistemde binde yarım. Nerede ve neden söndüğünü ölçtük.
*Cümle:* "Literatür farkı tek sayıya sıkıştırıyor; biz üç seviyede ayrı ölçtük."

**S2 — Kritik replikasyon (literatür eleştirisi).**
Zayıf baseline'a (random-init k-means) karşı tüm metalar +%6.5–8.5 iyileşiyor —
makalelerin raporladığı büyüklük. Güçlü baseline'a (KMeans++ n_init=10) karşı
fark kayboluyor. *Cümle:* "İyileşme iddiaları baseline seçimine bağlı."

**S3 — Ana sonuç (300 koşu, mühürlü).**
5/5 meta, B0'ı MAE ve NDCG'de anlamlı geçti; kullanıcı bazlı Wilcoxon p=5e-10…4e-63;
Friedman χ²=179.7. *Cümle:* "Kümeleme kalitesi algoritma seçimine bağlıdır."

**S4 — Sistem sonucu (en iyi değerimiz).**
K=6, kapasite kısıtlı, soft top-2, küme-MF+kNN karışımı → MAE 0.7450 / NDCG 0.8399,
havuzun %33'üyle; kümesiz kNN'i (0.7467/0.8352) her iki metrikte geçiyor.
*Cümle:* "Kümeleme doğruluğu düşürmez, doğru merkezlerle artırır — üstelik 3 kat ucuza."

---

## 5) Tezde "yenilik" diye sunulacaklar (öncelik sırasıyla)

1. **PACR / tahmine-hizalı merkez rafinasyonu:** K-means'in optimize edemediği
   hedefte (iç-validasyon tahmin hatası) meta-sezgisel merkez arama; warm start
   ile K-means çıktısından başlanır. Literatürde meta→K-means init var, tersi yok.
2. **Eşit-maliyetli karşılaştırma protokolü:** kapasite kısıtı (repair) + sabit
   havuz bütçesi → algoritmalar aynı hesap maliyetinde yarışır. Taranan CF-kümeleme
   makalelerinde havuz/küme boyutu hiç raporlanmıyor.
3. **Küme-başına MF + kNN karışımı:** tahminin iki bileşeni de kümelemeye bağlı;
   6 makalenin hiçbirinde MF karışımı yok.
4. **Dejenerasyon analizi:** kısıtsız doğruluk-fitness'ı kümelemeyi yok ediyor
   (943'ün 928'i tek kümede) — küme boyut dağılımı raporlamayan çalışmalar bu
   tuzağa açık.
5. **Hiperparametre transfer bulgusu:** iç-val ile parametre seçimi test'i
   kötüleştiriyor (Spearman −0.50) → "ayarlanmış sonuç" raporlayan çalışmalara uyarı.
6. **Fitness ablasyonu:** MAE / NDCG / hibrit hedef karşılaştırması; hibrit üç
   metrikte birden en iyi. Sıralama-hedefli fitness literatürde yok.

---

## 6) Zor sorular ve hazır cevaplar

**"Kullanıcıyı zorla başka kümeye atmak yanlış değil mi?"**
> Kapasiteli kümeleme yerleşik bir alan (Bradley 2000 constrained k-means;
> Malinen & Fränti 2014 balanced k-means). Ayrıca ölçtüm: yer değiştiren
> kullanıcıların %100'ünün en yakın kümesi soft top-2 havuzda kalıyor, yani
> komşularını kaybetmiyorlar. Kısıtı kaldırdığımda algoritma farkı 10 kat eriyor
> ve dejenerasyon geri geliyor — kısıt, ölçümü mümkün kılan koşul.

**"MAE 0.68 diyen makaleler var, sizinki 0.745?"**
> O değerler tek split, tek koşu ve raporlanmamış protokollerle elde edilmiş.
> Referans noktası: ayarlı SVD (Surprise kütüphanesi, 5-fold) 0.736, tam kNN 0.747.
> Kümeleme komşu havuzunu daralttığı için tam kNN'den iyi çıkması yapısal olarak
> beklenmez; 0.68 bu tavanın çok altında ve doğrulanamıyor. Bizim protokolümüz
> resmi 5-fold, sızıntısız ve tekrarlanabilir.

**"Neden ML-1M yok?"**
> Altyapı hazır, ölçek genellemesi için sıradaki adım. Önce protokolü ML-100K'da
> mühürledim çünkü literatürün tamamı bu veri setinde raporluyor.

**"Bu kadar bileşen fazla değil mi, sadeleştirilemez mi?"**
> Ablasyon tablosu var: her bileşenin katkısı ayrı ölçüldü (fitness, kısıt, soft
> havuz, MF karışımı, tür bilgisi). Katkısı çıkmayanlar zaten elendi.

**"Sonuçlar tekrarlanabilir mi?"**
> Tüm seed'ler sabit, resmi fold'lar, her koşu CSV'ye yazılıyor, komutlar README'de.

---

## 7) Görüşmeye götürülecekler

- [ ] Bu kılavuz + `MAKALE_KATKILARI.md` (21 maddelik katkı/eleme kaydı)
- [ ] `HOCAYA_RAPOR.md` (yöntem + literatür karşılaştırma tablosu)
- [ ] `PROTOKOL_SAVUNMASI.md` (kısıt ve soft havuz için kaynaklar)
- [ ] `INCELEME_6_MAKALE.md` (rakip makale karnesi)
- [ ] 3 grafik: `k_tarama_repair.png` (K taraması), `tamrun_B.png` (300 koşu
      dağılımları), `frontier.png` (doğruluk-maliyet cephesi)
- [ ] Tek sayfalık sonuç tablosu (S1–S4)

## 8) Görüşmeden çıkarken netleşmesi gerekenler

1. Tez mi, makale mi, ikisi mi? (elde makale düzeyinde malzeme var)
2. ML-1M genellemesi zorunlu mu?
3. Hedef dergi/konferans → format ve uzunluk kararı
4. Teslim takvimi ve ara teslim noktaları
