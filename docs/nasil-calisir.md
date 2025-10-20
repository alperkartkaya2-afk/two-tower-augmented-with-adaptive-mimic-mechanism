# Nasıl Çalışır?

Bu rehber projeyi uçtan uca anlatır: Veri nasıl alınır, nasıl temizlenir, iki kuleli model nasıl kurulur, eğitim nasıl yapılır ve elde edilen sonuçlar nasıl değerlendirilir? Mühendisler, veri bilimciler ve ürün ekipleri için pratik bir el kitabıdır.

> **Özet**
> - Veri → temizlenir, indekslenir, tensörlere döner  
> - Kuleler → ID embedding + metadata kodlayıcıları (adaptive mimic füzyonu)  
> - Eğitim → negatif örneklemeli BCE, hibrit optimizasyon  
> - Değerlendirme → ranking metrikleri, embedding diagnostikleri, özellik korelasyonları, nitel öneri incelemeleri → Markdown raporu

---

## 1. Sistem Genel Görünümü

```mermaid
flowchart LR
    A[Ham Veriler<br/>books.csv & users.csv] --> B[Yükleyiciler<br/>(limit + doğrulama)]
    B --> C[Ön İşleme<br/>filtreler + özellik matrisleri]
    C --> D[Eğitim Veri Seti]
    D --> E[Kule İnşası<br/>ID + metadata encoder'ları]
    E --> F[Eğitim Döngüsü<br/>negatif örnekleme + BCE]
    F --> G[Değerlendirme & Diagnostik]
    G --> H[Rapor<br/>artifacts/reports/...]
```

---

## 2. Veri Katmanı

| Bileşen | Ne yapar? | Neden önemli? |
|---------|-----------|----------------|
| **Ham CSV'ler** (`data/books.csv`, `data/users.csv`) | Kitap metaverisi + kullanıcı etkileşimleri | Git’e eklenmez; herkes yerelde temin etmeli. Trimmed versiyonlar hızlı prototip içindir. |
| **Konfigürasyon** (`configs/default.yaml:data`) | Dosya yolları, satır limitleri, min etkileşim eşikleri, özellik parametreleri | Deney yönetimini tek dosyada toplar. |
| **Loaders** (`src/data/loaders.py`) | CSV okur, limit uygular, kitap tablosunda olmayan etkileşimleri eler | Veri tutarlılığını garanti eder. |
| **Preprocessing** (`src/data/preprocessing.py`) | Etkileşimleri temizler, min kullanıcı/ürün frekansı koşullarını iteratif uygular, indeks map’leri + özellik matrisleri çıkartır | Gürültüyü azaltır, hesaplamayı ucuzlatır. |
| **Feature Engineering** (`src/data/features.py`) | Z-skorlu nümerikler, başlık kelime/karakter istatistikleri, kategori & yazar sinyalleri, kullanıcıya ait özetler | Cold-start dahil olmak üzere modelin metadata’dan faydalanmasını sağlar. |
| **Yardımcılar** (`indexers`, `datasets`, `samplers`) | Map yapıları, PyTorch `Dataset`, negatif örnekleme | Eğitim ve değerlendirme döngüsünün temel parçaları. |

> **Not:** Sadece satır limiti ekleyerek bile dakikalar içinde teşhis yapabilir, sonrasında limitleri kaldırıp tam veriyle eğitime geçebilirsiniz.

---

## 3. Model Katmanı

### Two-Tower Mimarisi (`src/models`)

| Parça | Teknik özet | Önemi |
|-------|-------------|-------|
| **ID Embedding’leri** (`build_id_embedding`) | Kullanıcı/ürün için 96 boyutlu, `sparse=True` destekli embedding tabloları | SparseAdam ile bellek dostu eğitim |
| **Özellik Kodlayıcıları** (`build_feature_encoder`) | Metadata vektörleri 192 → 96 boyutlu MLP’den geçer (dropout opsiyonlu) | Metadata ile collaborative sinyaller dengelenir |
| **Adaptive Mimic** (`AdaptiveMimicModule`) | ID embedding + metadata embedding’ini kapı mekanizmasıyla harmanlar | Cold-start kullanıcı/ürünlerde kaliteyi yükseltir |
| **TowerEncoder** | Füzyon modu (`identity`, `sum`, `concat`, `adaptive_mimic`) seçilebilir, cihaz yönetimini üstlenir | Config üzerinden kule mimarisini oynatmak kolay |
| **TwoTowerModel** | Kullanıcı ve ürünü tek arayüze bağlar, benzerlik (cosine/dot) döner | Eğitim & değerlendirme kodu detay bilmeden kuleleri kullanır |

---

## 4. Eğitim ve Değerlendirme

### 4.1 Eğitim Adımları

1. **Seed** – `experiment.seed` ile tekrar üretilebilirlik.  
2. **Veri Seti Oluşturma** – Min etkileşim filtreleri uygulanır, sayısal özet log’lanır.  
3. **Train/Validation Ayrımı** – Kullanıcı başına son etkileşim validation’a ayrılır.  
4. **Kule Kurulumu** – Config → builder → `TowerEncoder`; boyutlar log’lanır.  
5. **Optimizer Hazırlığı** – Dense parametreler için Adam/AdamW/SGD, embedding’ler için SparseAdam.  
6. **Mini-batch Döngüsü** – Negatif örnekleme, pozitif/negatif logit, BCE loss, geri yayılım, optimizasyon.  
7. **Epoch Logları** – Kayıp, veri özetleri, kule boyutları.

### 4.2 Değerlendirme ve Diagnostik

| Adım | İçerik |
|------|--------|
| **Ranking Metrikleri** | Recall/Precision/NDCG/MAP; aday havuzu pozitif + rastgele negatiflerden oluşur. |
| **Embedding Diagnostikleri** | Norm dağılımları, komşuluk kategori uyumu, kullanıcı embedding ↔ metadata uyumu. |
| **Özellik Korelasyon Analizi** | Pearson r + p-value; skorla en korele metadata sütunları listelenir. |
| **Nitel Öneriler** | Seçilen kullanıcılar için top-K öneriler, kategori/yazar isabet oranı. |
| **Rapor** | Tüm çıktılar `artifacts/reports/recommendation_report.md` dosyasında toplanır. |

> **İpucu:** `evaluation.candidate_samples` değerini arttırarak metrikleri daha kararlı hale getirebilirsiniz (maliyet artar).

---

## 5. Konfigürasyon Hızlı Tablo

```yaml
data:
  books_limit: null
  interactions_limit: null
  min_user_interactions: 5
  min_item_interactions: 10
  feature_params:
    numeric_columns: ["average_rating", "price", "rating_number"]
    category_top_k: 300
    author_top_k: 300
    user_aggregation: "mean"

model:
  user_encoder:
    id_embedding:
      params: { embedding_dim: 96, sparse: true }
    feature_encoder:
      type: mlp
      hidden_dims: [192]
      output_dim: 96
      dropout: 0.1
    fusion: adaptive_mimic
  item_encoder: { ... }
  similarity: "cosine"
  device: "cpu"

training:
  batch_size: 512
  num_epochs: 10
  negatives_per_positive: 5

evaluation:
  metrics_k: [5, 10, 20]
  candidate_samples: 500

diagnostics:
  item_sample_size: 50
  user_sample_size: 500
  neighbor_k: 5
  feature_corr_sample_size: 20000
  feature_corr_top_k: 20
  report_path: "artifacts/reports/recommendation_report.md"
```

---

## 6. Pratik İpuçları

| Durum | Aksiyon | Açıklama |
|-------|---------|----------|
| Büyük veri dosyası | Önce satır limitlerini kullan, sonra kaldır | Bellek taşmasını önler |
| Filtre sonrası veri kalmadı | `min_user/item_interactions` değerlerini düşür veya daha fazla satır yükle | Subsample edilmiş veriyle 5/10 eşiği agresif olabilir |
| Kategori/yazar genişliği | One-hot yerine embedding planla | Parametre yükü düşer, semantikler öğrenilir |
| Sparse optimizer hatası | `sparse` parametrelerini kontrol et | Yanlış kombinasyon eğitim hatasına yol açar |
| Rapor paylaşımı | Markdown → HTML/PDF dönüştür; CI’da otomatikleştir | Ekip içi görünürlük sağlanır |

---

## 7. Hızlı Metrik Panosu

| Metrik | Ne anlatır? | Kontrol |
|--------|--------------|---------|
| **Recall@K / Precision@K** | Doğru ürünleri bulma oranı | Popülerlik baz çizgisiyle kıyasla |
| **NDCG@K** | Doğru ürünlerin üst sıralarda olması | Düşüş varsa skor fonksiyonunu incele |
| **Hit Rate@K** | En az bir doğru öneri var mı | Kullanıcı memnuniyeti sinyali |
| **MAP@K** | Genel sıralama kalitesi | Uzun kuyruk kullanıcıları için önemli |
| **Feature Corr (r/p)** | Metadata sinyali skorla ilişkili mi | Yüksek p değerlerini gözden geçir |
| **Kategori/Yazar Match** | Geçmiş tercihlerle uyum | Adaptive mimic veya veri pipeline’ını tune et |

---

## 8. Yol Haritası

1. Kategori/yazar embedding tasarımına geçiş (hashing, regularization).  
2. Checkpoint + early stopping + hiperparametre taraması otomasyonu.  
3. MLflow / W&B gibi deney izleme araçlarıyla entegrasyon.  
4. Üretim ortamına servis geçişi (model + map + API).

---

## 9. Sözlük

- **Two-Tower Model:** Kullanıcı ve ürün embedding’lerinin benzerlik üzerinden karşılaştırıldığı mimari.  
- **Negative Sampling:** Rastgele eşleşme üretip modele “bu satın alınmadı” demek.  
- **Adaptive Mimic:** Metadata ve ID embedding’lerini dinamik olarak harmanlayan kapı mekanizması.  
- **Hybrid Optimizer:** Dense ve sparse parametreleri aynı döngüde optimize etmek.  
- **Ranking Metrikleri:** Öneri kalitesini ölçen metrik seti (Recall, Precision, NDCG, MAP, Hit Rate).  
- **Özellik Korelasyon Testi:** Skor ile özellik sütunları arasındaki Pearson r/p değerlerini analiz eder.

---

## 10. Hızlı Başlangıç Kontrol Listesi

1. `pip install -e .[dev]` ile bağımlılıkları kur.  
2. `data/` klasörüne `books.csv` & `users.csv` dosyalarını yerleştir.  
3. `python scripts/preprocess.py --config configs/default.yaml` çalıştır, log’ları kontrol et.  
4. `python scripts/train.py --config configs/default.yaml` ile eğit.  
5. `artifacts/reports/recommendation_report.md` raporuna göz at – metrikler, embedding diagnostikleri, özellik korelasyon analizleri ve örnek öneriler burada.

Başarılar!

