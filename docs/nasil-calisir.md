# Nasıl Çalışır?

Bu rehber, projeyi uçtan uca açıklar: Veri nereden gelir, nasıl dönüştürülür, iki kuleli model (two‑tower) nasıl çalışır, nasıl eğitilir ve sonuçlar nasıl değerlendirilir/yorumlanır. Mühendisler, veri bilimciler ve ürün paydaşları için yaşayan bir el kitabıdır.

> **Özet (tek bakışta)**
> - Veri → temizlenir, indekslenir → özellik tensörleri  
> - Kuleler → ID embedding’leri + metadata kodlayıcıları (adaptive‑mimic füzyonu)  
> - Eğitim → negatif örneklemeli BCE, hibrit optimizasyon (dense + sparse)  
> - Değerlendirme → sıralama metrikleri, embedding diagnostikleri, özellik korelasyonları, nitel denetimler → Markdown raporu

---

## 1. Sistem Genel Görünümü

```mermaid
flowchart LR
    A[Ham Veriler<br/>books.csv & users.csv] --> B[Yükleyiciler<br/>(limit + şema kontrolleri)]
    B --> C[Ön İşleme<br/>filtreler + özellik matrisleri]
    C --> D[Eğitim Veri Seti]
    D --> E[Kule İnşası<br/>ID + metadata kodlayıcıları]
    E --> F[Eğitim Döngüsü<br/>negatif örnekleme + BCE]
    F --> G[Değerlendirme & Diagnostik]
    G --> H[Rapor<br/>artifacts/reports/*.md]
```

Aşağıdaki her adım **teknik ayrıntılar** ve **neden önemli** notlarıyla gelir.

---

## 2. Veri Katmanı

| Bileşen | Ne yapar? | Neden önemli? |
|---|---|---|
| **Ham CSV’ler** (`data/books.csv`, `data/users.csv`) | Kitap metaverisi + kullanıcı etkileşimleri. | GIT’e alınmaz; herkes yerelde sağlar. Hızlı denemeler için kırpılmış sürümler bulunur. |
| **Konfigürasyon** (`configs/default.yaml:data`) | Dosya yolları, satır limitleri (`books_limit`, `interactions_limit`), minimum etkileşim eşikleri, özellik parametreleri. | Deneyleri tek yerden yönetir; farklı senaryolar için YAML’ı çoğaltabilirsiniz. |
| **Loaders** (`src/data/loaders.py`) | CSV okur, limit uygular, kitap tablosunda bulunmayan etkileşimleri log’layarak eler. | Referanssız etkileşimi önler; veri bütünlüğü. |
| **Preprocessing** (`src/data/preprocessing.py`) | Etkileşimleri temizler; kullanıcı/ürün minimum frekansını iteratif uygular; indeks eşlemleri + özellik matrisleri üretir; `TrainingDataset` döner. | Aşırı seyrek (gürültülü) kullanıcı/ürünleri temizler; doğruluğu artırır, maliyeti düşürür. |
| **Özellik Mühendisliği** (`src/data/features.py`) | Nümerik z‑skorlar; başlık kelime/karakter istatistikleri; kategori & yazar sinyalleri; kullanıcı özetleri. | Metadata sinyallerini modele taşır; cold‑start’ta kritik. |
| **Yardımcılar** (`indexers`, `datasets`, `samplers`) | İndeks map’leri, PyTorch `Dataset`, negatif örnekleme. | Eğitim/değerlendirme döngüsünün belkemiği. |

> **Neden bu tasarım?**  
> Tüm veriyi bilinçsizce CPU/GPU’ya yüklemek yerine, erken safhada örnekleyip filtreliyoruz; böylece ilk koşulda bile dakikalar içinde geri bildirim alıyoruz.

---

## 3. Model Katmanı

### Two‑Tower Mimarisi (`src/models`)

| Parça | Teknik özet | Neden önemli |
|---|---|---|
| **ID Embedding’leri** (`build_id_embedding`) | Kullanıcı/ürün için 96‑boyutlu embedding tabloları; `sparse=True` destekli. | `SparseAdam` ile bellek dostu ve ölçeklenebilir. |
| **Özellik Kodlayıcıları** (`build_feature_encoder`) | Metadata vektörünü MLP ile 192 → 96 kodlar (opsiyonel dropout). | Metadata ile collaborative sinyaller dengelenir. |
| **Adaptive Mimic** (`AdaptiveMimicModule`) | ID + metadata embedding’lerini öğrenilebilir kapı (gate) ile harmanlar. | Cold‑start kullanıcı/ürünlerde büyük katkı sağlar. |
| **Tower Encoder** (`TowerEncoder`) | Füzyon modu (`identity`/`sum`/`concat`/`adaptive_mimic`) + cihaz yönetimi. | Mimariyi yalnızca config ile değiştirebilirsiniz. |
| **TwoTowerModel** | İki kuleyi tek arayüzde birleştirir; benzerlik (cosine veya dot) döndürür. | Eğitim/değerlendirme kodu kule detayını bilmek zorunda kalmaz. |

> **Teknik not:** Sparse embedding’lerde `padding_idx` kullanılabilir; fakat `sparse=True` ile `max_norm` birlikte kullanılamaz. Config, bu yasa dışı kombinasyonları engeller.

---

## 4. Eğitim ve Değerlendirme Hattı

### 4.1 Eğitim (`src/pipelines/training.py`)

1. **Seed & Config** — tekrar üretilebilirlik için `experiment.seed`.  
2. **Veri Seti Kurulumu** — minimum etkileşim eşikleri; özellik matrisleri; log’lar.  
3. **Train/Validation Ayrımı** — kullanıcı başına en son etkileşimi validation’a ayırır (timestamp yoksa uyarır ve validation’ı atlar).  
4. **Kule Kurulumu** — config → builder → `TowerEncoder` (boyutlar log’lanır).  
5. **Optimizer’lar** — dense parametreler için Adam/AdamW/SGD; sparse embedding’ler için `SparseAdam`.  
6. **Mini‑batch Döngüsü**  
   - Negatif örnekleme (`sample_negative_items`)  
   - Kullanıcı/ürün embedding’leri; pozitif & negatif logit’ler  
   - BCE loss → backward → optimizer step  
7. **Epoch Logları** — loss, veri seti boyutları, kule boyutları.

### 4.2 Değerlendirme & Diagnostik

| Adım | İçerik |
|---|---|
| **Sıralama Metrikleri** | `_evaluate_model` pozitifler + rastgele negatiflerden oluşan sınırlı aday havuzuyla Recall/Precision/NDCG/MAP hesaplar. |
| **Embedding Diagnostiği** | Norm dağılımları; en yakın komşu kategori uyumu; kullanıcı embedding ↔ metadata hizası. |
| **Özellik Korelasyonları** | Skorlar ile metadata arasında Pearson **r** + **p** değeri; en bilgili özellikler raporda listelenir. |
| **Nitel Öneriler** | Örnek kullanıcılar için Top‑K sonuçlar; kategori/yazar isabetleri. |
| **Markdown Raporu** | Yukarıdaki tüm çıktılar `artifacts/reports/recommendation_report.md` dosyasında toplanır. |

> **İpucu:** Değerlendirme örneklenmiş negatifler kullandığı için `evaluation.candidate_samples` değerini artırmak doğruluğu artırır (maliyet de artar).

---

## 5. Konfigürasyon Hızlı Bakış

```yaml
data:
  books_limit: null                # null/None → tamamını yükle
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
  item_encoder: { ... }            # kullanıcı kulesiyle benzer yapı
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

## 6. Pratik İpuçları & Uyarılar

| Durum | Ne yapmalı? | Neden |
|---|---|---|
| **Çok büyük CSV’ler** | Önce `books_limit` / `interactions_limit` ile küçük başlayın. | OOM riskini azaltır, iterasyonu hızlandırır. |
| **Aşırı filtre sonucu veri azaldı** | `min_user_interactions` / `min_item_interactions` değerlerini düşürün veya daha çok satır yükleyin. | 5/10 eşikleri alt örneklemde agresif olabilir. |
| **Geniş kategori/yazar özellikleri** | One‑hot yerine öğrenilmiş embedding (örn. 32–64 boyut). | Parametre yükünü düşürür, semantiği yakalar. |
| **SparseAdam hataları** | Embedding katmanlarını `sparse=True` ile tanımladığınızdan emin olun. | Yanlış kombinasyon eğitimi kırar. |
| **Rapor paylaşımı** | Markdown’ı HTML/PDF’e çevirin; CI içinde otomatikleştirin. | Takım görünürlüğünü artırır. |

---

## 7. Hızlı Metrik Panosu

| Metrik | Anlamı | İzleme önerisi |
|---|---|---|
| **Recall@K / Precision@K** | Top‑K doğruluk. | Popülerlik baz çizgisine karşı kıyaslayın. |
| **NDCG@K** | Doğru öğelerin üst sıralara yerleşmesi. | Düşüş varsa skor/negatif örneklemeyi gözden geçirin. |
| **Hit Rate@K** | Top‑K içinde en az bir doğru öğe var mı? | Kullanıcı memnuniyeti için basit bir vekil. |
| **MAP@K** | Genel sıralama kalitesi. | Özellikle uzun kuyruk kullanıcılar için değerli. |
| **Feature Corr (r/p)** | Metadata ile skorların ilişkisi. | Zayıf (yüksek p) özellikleri sadeleştirin. |
| **Kategori/Yazar Uyumu** | Kullanıcı geçmişiyle uyum. | Adaptive‑mimic ve metadata hattını ayarlayın. |

---

## 8. Yol Haritası (Sonraki Adımlar)

1. Kategori/yazar için embedding’lere geçiş; hashing/regularization stratejileri.  
2. Checkpoint + early stopping + hiperparametre taramalarını otomatikleştirme.  
3. Deney izleme (MLflow/W&B) + raporları CI hattına entegre etme.  
4. Servisleşme: kuleleri ve indeks map’lerini REST/gRPC API ile sunma.

---

## 9. Sözlük

- **Two‑Tower Model:** Kullanıcı ve ürün embedding’lerini benzerlik üzerinden karşılaştıran öneri mimarisi.  
- **Negative Sampling:** Rastgele uyuşmayan eşleşmeler üretip modele “bu etkileşim olmadı” dedirtir.  
- **Adaptive Mimic:** Metadata ve ID embedding’lerini dinamik bir kapı mekanizmasıyla harmanlar.  
- **Hibrit Optimizasyon:** Dense (Adam/AdamW) ve sparse (SparseAdam) parametreleri birlikte eğitme.  
- **Sıralama Metrikleri:** Öneri kalitesi ölçümü (Recall, Precision, NDCG, MAP, Hit Rate).  
- **Özellik Korelasyon Testi:** Skor ile özellikler arasındaki Pearson r/p analizi; sürükleyici sinyalleri ortaya çıkarır.

---

## 10. Hızlı Başlangıç Kontrol Listesi

1. Bağımlılıkları kurun (varsayılan CPU):
   ```bash
   pip install -e .[dev]

   # GPU (sürücünüzle uyumlu CUDA etiketini seçin)
   pip install -e .[dev,gpu] --extra-index-url https://download.pytorch.org/whl/cu121
   ```
   Doğru CUDA wheel indeksini (cu118, cu121, cu124, ...) seçmek için PyTorch kurulum tablosuna bakın.
2. `books.csv` & `users.csv` dosyalarını `data/` klasörüne koyun.  
3. Ön işleme: `python scripts/preprocess.py --config configs/default.yaml` (log’larda kullanıcı/ürün sayısını kontrol edin).  
4. Eğitim: `python scripts/train.py --config configs/default.yaml`  
5. Raporu inceleyin: `artifacts/reports/recommendation_report.md` — metrikler, embedding diagnostikleri, özellik korelasyon tabloları ve örnek öneriler.

Başarılar ve iyi deneyler!
