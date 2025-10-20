# How Does It Work?

This guide explains the project end-to-end: where the data comes from, how it is transformed, what the two-tower model does, how we train it, and how we evaluate/interpret the results. Treat it as a living manual for engineers, data scientists, and product stakeholders.

> **In a hurry?**
> - Data → indexed & filtered → feature tensors  
> - Towers → ID embeddings + metadata encoders with adaptive mimic fusion  
> - Training → sampled BCE with hybrid optimisers  
> - Evaluation → ranking metrics, embedding diagnostics, feature correlations, qualitative audits → Markdown report

---

## 1. System Overview

```mermaid
flowchart LR
    A[Raw CSVs<br/>books.csv & users.csv] --> B[Loaders<br/>(limits + schema checks)]
    B --> C[Preprocessing<br/>filters + feature matrices]
    C --> D[Training Dataset]
    D --> E[Tower Builders<br/>ID + metadata encoders]
    E --> F[Training Loop<br/>negative sampling + BCE]
    F --> G[Evaluation & Diagnostics]
    G --> H[Report<br/>artifacts/reports/*.md]
```

Each step is covered below with “technical nuts & bolts” and “why it matters” callouts.

---

## 2. Data Layer

| Component | What it does | Why it matters |
|-----------|--------------|----------------|
| **Raw CSVs** (`data/books.csv`, `data/users.csv`) | Amazon kitap metaverisi + kullanıcı etkileşimleri | Git’e alınmıyor; herkes dosyaları yerel olarak sağlamalı. Trimmed versiyonlar hızlı denemeler içindir. |
| **Configuration** (`configs/default.yaml:data`) | Dosya yolları, satır limitleri (`books_limit`, `interactions_limit`), min etkileşim eşikleri, özellik parametreleri | Tek yerden deney yönetimi; farklı senaryolar için YAML kopyalayabilirsiniz. |
| **Loaders** (`src/data/loaders.py`) | CSV okur, limit uygular, kitap tablosunda bulunmayan etkileşimleri log’layarak eler | Veri bütünlüğü için “referanssız etkileşim” kalmaz. |
| **Preprocessing** (`src/data/preprocessing.py`) | Etkileşimleri temizler, min kullanıcı/ürün frekans eşiklerini iteratif uygular, indeks map’leri + özellik matrisleri üretir, `TrainingDataset` döner | Gürültülü (çok az etkileşimi olan) kullanıcı/ürünleri temizlemek doğruluğu artırır ve hesaplamayı ucuzlatır. |
| **Feature Engineering** (`src/data/features.py`) | Nümerik z-skorlar, başlık kelime/karakter istatistikleri, kategori & yazar sinyalleri, kullanıcı özetleri | Metadata sinyallerini modele taşır; cold-start durumlarında hayat kurtarır. |
| **Helpers** (`indexers`, `datasets`, `samplers`) | Map yapıları, PyTorch `Dataset`, negatif örnekleme | Eğitim ve değerlendirme döngülerinin temel taşı. |

> **Why this design?**  
> Bütün veriyi GPU/CPU’ya körlemesine yüklemek yerine, hızlıca örnekleyip filtreliyor; böylece ilk denemelerde bile dakikalar içinde sonuca ulaşabiliyoruz.

---

## 3. Model Layer

### Two-Tower Architecture (`src/models`)

| Parça | Teknik Özet | Önemi |
|-------|-------------|-------|
| **ID Embeddings** (`build_id_embedding`) | Kullanıcı ve ürün için 96 boyutlu, sparse destekli embedding tabloları | Kolayca ölçeklenebilir; SparseAdam ile bellek dostu |
| **Feature Encoders** (`build_feature_encoder`) | Metadata vektörünü 192 → 96 MLP ile kodlar (dropout opsiyonlu) | Metadata ile collaborative sinyalleri dengelememize yardımcı |
| **Adaptive Mimic** (`AdaptiveMimicModule`) | ID + metadata embedding’lerini kapı (gate) ağıyla birleştirir | Cold-start kullanıcı/ürünlerde en önemli hamle |
| **Tower Wrapper** (`TowerEncoder`) | Fusion modu (`identity/sum/concat/adaptive_mimic`) + cihaz yönetimi | Config üzerinden kule mimarisini değiştirmeyi basitleştirir |
| **TwoTowerModel** | İki kuleyi tek arayüzde buluşturur, benzerlik (cosine veya dot) döner | Eğitim/evaluation kodu kulelerin detayını bilmek zorunda değil |

**Technical note:**  
Sparseli embedding’ler için `padding_idx`, `max_norm` gibi opsiyonlar var; ama `sparse=True` iken `max_norm` kullanamazsınız. Config bu kombinasyonları kontrol eder.

---

## 4. Training & Evaluation Pipeline

### 4.1 Training (`src/pipelines/training.py`)

1. **Seed & Config Parse** – reproducible deneyler için `experiment.seed`.
2. **Dataset Build** – min etkileşim eşikleri, özellik matrisleri, log’lar.
3. **Train/Validation Split** – kullanıcı başına en son etkileşimi validation’a ayırır (timestamp yoksa uyarı verir ve validation’ı atlar).
4. **Tower Construction** – config → builder → `TowerEncoder` (boyutlar log’lanır).
5. **Optimiser Setup** – dense parametreler için Adam/AdamW/SGD, embedding’ler için SparseAdam.
6. **Mini-batch Loop**  
   - Negatif örnekleme (`sample_negative_items`)  
   - Kullanıcı/ürün embedding’leri, pozitif & negatif logit hesapları  
   - BCE loss + backward + optimizer step
7. **Epoch Logları** – `loss`, dataset count, tower boyutları.

### 4.2 Evaluation & Diagnostics

| Adım | Açıklama |
|------|----------|
| **Ranking Metrics** | `_evaluate_model` limited aday havuzu (pozitif + rastgele negatif) kullanır, Recall/Precision/NDCG/MAP hesaplar. |
| **Embedding Diagnostics** | Norm dağılımları, item komşuluk kategorileri, kullanıcı embedding ↔ metadata uyumu. |
| **Feature Correlations** | Pearson r + p-value ile skor vs. metadata ilişkisi; en anlamlı özellikler raporda listelenir. |
| **Qualitative Recommendations** | Örnek kullanıcılar için top-K öneriler, kategori/yazar isabet oranı. |
| **Markdown Report** | Yukarıdakilerin hepsi `artifacts/reports/recommendation_report.md` dosyasında toplanır. |

> **Tip:** Aday havuzu rastgele örneklerden oluştuğu için `evaluation.candidate_samples` değerini yükselterek doğruluğu artırabilirsiniz (daha pahalı).

---

## 5. Configuration Cheat Sheet

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

## 6. Practical Tips & Caveats

| Senaryo | Yapılması gereken | Neden |
|---------|-------------------|-------|
| **Büyük veri dosyaları** | `books_limit` / `interactions_limit` ile önce küçük örnekler deneyin | Bellek taşmalarını önler |
| **Aşırı filtre sonucu veri kalmadı** | `min_user/item_interactions` değerlerini düşürün veya daha fazla satır yükleyin | Subsample edilmiş veriyle 5/10 eşiği agresif olabilir |
| **Kategori/Author feature genişliği** | One-hot yerine embedding (ör. 32-64 boyut) planlayın | Parametre yükü azalır, semantik benzerlik yakalanır |
| **SparseAdam kullanımı** | `sparse=True` olmayan embedding’lerde çalışmaz | Yanlış kombinasyon training’i patlatır |
| **Raporları paylaşma** | Markdown → HTML/PDF dönüştürmek kolay; CI’de otomatik üretilebilir | Ekip içi görünürlük artar |

---

## 7. Quick Metrics Dashboard

| Metrik | Anlamı | İzleme önerisi |
|--------|--------|----------------|
| **Recall@K / Precision@K** | Önerilerin doğruluğu | Popülerlik baselines’ı ile kıyasla |
| **NDCG@K** | Doğru ürünlerin üst sıralara yerleşmesi | Düşüş varsa skor fonksiyonunu gözden geçir |
| **Hit Rate@K** | Kullanıcıya en az bir doğru ürün önerilmiş mi | Müşteri memnuniyeti proxy’si |
| **MAP@K** | Toplam sıralama kalitesi | Özellikle uzun kuyruk kullanıcılar için değerli |
| **Feature Corr (r/p)** | Metadata sinyallerinin skorla ilişkisi | Anlamsız (yüksek p) özellikleri temizle |
| **Category/Author Match** | Kullanıcının tarihi tercihlerine uyum | Adaptive mimic veya metadata pipeline’ını tune et |

---

## 8. Next Enhancements

1. Kategori/yazar embedding’lerine geçiş ve hashing/regularisation stratejileri.  
2. Checkpoint + early stopping + hiperparametre taramaları için otomasyon.  
3. Deney izleme (MLflow, W&B) + raporların CI pipeline’ına eklenmesi.  
4. Servisleşme: eğitilmiş kuleleri, indeks map’leri ile birlikte REST/GRPC API’ye taşımak.

---

## 9. Glossary

- **Two-Tower Model:** Kullanıcı ve ürün embedding’lerinin benzerlik üzerinden karşılaştırıldığı öneri mimarisi.  
- **Negative Sampling:** Rastgele eşleşme üretip modele “bu etkileşim olmadı” diyerek ayrım gücü kazandırmak.  
- **Adaptive Mimic:** Metadata ve ID embedding’lerini dinamik olarak harmanlayan kapı mekanizması.  
- **Hybrid Optimisation:** Aynı anda hem dense (Adam/AdamW) hem sparse (SparseAdam) parametreleri optimize etme.  
- **Ranking Metrics:** Recommendation kalitesini ölçen metrik seti (Recall, Precision, NDCG, MAP, Hit Rate).  
- **Feature Correlation Test:** Skor ile özellik kolonu arasındaki Pearson r/p-value analizi; aşırı/yan etkileri tespit eder.

---

## 10. Quick Start Checklist

1. `pip install -e .[dev]` ile bağımlılıkları kur.
2. `data/` klasörüne `books.csv` & `users.csv` dosyalarını yerleştir.
3. `python scripts/preprocess.py --config configs/default.yaml` çalıştır; log’lardaki kullanıcı/ürün sayısını kontrol et.
4. `python scripts/train.py --config configs/default.yaml` ile eğit.
5. `artifacts/reports/recommendation_report.md` dosyasını incele – metrikler, embedding diagnostikleri, feature corr tabloları ve örnek öneriler burada.

Hazırsınız – iyi deneyler!

