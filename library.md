## **Library**:

- System Design: - Generation (Candidate Generation) - Ranking - Re-Ranking

* Python (Script & Notebook)
* JavaScript (Node & Browser?)

### Module: Datasets

Datasets

- Movies (Basic)
- Hacker News (Text)
- Mobile Phones (Items Only)
- Trivago (Session Based) https://recsys.trivago.cloud/challenge/dataset/

### Module: Load

- From CSV
- From Files
- From DB / Parquet (Arrow-based) _Later_

### Module: Encode

Features Encoders

- Index
- Binary
- Numerical
- Category
- Date
- Categorical (Hierachical) - Set / Bag (like tags)
- Sequence (??)
- Text (Tokenizer, Spacy) - Timeseries (Session) - Date
- Image
- Spatial (Lat + Lon, H3)
- Graph (later...)

- Good Defaults
- Transform Functions
- Cross-Products Functions
- Embedding Transformers

User & Item Features

- Domain Features
- ??
- ??

Interaction / Signal Encoders:

- Explicit (Strong Signal)
- Implicit (Weak Signal)
- Implicit Weighted Measure
- Signal Discrimination and Rasch-Andrich Thresholds (Adaptive Modelling)
- Negative Sampling
- ...

Splitting:

- Random
- Stratified
- Chrono

### Module: Model

- Baseline: Random, SAR
- 2Vec Models: Item2Vec
- Interaction Only: ALS, Low-Rank Approximation (Conjugate Gradient)
- Factorisation Model with Features (Explicit / Implicit)
- Neural Models (NCF, DMF)
- Ranking Models: Learning to Rank
- Context Models:
  - Spatial
  - Temporal
  - Social
- Graph Models
  - Knowledge Graphs
  - Curated Graphs
  - Domain Modelling
- Ensemble Models:
  - Sequential (Cascade, Feature Augmentation)
  - Parallel (Weighted, Switching)

### Module: Evaluate, Intrepret, Explain

- Losses: RMSE, BPR, ....
- Ranking@K metrics: Precision, Recall, MAP, NDCG, BPR, MRR
- Novelty:
- Serendipity:
- Coverage Metrics: ...
- Fairness, Accountability & Trust:
- Explaination: ...

### Module: Visualisation

- Architecture
- Weigths
- Training Visualisation
- Evaluation Visualisation
- Explaination Visualisation - Instance

### Module: Recommend

- Build Similarity: Approximate NN
- Similar Items
- Similar Users
- Given a user, give ranked items
- Given an item, give ranked items

### Module: System Design

- Interaction /Signal Update

  - New Items?
  - New User?
  - Existing User + New Signal
  - Learning New Preference -> Explore vs. Exploit

- Update for Objectives

  - Margin Optimisation
  - Weights? Discovery -> Novelty

- Set of Models
  - Generation Model
  - Ranking Model
  - Rules / Re-Ranking Model

### Module: Serve

- Orchestrate
- Serving: Multiple
- Memory cached
- API
- A/B testing

## Additiona points to Cover

- Full Page Optimisation
- UI linkage
- Loop --- How to re-train, Add one new user or item ...
