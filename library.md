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

Interaction / Signal Encoders:

- Explicit (Strong Signal)
- Implicit (Weak Signal)
- Implicit Weighted Measure
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
- Sequence Models:
- Graph Models
- Ensemble Models:
  - Sequential
  - Parallel

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

### Module: Serve

- Orchestrate
- Serving: Multiple
- Memory cached
- API
- A/B testing

## Additiona points to Cover

- Ensemble
- Generation / Ranking / Re-Ranking
- Twin Tower
- Full Page Optimisation
- A/ B Testing
- UI linkage
- Loop --- How to re-train, Add one new user or item ...
