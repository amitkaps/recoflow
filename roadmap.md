# Design Philosophy

Input -> CSV
Output -> Working Recsys

- System Design: 
	- Generation (Candidate Generation)
	- Ranking
	- Re-Ranking

# Modules

- Data Processing: 
	- Encoding & Decoding: Index, Continuous, Categorical, Text, Image 
	- Splitting: Random, Stratified, Chrono
- Model Building
	- Embedding
		- Bi-linear
		-
	- Architectures
		- 2vec models
		- Factorisation Model: Explicit, Implicit
		- Neural Models
		- Sequence Models
	- Losses
		- RMSE
		- BPR
		- ....
- Ranking Metrics
	- @K metrics: Precision, Recall, MAP, NDCG, BPR, MRR
	- Novelty
	- Coverage Metrics
- Recommendation
	- Similar Items
	- Similar Users
	- Given a user, give ranked items 
	- Given an item, give ranked items
	- Explainability
- Serving:
	- Memory cached
	- API
	- A/B testing
