# Book & Guide

## Key Concepts

- **Theory**: ML & DL Formulation, Prediction vs. Ranking, Similiarity, Biased vs. Unbiased
- **Paradigms**: Content-based, Collaborative filtering, Knowledge-based, Hybrid and Ensembles
- **Data**: Tabular, Images, Text (Sequences)
- **Models**: (Deep) Matrix Factorisation, Auto-Encoders, Wide & Deep, Rank-Learning, Sequence Modelling
- **Methods**: Explicit vs. implicit feedback, User-Item matrix, Embeddings, Convolution, Recurrent, Domain Signals: location, time, context, social,
- **Process**: Setup, Encode & Embed, Design, Train & Select, Serve & Scale, Measure, Test & Improve
- **Tools**: python-data-stack: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `tfranking`, `implicit`, `spacy`

## Chapter 1: Introduction

- Why build recommendation systems?
  - Scope and evolution of recsys
  - Prediction and Ranking
  - Relevance, novelty, serendipity & diversity
- Similiarity
  - items
  - users / groups
  - location
  - time
  - context
  - social
- Systems & Loops

  - UI / Signals
  - Search
  - Experience Design
  - Interaction Design

- Paradigms in recommendations:
  - Content-based
  - Collaborative filtering
  - Knowledge-based
  - Hybrid and Ensembles
- Key concepts in recsys:
  - Explicit vs. implicit feedback
  - User-Item matrix
  - Domain signals: location, time, context, social
- Why use deep learning for recsys?
  - Primer on deep learning
  - Traditional vs deep learning approaches
  - Examples and use-cases

### Chapter 2: Content-Based

- Introduction to the case #1: product recommendation
- Feature extraction using deep learning: Embeddings for Hetrogenous data
- _Exercise: Recommending items using similarity measures_

### Chapter 3: Colloborative-Filtering

- Overview of traditional Colloborative-Filtering for recsys
- Primer on deep learning approaches
  - Deep matrix factorisation
  - Auto-Encoders
- _Exercise: Recommending items using Colloborative-Filtering_

### Chapter 4: Learning-to-Rank

- Why learning-to-rank? Prediction vs Ranking
- Rank-learning approaches: pointwise, pairwise and listwise
- Deep learning approach to combine prediction and ranking
- _Exercise: Recommending items using Learning-to-Rank_

### Chapter 5: Hybrid Recommender

- Introduction to the case #2: text recommendation
- Combining content-based and collaborative filtering
- Primer on Wide & Deep Learning for Recommender Systems
- _Exercise: Recommending items using Hybrid recommender_

### Chapter 6: Time and Context

- Adding temporal component: window and decay-based
- Adding context context through group recommendations
- Dynamic and Sequential modelling using Recurrent Neural Networks
- _Exercise: Recommending items using RNN recommender_

### Chapter 7: Deployment & Monitoring

- Deploying the recommendation system models
- Measuring improvements from recommendation system
- Improving the models based on the feedback from production
- Architecture design for recsys: Offline, Nearline and Online

### Chapter 8: Evaluation, Challenges & Way Forward

- A/B testing for recommendation systems
- Challenges in recsys:
  - Building explanations
  - Model debugging
  - Scaling-out & up
  - Fairness, accountability and trust
- Bias in recsys: training data, UI → Algorithm → UI, private
- When not to use deep learning for recsys
- Recap and next steps, Learning Resources
