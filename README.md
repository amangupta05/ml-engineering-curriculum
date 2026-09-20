# The Machine Learning Engineer's Curriculum

A general, complete reference for what a machine learning engineer should know. Forty-six chapters, each taking its topic from zero to the level where senior engineers disagree.

Not tied to any company, any job description, or any individual. Equally useful to someone at a sensing startup, a bank, an advertising platform, or a search company.

**Start with [Chapter 30](30-Learning-Paths-Progression-and-Interviews.md).** It has six reading paths through the book by goal, so you read the 30 percent that serves you rather than all of it in order.

## The four levels, which is what makes this beginner to advanced

Every chapter is organised the same way, so you can enter any topic at zero and stop at whatever depth you need.

| Level | What it gives you | Who stops here |
|---|---|---|
| **1 Foundations** | What the topic is, what problem it solves, the mental model. No jargon without definition | Anyone who needs to hold a conversation about it |
| **2 Working knowledge** | Standard practice, default choices, the tools, the workflow, the mistakes everyone makes first | Anyone who needs to do it competently |
| **3 Depth** | The mechanism, the mathematics, the failure modes, the trade-offs that appear at scale | Anyone who must debug it or decide without a checklist |
| **4 Mastery** | What senior engineers argue about, where standard advice is wrong, the research frontier | Anyone leading the decision |

Each chapter then closes with a **subtopic checklist** for self-assessment, a **misconceptions table**, **practice exercises** with acceptance criteria, **interview questions with answers**, a **summary**, and **real references**.

## Contents

### Part I: Foundations

| Chapter | Covers |
|---|---|
| [1. Mathematics for Machine Learning](01-Mathematics-for-Machine-Learning.md) | Linear algebra through what ML uses, matrix calculus, optimisation, numerical stability, information theory |
| [2. Probability and Statistics](02-Probability-and-Statistics.md) | Distributions, estimation, hypothesis testing, power, confidence intervals, multiple comparisons, Bayesian methods, distribution comparison |
| [3. Python for ML Engineering](03-Python-for-Machine-Learning-Engineering.md) | The data model, typing, performance and vectorisation, memory, concurrency, data libraries, packaging, testing, reproducibility |
| [4. Classical Machine Learning](04-Classical-Machine-Learning.md) | Linear models, trees, ensembles in depth, clustering, dimensionality reduction, anomaly detection, calibration, imbalance, interpretability |
| [5. Evaluation, Validation, and Experimental Design](05-Evaluation-Validation-and-Experimental-Design.md) | Metric choice as a cost decision, ranking metrics, cross-validation design, leakage, model comparison, hyperparameter search, online experiments |

### Part II: Deep Learning

| Chapter | Covers |
|---|---|
| [6. Neural Networks and How They Train](06-Neural-Networks-and-How-They-Train.md) | Backpropagation, initialisation, normalisation, optimisers, regularisation, mixed precision, debugging a training run |
| [7. Architectures](07-Architectures-Convolutional-Recurrent-and-Attention.md) | Convolutions and receptive fields, recurrence and its limits, attention derived, efficient attention, graph networks |
| [8. Transformers and Language Models](08-Transformers-and-Language-Models.md) | The block assembled, tokenisation, pretraining objectives, scaling, inference and the key-value cache, sampling, serving arithmetic |
| [9. Representation Learning and Transfer](09-Representation-Learning-Transfer-and-Self-Supervision.md) | Embeddings, autoencoders, contrastive learning, transfer and fine-tuning, parameter-efficient adaptation, multi-task learning |

### Part III: Data and Modalities

| Chapter | Covers |
|---|---|
| [10. Feature Engineering](10-Feature-Engineering.md) | Numeric, categorical, datetime, text and aggregation features, missing data, leakage in depth, selection, versioning |
| [11. Time-Series and Sequential Modelling](11-Time-Series-and-Sequential-Modelling.md) | Stationarity, classical forecasting, tree-based forecasting, deep sequence models, probabilistic forecasting, anomaly detection, validation without leakage |
| [12. Signals and Sensor Data](12-Signals-and-Sensor-Data.md) | Sampling and aliasing, the frequency domain, filtering, artifacts, windowing, feature families, sensor fusion, subject-wise validation |
| [13. Vision, Audio, and Multimodal Fusion](13-Vision-Audio-and-Multimodal-Fusion.md) | Vision and video tasks, audio representations, alignment, the fusion taxonomy, cross-modal attention, missing modalities, proving fusion earned its cost |
| [14. Recommenders, Ranking, and Causal Inference](14-Recommenders-Ranking-and-Causal-Inference.md) | Collaborative filtering, two-tower retrieval and ranking, learning to rank, bias, bandits, causal identification, uplift modelling |

### Part IV: Generative AI

| Chapter | Covers |
|---|---|
| [15. Large Language Models in Practice](15-Large-Language-Models-in-Practice.md) | The adaptation ladder, prompting, retrieval-augmented generation in depth, fine-tuning, preference optimisation, evaluation, inference economics |
| [16. Building Generative AI Applications](16-Building-Generative-AI-Applications.md) | Orchestration, tool use, agents and durable execution, context management, guardrails, prompt injection, observability, interfaces |

### Part V: Data Engineering

| Chapter | Covers |
|---|---|
| [17. Data Storage, Formats, and Modelling](17-Data-Storage-Formats-and-Modelling.md) | Columnar formats, compression, table formats, partitioning, data modelling, database engines, query optimisation, data contracts |
| [18. Distributed Computing with Spark](18-Distributed-Computing-with-Spark.md) | The execution model, Catalyst, the shuffle, skew, joins, memory, user-defined functions, Structured Streaming, Spark for ML |
| [19. Streaming and Real-Time Pipelines](19-Streaming-and-Real-Time-Pipelines.md) | Event versus processing time, windows, watermarks, delivery semantics, log-based messaging, stateful processing, backpressure |
| [20. Feature Stores and Data Quality](20-Feature-Stores-and-Data-Quality.md) | Training-serving skew, point-in-time correctness, store architecture, backfills, data validation, observability, lineage, privacy |

### Part VI: Platform and Infrastructure

| Chapter | Covers |
|---|---|
| [21. Machine Learning System Design](21-Machine-Learning-System-Design.md) | Requirements and capacity, the platform reference architecture, serving modes, backend integration, four worked designs |
| [22. Containers, Kubernetes, and Cloud](22-Containers-Kubernetes-and-Cloud.md) | Images, the Kubernetes object model from zero, jobs and cron jobs, scheduling, GPUs, probes, autoscaling, debugging, cloud services |
| [23. Training Infrastructure and Distributed Training](23-Training-Infrastructure-and-Distributed-Training.md) | Accelerator hardware, training memory arithmetic, throughput, data parallelism, sharding, model parallelism, checkpointing, cost |
| [24. Model Serving and Inference](24-Model-Serving-and-Inference.md) | Packaging, batching, hardware choice, graph optimisation and quantisation, multi-model serving, load testing, capacity, edge |

### Part VII: Production Machine Learning

| Chapter | Covers |
|---|---|
| [25. Model Lifecycle, Versioning, and Registries](25-Model-Lifecycle-Versioning-and-Registries.md) | What must be versioned, experiment tracking, the registry as a state machine, deployment patterns, retraining strategy, governance |
| [26. CI and CD for Machine Learning](26-Continuous-Integration-and-Delivery-for-Machine-Learning.md) | The ML testing pyramid, data and model tests, the pipeline stages, the evaluation gate, secrets, infrastructure as code |
| [27. Monitoring, Drift, and Retraining](27-Monitoring-Drift-and-Retraining.md) | What to monitor, the drift taxonomy and why only one kind forces a retrain, detection methods, delayed labels, alerting, incident response |
| [28. Reliability, Cost, Security, and Compliance](28-Reliability-Cost-Security-and-Compliance.md) | Service objectives and error budgets, resilience patterns, unit economics, the ML threat model, privacy, compliance, fairness |

### Part IX: MLOps in Depth

The production chapters above cover the lifecycle end to end. This part goes deeper on the eight disciplines that carry the most operational weight and that a single chapter cannot hold.

| Chapter | Covers |
|---|---|
| [31. Workflow Orchestration and Pipeline Engineering](31-Workflow-Orchestration-and-Pipeline-Engineering.md) | Directed acyclic graphs, scheduling semantics and the logical-versus-wall-clock date, idempotency and safe backfills, dependency patterns, dynamic pipelines, testing pipelines, orchestrator families and migration |
| [32. Experiment Tracking and Reproducibility](32-Experiment-Tracking-and-Reproducibility.md) | What to log, run organisation, the reproducibility spectrum, every source of non-determinism, environment and data capture, the research-to-production handoff, notebooks |
| [33. Testing Machine Learning Systems](33-Testing-Machine-Learning-Systems.md) | Testing code, data, models, pipelines and systems; behavioural and metamorphic tests, slice testing, regression tests built from incidents, golden datasets, flakiness |
| [34. Online Experimentation Platforms](34-Online-Experimentation-Platforms.md) | Assignment versus exposure, randomisation units, layered experiments, the metric repository, sample ratio mismatch, guardrails, experiment governance, long-term holdouts |
| [35. LLMOps, Operating Generative AI Systems](35-LLMOps-Operating-Generative-AI-Systems.md) | Versioning prompts and indexes and routing policies, the provider dependency problem, continuous evaluation, token cost attribution, trace design, guardrail operations, index operations |
| [36. Model Governance, Risk, and Compliance Operations](36-Model-Governance-Risk-and-Compliance-Operations.md) | The model inventory, risk tiering, documentation as a deliverable, independent validation, approval and change control, audit trails, the regimes at engineering depth |
| [37. Incident Response and On-Call for ML](37-Incident-Response-and-On-Call-for-Machine-Learning.md) | Silent incidents, severity when the service is up and the answers are wrong, runbooks, the diagnostic method, playbooks for the incidents that actually happen, postmortems |
| [38. MLOps Maturity, Team Topologies, and Platform Strategy](38-MLOps-Maturity-Team-Topologies-and-Platform-Strategy.md) | The maturity progression and what unlocks each stage, team models and their characteristic failures, the platform as a product, measuring a platform, technical debt, migration |

### Part X: Time-Series and Sequential Modelling in Depth

Chapter 11 introduces every topic below at one subsection each. This part is the depth behind it. Read chapter 11 first, then the chapters here that your problem actually stresses.

| Chapter | Covers |
|---|---|
| [39. Time-Series Analysis, Decomposition, and Stationarity](39-Time-Series-Analysis-Decomposition-and-Stationarity.md) | Decomposition methods, autocorrelation read properly, stationarity tests and their opposing null hypotheses, differencing, transformations and back-transform bias, calendar effects, outlier types, the pre-modelling checklist |
| [40. Classical Forecasting](40-Classical-Forecasting-Exponential-Smoothing-State-Space-and-ARIMA.md) | Exponential smoothing derived, the error-trend-season taxonomy, the state space form, ARIMA identification and diagnostics, dynamic regression, intermittent demand, and when classical beats machine learning |
| [41. Feature-Based and Global ML Forecasting](41-Feature-Based-and-Global-Machine-Learning-Forecasting.md) | Forecasting as supervised learning, local versus global models, feature construction and the cut-off discipline, recursive versus direct strategies, the tree extrapolation limit, a worked recipe |
| [42. Deep Learning Architectures for Time Series](42-Deep-Learning-Architectures-for-Time-Series.md) | When neural models earn their cost, temporal convolutions, basis expansion and attention families, the linear-model result and what it exposed, normalisation for non-stationary series, foundation models assessed honestly |
| [43. Probabilistic Forecasting and Uncertainty](43-Probabilistic-Forecasting-and-Uncertainty-Quantification.md) | Sources of uncertainty, quantile regression and the pinball loss, distributional heads, conformal prediction and why it breaks on time series, proper scoring rules, calibration, the newsvendor decision |
| [44. Hierarchical, Multivariate, and Panel Forecasting](44-Hierarchical-Multivariate-and-Panel-Forecasting.md) | Why independent forecasts do not add up, reconciliation derived including the minimum trace solution, vector autoregression, factor models, panel data, forecasting at the scale of many series |
| [45. Classification, Clustering, and Anomaly Detection](45-Time-Series-Classification-Clustering-and-Anomaly-Detection.md) | Dynamic time warping derived, shapelets and dictionary and kernel methods, representation learning, the anomaly taxonomy, thresholds from an alert budget, and why the standard evaluation protocol inflates results |
| [46. Forecasting Systems in Production](46-Forecasting-Systems-in-Production.md) | Backtesting infrastructure, the forecast store and why creation time and target time are both mandatory, retraining policy, the long tail of series, monitoring with delayed truth, forecast value added |

### Part VIII: Practice

| Chapter | Covers |
|---|---|
| [29. The Toolkit and How to Choose](29-The-Toolkit-and-How-to-Choose.md) | The landscape by job rather than vendor, selection criteria, build versus buy, the minimum viable platform, anti-patterns |
| [30. Learning Paths, Progression, and Interviews](30-Learning-Paths-Progression-and-Interviews.md) | Six reading paths, how to study, seniority, the first 90 days, the interview rounds and what each really tests, career mistakes |

## How to use it

| If you are | Read |
|---|---|
| Starting out | Path A in chapter 30: chapters 1, 2, 3, 4, 5, 10, 6, 21, 25, 27 |
| A data scientist moving to engineering | Path B: 3, 17, 18, 21, 22, 24, 25, 26, 27 |
| A backend engineer moving to ML | Path C: 2, 4, 5, 10, 6, 11, 27 |
| Taking on a platform | Path D: 17 through 28 in order |
| Building with language models | Path E: 8, 9, 15, 16, 24, 27, 28 |
| Interviewing in three to six weeks | Path F: 5, 4, 21, 27, then section 30.4 |

Passive reading is close to worthless. Derive the formulas by hand, do the practice exercises, and answer the interview questions out loud before opening them.

## Standards this material holds to

Every diagram renders on GitHub and has been checked in a browser. Every formula with practical use has a worked numerical example with its assumptions stated. No benchmark number, vendor price, dataset statistic, or version number is invented; where a figure is needed for an example it is marked as an assumption, and version-dependent behaviour is marked to check. Citations are real works by author and year, and where a title or date was uncertain the idea is described rather than guessed. Where practice is contested, the material says so and gives the practical default.

Run `python verify_curriculum.py` to check any chapter against these rules.

## Related material

| Need | Where |
|---|---|
| Deeper derivations for language model internals | [handbook](../handbook/README.md), 25 chapters |
| Architecture patterns and worked system designs | [AI_Engineering_System_Design.md](../AI_Engineering_System_Design.md) |
| Preparation for one specific role | [temple-ml-engineer](../temple-ml-engineer/README.md) |
| Defending your own experience in an interview | [experience-deep-dive](../experience-deep-dive/README.md) |
