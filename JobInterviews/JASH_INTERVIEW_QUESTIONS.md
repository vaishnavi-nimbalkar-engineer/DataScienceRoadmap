# Interview Questions

## 1. Python Fundamentals & Pandas

### 1.1 Explain the difference between a list, tuple, set, and dictionary in Python.

#### Answer 1:
A list is an ordered, mutable sequence allowing duplicates; a tuple is ordered and immutable; a set is an unordered collection of unique items; and a dictionary stores key–value pairs with unique keys.

#### Answer 2:
Lists ([]) and tuples (()) both preserve order, but only lists can be modified. Sets ({}) drop duplicates and lack order, while dicts ({key: value}) map unique keys to values.

### 1.2 How do list comprehensions differ from generator expressions? Give an example of each.

#### Answer 1:
List comprehensions produce an in-memory list immediately (squares = [i*i for i in range(5)]), while generator expressions yield items on demand (squares_gen = (i*i for i in range(5))).

#### Answer 2:
A list comp returns [] with all values stored, e.g. [x for x in iterable]; a generator uses () and is lazy, e.g. (x for x in iterable), saving memory for large iterables.

### 1.3 What are Python decorators and how would you use them?

#### Answer 1:
Decorators are callables that wrap a function to modify its behavior; e.g., @timer can log execution time around a function.

#### Answer 2:
They apply higher-order functions via the @ syntax; for instance, @login_required checks authentication before the wrapped view runs.

### 1.4 Describe context managers and the with statement. When would you implement a custom context manager?

#### Answer 1:
Context managers define __enter__/__exit__ to manage resources (like files) in a with block; you implement one when you need guaranteed setup/teardown, e.g. database transactions.

#### Answer 2:
The with statement uses a context manager to ensure cleanup; you’d write a custom one if you need deterministic resource release, such as locking or network connections.

### 1.5 In Pandas, what is the difference between .loc[] and .iloc[]?

#### Answer 1:
.loc[] selects by label (row/column names), while .iloc[] selects by integer position.

#### Answer 2:
Use .loc[row_label, col_label] for explicit index names and .iloc[row_index, col_index] for zero-based index positions.

### 1.6 How do you handle missing or NaN values in a DataFrame?

#### Answer 1:
You can drop them with df.dropna(), fill with df.fillna(value), or interpolate using df.interpolate().

#### Answer 2:
Strategies include imputation (df['col'].fillna(df['col'].mean())), forward/backward fill (ffill/bfill), or model-based prediction for missing values.

### 1.7 Explain the difference between vectorized operations and using .apply() in Pandas—when might one be preferable?

#### Answer 1:
Vectorized operations leverage C loops under the hood and are much faster, e.g. df['x'] + df['y']; use .apply() when you need row- or element-wise custom Python logic.

#### Answer 2:
Vectorization (df.mean()) uses low-level routines for speed, whereas .apply() invokes Python functions per element/row; prefer vectorized methods for performance and .apply() for complex transformations.

### 1.8 How would you merge two DataFrames on multiple columns and handle keys that don’t match?

#### Answer 1:
Use pd.merge(df1, df2, on=['A','B'], how='outer') to include non-matching keys, or how='left'/'right' to keep one side.

#### Answer 2:
Specify on=['col1','col2'] and choose how='inner'/'outer' as needed; unmatched rows get NaNs in outer joins.

### 1.9 Describe how grouping and aggregation works in Pandas. How can you apply multiple aggregation functions at once?

#### Answer 1:
df.groupby('key')['val'].agg(['mean','sum']) groups rows by key and computes both mean and sum in one call.

#### Answer 2:
Call groupby on columns, then .agg({'col1':'sum','col2':['min','max']}) to apply different functions to different columns simultaneously.

### 1.10 How do you optimize memory usage when working with very large DataFrames?

#### Answer 1:
Downcast numeric types with pd.to_numeric(..., downcast='float'), convert object columns to categorical, and load data in chunks with read_csv(chunksize=…).

#### Answer 2:
Use efficient dtypes, drop unnecessary columns early, and consider sparse or Dask DataFrames for out-of-core processing.

## 2. Machine Learning with scikit-learn

### 2.1 Explain the bias–variance tradeoff and how it influences model selection.

#### Answer 1:
High bias (underfitting) means the model is too simple, high variance (overfitting) means too complex; the goal is to find a balance to minimize total error.

#### Answer 2:
We choose simpler models when variance dominates and more flexible models when bias dominates, often guided by validation curve diagnostics.

### 2.2 How does Pipeline help in building robust ML workflows? Give an example.

#### Answer 1:
A Pipeline chains preprocessing and estimator steps so transformations and training happen consistently, e.g., Pipeline([('scaler', StandardScaler()), ('clf', SVC())]).

#### Answer 2:
It encapsulates data preprocessing and modeling into one object, preventing leakage and simplifying grid search with parameter names like clf__C.

### 2.3 Describe how GridSearchCV and RandomizedSearchCV work. When would you choose one over the other?

#### Answer 1:
GridSearchCV exhaustively tests all parameter combinations; use it when the grid is small. RandomizedSearchCV samples parameter space randomly, ideal for large or continuous search spaces.

#### Answer 2:
Grid search guarantees finding the best combination in the grid but is expensive; randomized search trades exhaustiveness for speed when parameter options are many.

### 2.4 What are the differences between Random Forests and Gradient Boosting Machines?

#### Answer 1:
Random Forests build trees in parallel on bootstrap samples and average predictions; GBMs build trees sequentially to correct previous errors via gradient descent.

#### Answer 2:
RF reduces variance by bagging; GBM reduces bias by boosting, often requiring more careful tuning of learning rate and number of trees.

### 2.5 How do you handle imbalanced classes when training a classifier in scikit-learn?

#### Answer 1:
Use class_weight='balanced' in the estimator, resample the dataset (SMOTE or undersampling), or adjust decision thresholds.

#### Answer 2:
Techniques include oversampling the minority class, undersampling the majority, or using ensemble methods that focus on hard examples.

### 2.6 Explain the purpose of cross-validation and how you’d implement nested cross-validation.

#### Answer 1:
Cross-validation estimates generalization error by splitting data into k folds; nested CV wraps an inner loop for hyperparameter tuning and an outer loop for performance estimation.

#### Answer 2:
You perform GridSearchCV inside each fold of a higher-level cross_val_score, ensuring unbiased performance estimates by separating tuning from evaluation.

### 2.7 How would you implement custom feature selection within a scikit-learn pipeline?

#### Answer 1:
Write a transformer with fit and transform methods (inheriting BaseEstimator, TransformerMixin) and include it as a step in Pipeline.

#### Answer 2:
Use SelectKBest(score_func=my_func) or RFE(estimator=my_model) directly in the pipeline, specifying parameters in GridSearchCV.

### 2.8 Discuss at least three evaluation metrics for regression and three for classification, and when to use each.

#### Answer 1:
Regression: MSE (sensitive to large errors), MAE (robust to outliers), R² (variance explained). Classification: Accuracy (balanced classes), ROC AUC (ranking quality), F1 score (imbalanced classes).

#### Answer 2:
Use RMSE for originally scaled errors, MAE when all errors are equally weighted, and adjusted R² for multiple predictors; for classifiers, precision/recall balance vs. ROC AUC tradeoff.

### 2.9 How can you save and load a trained scikit-learn model for production?

#### Answer 1:
Use joblib.dump(model, 'model.joblib') and model = joblib.load('model.joblib').

#### Answer 2:
Serialize with pickle (pickle.dump) or sklearn.externals.joblib, ensuring package versions match when loading.

### 2.10 Describe how ensemble methods like voting and stacking work and when to use them.

#### Answer 1:
Voting averages (hard or soft) multiple model predictions; use when you have diverse base estimators. Stacking trains a meta-model on base model outputs to learn optimal combinations.

#### Answer 2:
Voting is a simple weighted/unweighted consensus; stacking adds a second-level learner to exploit strengths of base learners, beneficial when models capture different data aspects.

## 3. Deep Learning with TensorFlow

### 3.1 Compare TensorFlow’s eager execution mode with graph mode.

#### Answer 1:
Eager mode runs operations immediately for intuitive debugging; graph mode builds static computation graphs optimized for deployment and performance.

#### Answer 2:
Graph mode (tf.function) compiles a callable graph for speed, while eager is Pythonic and interactive but slightly slower.

### 3.2 What are the main differences between the Sequential and Functional APIs in Keras?

#### Answer 1:
Sequential stacks layers linearly; Functional allows building complex graphs with multiple inputs/outputs or shared layers.

#### Answer 2:
Use Sequential for simple feedforward models; use Functional for branching, residual connections, or multi-input architectures.

### 3.3 How would you write a custom training loop in TensorFlow?

#### Answer 1:
Use tf.GradientTape() inside an epoch loop to compute gradients, then apply them with an optimizer’s apply_gradients.

#### Answer 2:
Manually iterate batches: record forward pass in GradientTape, compute loss, call tape.gradient, and optimizer.step().

### 3.4 Explain how backpropagation works under the hood in a neural network.

#### Answer 1:
During the backward pass, gradients of the loss w.r.t. weights are computed via the chain rule, propagating errors layer by layer.

#### Answer 2:
Each tf.Variable records operations in the forward pass; GradientTape traces them and computes partial derivatives automatically.

### 3.5 What’s the vanishing gradient problem and how do techniques like batch normalization and ReLU help mitigate it?

#### Answer 1:
In deep nets, gradients shrink through many layers, slowing learning; ReLU maintains gradients by avoiding saturation, and batch norm standardizes inputs to layers for stable gradients.

#### Answer 2:
Vanishing gradients make early layers learn slowly; using non-saturating activations (ReLU/LeakyReLU) and normalizing layer inputs preserves gradient magnitude.

### 3.6 How do you apply callbacks such as EarlyStopping and ModelCheckpoint during training?

#### Answer 1:
Pass them to model.fit(..., callbacks=[tf.keras.callbacks.EarlyStopping(patience=2), tf.keras.callbacks.ModelCheckpoint('best.h5', save_best_only=True)]).

#### Answer 2:
Define callback instances and include in the callbacks list to monitor val_loss or val_accuracy, stopping or saving the model based on conditions.

### 3.7 Describe how to perform transfer learning with a pretrained model in TensorFlow.

#### Answer 1:
Load a base model with include_top=False, freeze its layers (layer.trainable=False), add new classification head, then train only the head.

#### Answer 2:
Use tf.keras.applications to import a pretrained network, replace the output layer, optionally unfreeze some top layers, and fine-tune with a low learning rate.

### 3.8 How do you monitor training with TensorBoard? What kinds of visualizations are most useful?

#### Answer 1:
Instantiate tf.keras.callbacks.TensorBoard(log_dir='logs'), run tensorboard --logdir logs, and view graphs of loss, metrics, and histograms of weights.

#### Answer 2:
Log scalars (loss, accuracy), images, and computational graphs; the histograms and projector tabs help inspect layer activations and embeddings.

### 3.9 Explain how to distribute training across multiple GPUs or machines using tf.distribute.Strategy.

#### Answer 1:
Wrap model creation and compilation within a strategy scope, e.g.

```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    model.compile(...)
```

#### Answer 2:
Choose a strategy (MirroredStrategy, MultiWorkerMirroredStrategy), define dataset with strategy.experimental_distribute_dataset, and use strategy.run(train_step, args=(…)).

### 3.10 How do you export a TensorFlow model for serving (e.g., SavedModel vs. TFLite vs. TensorFlow.js)?

#### Answer 1:
Use model.save('export_dir', save_format='tf') for SavedModel; convert to TFLite with TFLiteConverter, and to TF.js with tensorflowjs_converter.

#### Answer 2:
SavedModel retains full graph and variables, ideal for TF Serving; TFLite targets mobile/edge; TensorFlow.js export allows inference in browsers via JavaScript.

## 4. Deep Learning with PyTorch

### 4.1 How does PyTorch’s dynamic computation graph differ from TensorFlow’s static graph?

#### Answer 1:
PyTorch builds the computation graph on the fly during the forward pass, enabling Pythonic control flow and easy debugging.

#### Answer 2:
TensorFlow’s static graph (pre-2.x) requires building and compiling a graph ahead of time, while PyTorch’s eager execution is dynamic and interactive.

### 4.2 Explain how autograd works in PyTorch to compute gradients.

#### Answer 1:
Tensors track operations with a DAG; calling .backward() traverses this graph backward, accumulating gradients in .grad attributes.

#### Answer 2:
Each torch.Tensor has requires_grad; operations create Function objects that record inputs and outputs, enabling automatic differentiation.

### 4.3 Describe how to build a custom torch.utils.data.Dataset and DataLoader.

#### Answer 1:
Subclass Dataset, implement __len__ and __getitem__, then wrap it in DataLoader(dataset, batch_size, shuffle=True).

#### Answer 2:
Provide data and labels in your Dataset’s __getitem__, optionally apply transforms, and let DataLoader handle batching and parallel loading.

### 4.4 Outline the components of a typical PyTorch training loop and where you’d place optimizer.zero_grad(), loss.backward(), and optimizer.step().

#### Answer 1:
In each batch: call optimizer.zero_grad() first, then compute output = model(input), loss = criterion(output, target), loss.backward(), and finally optimizer.step().

#### Answer 2:
Zero grads to clear previous epoch, forward pass to get predictions, compute loss, backward pass to compute gradients, and step to update weights.

### 4.5 How do you move models and data between CPU and GPU in PyTorch?

#### Answer 1:
Call .to(device) on model and data tensors, e.g. device = torch.device('cuda'); model.to(device); data = data.to(device).

#### Answer 2:
Use .cuda() on modules and tensors or .cpu() to move back; ensure both model and inputs share the same device.

### 4.6 Demonstrate how to implement a custom nn.Module layer.

#### Answer 1:
```python
class MyLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_feats, out_feats))
    def forward(self, x):
        return x @ self.weight
```

#### Answer 2:
Define __init__ for parameters and layers, and implement forward(self, x) to specify computations, registering any nn.Parameter or submodules.

### 4.7 What are common techniques for preventing overfitting in PyTorch models?

#### Answer 1:
Use dropout layers (nn.Dropout), weight decay in the optimizer, and early stopping based on validation loss.

#### Answer 2:
Augment data, apply batch normalization, and reduce model complexity (fewer layers/units).

### 4.8 Explain how to save and load model weights and entire model architectures.

#### Answer 1:
Save state dict with torch.save(model.state_dict(), path) and load via model.load_state_dict(torch.load(path)).

#### Answer 2:
For full model serialization, torch.save(model, 'model.pt') and load with model = torch.load('model.pt'), though state dict is preferred for flexibility.

### 4.9 How would you export a PyTorch model for inference in a non-Python environment (e.g., via TorchScript)?

#### Answer 1:
Script the model with ts_model = torch.jit.script(model) or trace with torch.jit.trace, then save ts_model.save('model.pt').

#### Answer 2:
Use ONNX export (torch.onnx.export) to generate an interoperable model file for other runtimes.

### 4.10 Describe gradient clipping and when it is necessary.

#### Answer 1:
Gradient clipping constrains gradient norms (torch.nn.utils.clip_grad_norm_) to prevent exploding gradients in RNNs or deep nets.

#### Answer 2:
It’s applied before optimizer.step() to cap gradients at a threshold, stabilizing training in architectures prone to large updates.

## 5. SQL & Database Performance Tuning

### 5.1 Explain the difference between OLTP and OLAP systems and give use-case examples.

#### Answer 1:
OLTP supports transactional workloads with many small reads/writes (e.g., banking), while OLAP handles analytical queries on large datasets (e.g., BI dashboards).

#### Answer 2:
OLTP ensures ACID compliance for operational systems; OLAP is optimized for complex aggregations and reporting.

### 5.2 How do indexes improve query performance? What are potential downsides of over-indexing?

#### Answer 1:
Indexes speed lookups by providing sorted pointers, reducing full scans; too many indexes slow down writes and consume storage.

#### Answer 2:
They create B-tree or hash structures for quick seeks; but each insert/update must maintain indexes, increasing overhead.

### 5.3 Walk through how you would interpret an EXPLAIN ANALYZE query plan in PostgreSQL.

#### Answer 1:
Review node costs, actual vs. estimated rows, and timing to identify slow operations or misestimates.

#### Answer 2:
Check for sequential scans vs. index scans, nested loops vs. hash joins, and look for high buffers or repeated sorts.

### 5.4 Compare and contrast inner joins, left/right outer joins, and full outer joins.

#### Answer 1:
Inner join returns only matching rows; left/right retains all from one side with nulls for non-matches; full outer retains both sides.

#### Answer 2:
Use inner to intersect, left/right to preserve primaries with optional matches, and full outer for union-like merging including unmatched.

### 5.5 How do window functions work? Provide an example use case.

#### Answer 1:
Window functions compute over partitions without collapsing rows, e.g., ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC).

#### Answer 2:
They allow calculations like running totals (SUM(amount) OVER (ORDER BY date)) or ranking within groups.

### 5.6 When would you use a Common Table Expression (CTE) versus a subquery?

#### Answer 1:
Use CTEs for readability and reuse of intermediate results; subqueries are fine for simple, one-off filters.

#### Answer 2:
CTEs can improve clarity and enable recursion; subqueries may be inlined by the optimizer but can be harder to maintain.

### 5.7 What strategies can you use to optimize large table scans?

#### Answer 1:
Create appropriate indexes, partition the table, and filter early to reduce scanned rows.

#### Answer 2:
Use materialized views for repeated queries, increase parallel workers, and tune planner cost settings.

### 5.8 Explain table partitioning in PostgreSQL and when to employ it.

#### Answer 1:
Partition tables by range or list to split data into child tables, improving query performance and maintenance for large datasets.

#### Answer 2:
Use it when tables grow very large (billions of rows) and queries target specific partitions (e.g., by date).

### 5.9 How do you tune database parameters (e.g., work_mem, shared_buffers) for better performance?

#### Answer 1:
Adjust shared_buffers to 25% of system RAM, increase work_mem for complex queries, and set maintenance_work_mem for vacuum/index creation.

#### Answer 2:
Monitor slow queries, tune effective_cache_size to reflect OS cache, and use pg_stat_statements metrics to guide optimizations.

### 5.10 Describe ACID transactions and isolation levels; how do they affect concurrency?

#### Answer 1:
ACID ensures Atomicity, Consistency, Isolation, Durability; higher isolation (e.g., SERIALIZABLE) reduces anomalies but lowers concurrency.

#### Answer 2:
Read Uncommitted allows dirty reads, Read Committed avoids them, Repeatable Read guarantees snapshot consistency, and Serializable prevents phantom reads at the cost of throughput.

## 6. Data Engineering: Pipelines & Data Lakes/Warehouses

### 6.1 Compare ETL and ELT patterns. When would you choose one over the other?

#### Answer 1:
ETL transforms data before loading into target, suitable for smaller warehouses; ELT loads raw data then transforms in-dw, ideal for large-scale analytics engines.

#### Answer 2:
Use ETL when strict data quality is needed pre-load; choose ELT when you have powerful compute in the warehouse and need flexible transformations.

### 6.2 Describe how you would design an Airflow DAG for a daily batch pipeline.

#### Answer 1:
Define a DAG with schedule_interval='@daily', set start_date, create tasks (extract, transform, load) as operators, and set dependencies with >>.

#### Answer 2:
Use task retries and SLAs, group tasks with TaskGroups, and configure DAG-level default_args for retries and notifications.

### 6.3 What is idempotency in data pipelines, and how do you ensure it?

#### Answer 1:
Idempotent tasks produce the same result when run multiple times; ensure it by using upserts or writing to date-partitioned sinks.

#### Answer 2:
Design extract jobs to overwrite partitions atomically, use checkpoints, and make transformations deterministic.

### 6.4 How do you handle schema evolution when writing Parquet files?

#### Answer 1:
Use schema merging in Spark/Athena or maintain a schema registry; evolve by adding nullable fields and handling missing columns.

#### Answer 2:
Write schemas explicitly and apply Avro/Glue schema registries to manage versioning, ensuring backward compatibility.

### 6.5 Explain partitioning strategies for large datasets in a data lake.

#### Answer 1:
Partition by time (year/month/day) for time-series data, or by business keys (region, category) to prune data at query time.

#### Answer 2:
Use Hive-style directories, bucket small partitions, and avoid too many small files by compaction.

### 6.6 What is a feature store, and how does it integrate with ML pipelines?

#### Answer 1:
A feature store centralizes feature definitions and metadata, providing consistent, low-latency feature retrieval for training and serving.

#### Answer 2:
It manages feature computation pipelines, online/offline stores, and ensures reproducibility by versioning feature transformations.

### 6.7 How would you implement incremental data loads versus full refresh?

#### Answer 1:
Track watermark columns (e.g., updated_at), query only new/changed records, and append to target tables.

#### Answer 2:
Implement CDC (Change Data Capture) streams or use file manifests to process only deltas, falling back to full loads when necessary.

### 6.8 Compare Amazon S3, HDFS, and Azure Data Lake for storing raw and processed data.

#### Answer 1:
S3 and ADLS are object stores with native cloud integration and strong security features; HDFS is on-prem, high-throughput but less elastic.

#### Answer 2:
Choose S3/ADLS for managed scaling and cost benefits; use HDFS when you control hardware and need tight Hadoop ecosystem integration.

### 6.9 How do you implement data quality checks and monitoring in a production pipeline?

#### Answer 1:
Use tools like Great Expectations to define expectations and run checks in DAGs, alerting on anomalies.

#### Answer 2:
Integrate row-count, null-rate, and distribution checks with custom sensors and notify via Slack or email on failures.

### 6.10 Describe best practices for securing data in transit and at rest in a data warehouse.

#### Answer 1:
Use TLS for network encryption, server-side encryption for storage (SSE-S3, TDE in databases), and IAM roles for least-privilege access.

#### Answer 2:
Rotate keys regularly, enable encryption at rest, implement VPC peering or private endpoints, and audit with CloudTrail or equivalent.

## 7. Model Deployment with Python Web Frameworks

### 7.1 Compare Flask, FastAPI, and Django for serving ML models. What are trade-offs?

#### Answer 1:
Flask is minimal and flexible but requires more boilerplate; FastAPI offers async, automatic docs, and pydantic validation; Django provides batteries-included but heavier.

#### Answer 2:
Use Flask for simple microservices, FastAPI for performance and type safety, and Django when you need full ORM, admin, and built-in auth.

### 7.2 How do you perform input validation and serialization in FastAPI using Pydantic?

#### Answer 1:
Define request models by subclassing BaseModel; FastAPI auto-validates and converts JSON to typed Python objects.

#### Answer 2:
Use field types and validators in Pydantic models; return response models to control output schema and exclude sensitive fields.

### 7.3 Explain how to load and serve a pickled model versus an ONNX model.

#### Answer 1:
For pickle: model = pickle.load(open('model.pkl','rb')), then call model.predict. For ONNX: load with onnxruntime.InferenceSession and run with tensor inputs.

#### Answer 2:
Pickled models require Python runtime; ONNX models can be served in multiple languages and optimized with execution providers.

### 7.4 Describe the process of containerizing a Python inference service with Docker.

#### Answer 1:
Write a Dockerfile with a base image, copy code, install dependencies, expose port, and CMD to start the app.

#### Answer 2:
Use multi-stage builds to minimize image size, pin dependency versions, and include health check instructions.

### 7.5 How would you set up CI/CD for deploying model updates to production?

#### Answer 1:
Use GitHub Actions or Jenkins to run tests, build Docker images, push to a registry, and deploy to Kubernetes or cloud services.

#### Answer 2:
Implement pipelines that validate model performance, bump version, trigger blue-green deployment, and run smoke tests post-deployment.

### 7.6 What strategies can you use to scale an inference service under high load?

#### Answer 1:
Horizontal scale via multiple replicas behind a load balancer, autoscaling based on CPU/GPU metrics.

#### Answer 2:
Batch requests, use asynchronous processing, or serve with gRPC and optimize model (quantization, pruning) for faster inference.

### 7.7 How do you implement logging, metrics, and health checks for a model API?

#### Answer 1:
Integrate Python’s logging module for request logs, expose Prometheus metrics via /metrics, and add /health endpoint returning status.

#### Answer 2:
Use structured logging (JSON), OpenTelemetry for distributed tracing, and readiness/liveness probes in Kubernetes manifests.

### 7.8 Explain CORS and why it matters when serving APIs to web clients.

#### Answer 1:
CORS is a browser security feature that restricts cross-origin requests; you configure allowed origins and methods on the server.

#### Answer 2:
Without proper CORS headers (e.g., Access-Control-Allow-Origin), browsers block frontend apps from calling your API on a different domain.

### 7.9 How do you secure an API endpoint that serves sensitive predictions?

#### Answer 1:
Implement authentication (OAuth2/JWT), enforce HTTPS, validate tokens in middleware, and check scopes for authorization.

#### Answer 2:
Use API gateways with rate limiting, IP whitelisting, and encryption of request/response bodies at rest and in transit.

### 7.10 Describe canary or blue-green deployment strategies for model rollouts.

#### Answer 1:
Blue-green: deploy new version alongside old, switch traffic when ready; Canary: route a small percentage to new version and monitor before full roll-out.

#### Answer 2:
Use service mesh or load balancer rules to shift traffic gradually, run A/B tests on prediction quality, and roll back on errors.

## 8. Kaggle Competitions & Real-world Projects

### 8.1 Walk through your standard process for exploring a new Kaggle dataset.

#### Answer 1:
Start with df.info()/df.describe(), visualize distributions and missingness, then examine relationships via pairplots or correlation matrices.

#### Answer 2:
Generate summary statistics, plot key features, check target imbalance, and document insights in a notebook for reproducibility.

### 8.2 How do you decide which baseline model to build first?

#### Answer 1:
Choose a simple, fast model like logistic regression or decision tree to set a performance floor quickly.

#### Answer 2:
Use domain-informed heuristics (e.g., mean target predictor) or a lightweight ensemble to gauge initial viability.

### 8.3 Give examples of three feature engineering techniques you’ve used to boost performance.

#### Answer 1:
Created interaction terms, encoded cyclical features (sin/cos for dates), and aggregated statistics with groupby.

#### Answer 2:
Engineered sentiment scores from text, applied target encoding for high-cardinality categories, and generated lag features for time series.

### 8.4 How do you handle missing or corrupted data in competition datasets?

#### Answer 1:
Impute with median/mean or model-based methods, drop rows if few, and flag missingness with indicator variables.

#### Answer 2:
Use KNN imputation, drop columns with excessive nulls, and validate imputations against known distributions.

### 8.5 Describe your approach to avoiding data leakage when splitting data.

#### Answer 1:
Use time-based splits for temporal data, group-aware CV (GroupKFold) if related samples exist, and exclude target-related features.

#### Answer 2:
Ensure features derived from future information aren’t included, perform preprocessing inside CV folds only, and keep test data isolated.

### 8.6 What are blending and stacking ensembles, and how do you implement them?

#### Answer 1:
Blending holds out a validation set to train a meta-learner on base predictions; stacking uses CV out-of-fold predictions for the meta model.

#### Answer 2:
Implement with sklearn.ensemble.StackingClassifier or manually generate out-of-fold predictions, then fit a second-level model.

### 8.7 How do you prevent overfitting when you have a small training set?

#### Answer 1:
Use simpler models, strong regularization, cross-validation, and data augmentation if applicable.

#### Answer 2:
Employ ensembling of diverse models, early stopping, and feature selection to reduce noise.

### 8.8 Explain how you manage experiment tracking and reproducibility.

#### Answer 1:
Use MLflow or Weights & Biases to log parameters, metrics, and artifacts; version code and data with git and DVC.

#### Answer 2:
Maintain consistent conda environments, seed RNGs, and store notebooks with clear step-by-step pipelines.

### 8.9 How do you use the Kaggle API to automate data downloads and submissions?

#### Answer 1:
Authenticate with kaggle.json, run kaggle competitions download -c <comp> to fetch data and kaggle competitions submit -c <comp> -f file.csv -m "msg".

#### Answer 2:
Integrate Kaggle CLI commands into scripts or CI pipelines, parsing JSON responses for status and logs.

### 8.10 Discuss a time when you turned a competition solution into a production-ready pipeline.

#### Answer 1:
I refactored exploratory notebooks into modular code, containerized with Docker, and deployed on AWS Lambda for real-time scoring.

#### Answer 2:
I wrapped model training and inference in Airflow tasks, added monitoring, CI tests, and automated data validation before deployments.

# Interview Questions, Part II

## 1. Python Fundamentals

### 1.1 What is the difference between `append` and `extend` on Python lists?
**A:** `append` adds its argument as a single element; `extend` iterates over its argument adding each element.

### 1.2 How does Python’s garbage collector manage memory?
**A:** It uses reference counting plus a generational cycle-detecting GC to reclaim cyclically referenced objects.

### 1.3 When would you use a list versus a tuple?
**A:** Use a list for mutable sequences; use a tuple for immutable, hashable collections or fixed data.

### 1.4 How do you handle exceptions with multiple `except` blocks?
**A:** Place more specific exception types first, then broader exceptions or group multiple in a tuple.

### 1.5 Explain slicing syntax `my_list[start:stop:step]`.
**A:** It returns elements from `start` up to but excluding `stop`, advancing by `step` each time.

### 1.6 How do you flatten a nested list in Python?
**A:** With a list comprehension: `[y for x in nested for y in x]` (only one level deep).

### 1.7 Write a function to flatten a nested list of arbitrary depth.
```python
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# flatten([1, [2, [3, 4]], 5]) -> [1, 2, 3, 4, 5]
```

### 1.8 Implement a decorator that logs function execution time.
```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.4f}s")
        return result
    return wrapper

@timer
def slow():
    time.sleep(1)

# slow() prints: slow took 1.0001s
```

## 2. Pandas

### 2.1 How do you convert a column to datetime in Pandas?
**A:** Use `pd.to_datetime(df["col"], format="…")` to parse strings into `datetime64[ns]`.

### 2.2 What does `df.merge()` do versus `df.join()`?
**A:** `merge()` performs SQL-style joins on columns; `join()` aligns DataFrames on their index by default.

### 2.3 How can you rename multiple columns at once in a DataFrame?
**A:** Use `df.rename(columns={"old1":"new1","old2":"new2"}, inplace=True)`.

### 2.4 Explain the difference between `df.apply()` and `df.transform()`.
**A:** `apply()` can reduce or transform groups arbitrarily; `transform()` returns an output with the same index shape.

### 2.5 How do you pivot a DataFrame from long to wide format?
**A:** Use `df.pivot(index="row", columns="col", values="val")`.

### 2.6 Describe chaining operations in Pandas and when it’s useful.
**A:** You can `df.dropna().assign(...).groupby(...)` to build concise, readable pipelines without intermediate variables.

### 2.7 How do you sample a fraction of rows randomly in a DataFrame?
**A:** Use `df.sample(frac=0.1, random_state=42)`.

### 2.8 How do you inspect memory usage of a DataFrame?
**A:** Call `df.info(memory_usage="deep")` or `df.memory_usage(deep=True)`.

### 2.9 What is the purpose of `df.query()`?
**A:** To filter rows using a string expression, e.g. `df.query("col > 0")`.

### 2.10 How can you merge rows with identical keys?
**A:** Use `df.groupby("key").agg({"col1":"sum","col2":"mean"})`.

### 2.11 Create a DataFrame of 100 random integers and compute the rolling mean with window size 5.
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'vals': np.random.randint(0,100,size=100)})
df['rolling_mean'] = df['vals'].rolling(window=5).mean()
```

### 2.12 Using Pandas, read a large CSV in chunks of 10 000 rows and count total rows without loading all into memory.
```python
import pandas as pd

row_count = 0
for chunk in pd.read_csv('large.csv', chunksize=10000):
    row_count += len(chunk)
print(row_count)
```

---
## 2. Machine Learning with scikit-learn

### 2.1 Explain the bias–variance tradeoff.
**A:** Bias is underfitting error; variance is overfitting error. You tune model complexity to minimize total error.

### 2.2 What is stratified k-fold cross-validation?
**A:** It splits data so each fold preserves the target class distribution.

### 2.3 When to use `StandardScaler` vs `MinMaxScaler`?
**A:** `StandardScaler` centers to zero mean and unit variance; `MinMaxScaler` scales to a fixed range [0,1].

### 2.4 How does `LabelEncoder` differ from `OneHotEncoder`?
**A:** `LabelEncoder` maps categories to integer codes; `OneHotEncoder` creates binary indicator columns.

### 2.5 Describe how `Pipeline` prevents data leakage.
**A:** It ensures transforms (e.g. scaling, imputation) are learned only on training folds during CV.

### 2.6 How do you handle categorical features with high cardinality?
**A:** Use target encoding, embeddings, or frequency encoding instead of one-hot.

### 2.7 What metrics are suitable for imbalanced classification?
**A:** F1-score, ROC AUC, precision–recall AUC.

### 2.8 How can you parallelize model training?
**A:** Set `n_jobs=-1` in estimators like `RandomForestClassifier`.

### 2.9 Explain the difference between bagging and boosting.
**A:** Bagging builds base learners in parallel (reducing variance); boosting builds sequentially (reducing bias).

### 2.10 How to detect multicollinearity?
**A:** Compute Variance Inflation Factor (VIF) or inspect a correlation matrix.

### 2.11 How do you calibrate predicted probabilities?
**A:** Use `CalibratedClassifierCV` with `"sigmoid"` or `"isotonic"` methods.

### 2.12 Describe how `SelectFromModel` works.
**A:** It selects features whose importance (from a fitted estimator) exceeds a threshold.

### 2.13 How to handle missing values in scikit-learn?
**A:** Use `SimpleImputer` or `IterativeImputer` within a `Pipeline`.

### 2.14 Explain partial dependence plots.
**A:** They show the marginal effect of one or two features on the predicted outcome.

### 2.15 How to implement early stopping in GradientBoosting?
**A:** Set `n_iter_no_change` and `validation_fraction` parameters.

### 2.16 What is the `feature_importances_` attribute?
**A:** It gives relative importance scores for features in tree-based models.

### 2.17 Split data, train a RandomForest, and output feature importances.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data.csv')
X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head())
```

### 2.18 Build a `GridSearchCV` to tune an SVM’s `C` and `kernel`.
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10], 'kernel':['linear','rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

### 2.19 Create an imbalanced dataset sampler using SMOTE.
```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
```

### 2.20 Use `Pipeline` to chain imputation, scaling, and logistic regression.
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('impute', SimpleImputer()),
    ('scale', StandardScaler()),
    ('clf', LogisticRegression())
])
pipe.fit(X_train, y_train)
```

---

## 9. Natural Language Processing (NLP)

### 9.1 Tokenize a sentence using NLTK’s `word_tokenize`
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Natural Language Processing is fascinating!"
tokens = word_tokenize(text)
print(tokens)
# ['Natural', 'Language', 'Processing', 'is', 'fascinating', '!']
```

### 9.2 Remove English stopwords from a list of tokens
```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

tokens = ['this', 'is', 'a', 'sample', 'text']
stops = set(stopwords.words('english'))
filtered = [t for t in tokens if t.lower() not in stops]
print(filtered)
# ['sample', 'text']
```

### 9.3 Lemmatize words using spaCy
```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("The striped bats are hanging on their feet.")
lemmas = [token.lemma_ for token in doc]
print(lemmas)
# ['the', 'striped', 'bat', 'be', 'hang', 'on', 'their', 'foot', '.']
```

### 9.4 Compute a TF–IDF matrix for a corpus with scikit-learn
```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love machine learning.",
    "Natural language processing with Python.",
    "Deep learning models are powerful."
]
vec = TfidfVectorizer(stop_words='english')
X = vec.fit_transform(corpus)
print(X.toarray())
print(vec.get_feature_names_out())
```

### 9.5 Train a simple logistic regression sentiment classifier (binary labels)
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression())
])
texts = ["I love it", "I hate it"]
labels = [1, 0]
pipeline.fit(texts, labels)
print(pipeline.predict(["I love this tool", "I dislike bugs"]))
# [1 0]
```

### 9.6 Generate word embeddings with Gensim’s Word2Vec
```python
from gensim.models import Word2Vec

sentences = [
    ["natural", "language", "processing"],
    ["machine", "learning", "models"],
    ["word2vec", "embeddings"]
]
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, workers=1)
print(model.wv['language'])      # 50-dimensional vector
print(model.wv.most_similar('language'))
```

### 9.7 Use a pretrained Transformer to embed sentences with HuggingFace
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

sentences = ["Hello world", "NLP is fun"]
inputs = tokenizer(sentences, padding=True, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)
print(embeddings.shape)  # (2, 384)
```

### 9.8 Build a simple RNN text classifier in PyTorch
```python
import torch.nn as nn
import torch

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# Example instantiation:
model = RNNClassifier(vocab_size=5000, embed_dim=100, hidden_dim=64, n_classes=2)
```

### 9.9 Compute BLEU score for a candidate vs. reference
```python
from nltk.translate.bleu_score import sentence_bleu

reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'a', 'trial']
score = sentence_bleu(reference, candidate)
print(f"BLEU score: {score:.2f}")
```

### 9.10 Perform topic modeling with Gensim’s LDA
```python
from gensim import corpora, models

documents = [
    "cats and dogs",
    "dogs and wolves",
    "cats and lions",
    "lions and tigers"
]
texts = [doc.split() for doc in documents]
dic = corpora.Dictionary(texts)
corpus = [dic.doc2bow(text) for text in texts]
lda = models.LdaModel(corpus, num_topics=2, id2word=dic, passes=10)
print(lda.print_topics())
```

### 9.11 Extract named entities with spaCy
```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
ents = [(ent.text, ent.label_) for ent in doc.ents]
print(ents)
# [('Apple', 'ORG'), ('U.K.', 'GPE'), ('$1 billion', 'MONEY')]
```

### 9.12 Compute cosine similarity between two sentences
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = ["machine learning", "deep learning"]
vec = TfidfVectorizer().fit_transform(texts)
sim = cosine_similarity(vec[0], vec[1])
print(f"Cosine similarity: {sim[0][0]:.2f}")
```

### 9.13 Perform dependency parsing with spaCy
```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("The quick brown fox jumps over the lazy dog")
for token in doc:
    print(token.text, token.dep_, token.head.text)
```

### 9.14 Implement text cleaning and normalization
```python
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print(clean_text("Hello, WORLD!! 123"))
# "hello world 123"
```

### 9.15 Train a TF–IDF + SVM pipeline for multi-class text classification
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', LinearSVC())
])
texts = ["sports news", "politics today", "tech review"]
labels = [0, 1, 2]
pipeline.fit(texts, labels)
print(pipeline.predict(["latest sports update", "new smartphone"]))
```

### 9.16 Generate text with a pretrained GPT-2 using HuggingFace
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
input_ids = tokenizer.encode("In a distant future,", return_tensors='pt')
outputs = model.generate(input_ids, max_length=30, num_return_sequences=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 9.17 Calculate word frequencies from a corpus
```python
from collections import Counter

corpus = ["this is a test", "this test is fun"]
words = " ".join(corpus).split()
freq = Counter(words)
print(freq.most_common())
# [('this', 2), ('is', 2), ('a', 1), ('test', 2), ('fun', 1)]
```

### 9.18 Cluster documents with TF–IDF + KMeans
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

docs = [
    "cats and dogs",
    "lions and tigers",
    "machine learning models",
    "deep learning techniques"
]
vec = TfidfVectorizer(stop_words='english')
X = vec.fit_transform(docs)
km = KMeans(n_clusters=2, random_state=42).fit(X)
print(km.labels_)
```

### 9.19 Evaluate text classifier with precision, recall, F1
```python
from sklearn.metrics import classification_report

y_true = [0,1,0,1]
y_pred = [0,1,1,1]
print(classification_report(y_true, y_pred, target_names=['neg','pos']))
```

### 9.20 Translate English to German with a MarianMT model
```python
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-de'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Machine learning is transforming the world."
inputs = tokenizer(text, return_tensors="pt")
translated = model.generate(**inputs)
print(tokenizer.decode(translated[0], skip_special_tokens=True))
```
# Interview Qiestions Part, III

##  Python Fundamentals & Pandas

### 1. Explain the difference between `__getitem__` and `__getattr__` in Python classes.  
**Answer:**  
- `__getitem__(self, key)` is invoked when you use `obj[key]`.  
- `__getattr__(self, name)` is called only if the attribute `name` is not found the usual way (i.e. `obj.name`).  

### 2. When does Pandas issue a “chained assignment” warning and how can you avoid it?  
**Answer:**  
- It happens when you write `df[df.col > 0].col2 = 5`, which returns a view then assigns.  
- Avoid it by using `.loc`, e.g. `df.loc[df.col > 0, 'col2'] = 5`, or call `.copy()` on the slice.

### 3. How do you convert a column to categorical and why would you?  
**Answer:**  
- Use `df['col'] = df['col'].astype('category')`.  
- It reduces memory and speeds up groupby/merge when the column has few distinct values.

### 4. What’s the difference between a view and a copy in Pandas slicing?  
**Answer:**  
- A view shares the same underlying data buffer; modifying it may change the original.  
- A copy is a separate object; you can force it via `df.iloc[...] .copy()`.

### 5. How can you use `itertools.groupby` vs. `df.groupby`?  
**Answer:**  
- `itertools.groupby` works on sorted Python iterables and groups consecutive items.  
- `df.groupby` works on column values regardless of order and supports aggregation, transformation, and filtering.

## Machine Learning with scikit-learn

### 1. How do you implement a custom scorer for `GridSearchCV`?  
**Answer:**  
```python
from sklearn.metrics import make_scorer, f1_score
scorer = make_scorer(f1_score, average='macro')
grid = GridSearchCV(estimator, param_grid, scoring=scorer)
2. How can you preprocess numeric and categorical features differently in one pipeline?
```

**Answer:**

Use ColumnTransformer:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(), cat_cols)
])
pipe = Pipeline([('pre', pre), ('clf', LogisticRegression())])
3. What is a learning curve and how do you interpret it?
```

**Answer:**

```python
A plot of training vs. validation score as a function of training set size.

Converging high scores → low bias.

Large gap → high variance (overfitting).

4. How do you calibrate classifier probabilities?
```

**Answer:**

Use CalibratedClassifierCV:

```python
from sklearn.calibration import CalibratedClassifierCV
clf = CalibratedClassifierCV(base_estimator, method='isotonic', cv=5)
clf.fit(X_train, y_train)
5. How do you train a multi-output regression model?
```

**Answer:**

Wrap any regressor in MultiOutputRegressor:

```python
from sklearn.multioutput import MultiOutputRegressor
model = MultiOutputRegressor(RandomForestRegressor()).fit(X, y_multi)
Deep Learning with TensorFlow
1. How do you apply label smoothing in Keras?
```

**Answer:**

Pass label_smoothing to the loss:

```python
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
  metrics=['accuracy'])
```

### 2. How can you build a custom layer using Lambda?

**Answer:**

```python
from tensorflow.keras.layers import Lambda
square = Lambda(lambda x: x**2)
output = square(input_tensor)
```

### 3. How do you control GPU memory growth?

**Answer:**

```python
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
```

### 4. How would you accumulate gradients manually across multiple batches?

**Answer:**

```python
optimizer = tf.keras.optimizers.Adam()
accum_steps = 4
@tf.function
def train_step(data):
    for i, (x, y) in enumerate(data):
        with tf.GradientTape() as tape:
            loss = compute_loss(x, y)
        grads = tape.gradient(loss, model.trainable_variables)
        if i % accum_steps == 0:
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
```


### 5. How do you load a pretrained model from TensorFlow Hub?

**Answer:**

```python
import tensorflow_hub as hub
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2", input_shape=[], dtype=tf.string),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## Deep Learning with PyTorch

### 1. How do you implement a custom loss function?

**Answer:**

```python
import torch.nn as nn
import torch
class MyLoss(nn.Module):
    def forward(self, output, target):
        return torch.mean((output - target)**2 + 0.1*output.abs())
loss_fn = MyLoss()
2. Explain model.train() vs. model.eval().
```

**Answer:**

```python

model.train() enables dropout and batchnorm updates.

model.eval() disables dropout and uses running stats in batchnorm.
```

### 3. How do you use torch.utils.data.DataLoader with a custom collate_fn?

**Answer:**

```python
def collate(batch):
    # custom padding or stacking logic
    return torch.utils.data.dataloader.default_collate(batch)

loader = DataLoader(dataset, batch_size=32, collate_fn=collate)
```

### 4. How do you apply gradient checkpointing to save memory?
```

**Answer:**

Wrap blocks in torch.utils.checkpoint.checkpoint():

```python
from torch.utils.checkpoint import checkpoint
out = checkpoint(self.block, x)
```

### 5. How do you set up Distributed Data Parallel (DDP) training?

**Answer:**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group('nccl')
model = DDP(model.to(device))
```

## SQL & Database Performance Tuning

### 1. How do you use EXPLAIN (ANALYZE, BUFFERS) in PostgreSQL?

**Answer:**

```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM table WHERE col > 100;

# Shows actual row counts, timing, and buffer usage.
```

### 2. What are BRIN indexes and when are they useful?

**Answer:**

```python
BRIN (Block Range INdex) stores summary info per block range.

Good for very large, naturally ordered data like time-series.
```

### 3. How can you speed up bulk inserts?

**Answer:**

Use COPY instead of individual INSERT statements.

Disable indexes/triggers before load and re-enable after.

### 4. How do you update statistics on partitioned tables?

**Answer:**

Run ANALYZE on each child or use:

```sql
ANALYZE parent_table;
```

which cascades to partitions.

### 5. How do you detect and fix table bloat?

**Answer:**

Query pg_stat_user_tables for n_dead_tup.

Run VACUUM (FULL) or use pg_repack to reclaim space.

## Data Engineering: Pipelines & Data Lakes/Warehouses

### 1. How do you backfill a missed run in Airflow?

**Answer:**

Set catchup=True in your DAG and ensure start_date covers the gap; Airflow will schedule past runs.

### 2. What are AWS Glue bookmarks?

**Answer:**

They track data processed in previous runs to enable incremental ETL.

### 3. How can you enforce DataFrame schema with Pandera?

**Answer:**

```python
import pandera as pa
schema = pa.DataFrameSchema({
    "col": pa.Column(pa.Int, pa.Check(lambda s: s >= 0))
})
validated_df = schema.validate(df)
```

### 4. How do you write unit tests for an ETL function?

**Answer:**

Use pytest to feed known input, call the transformation function, and assert the output matches expected.

### 5. How do you integrate Great Expectations into a pipeline?

**Answer:**

Create expectation suites, then call great_expectations.checkpoint in your DAG to validate data against them.

## Model Deployment with Python Web Frameworks

### 1. How do you implement a health-check endpoint in Flask?

**Answer:**

```python
@app.route("/health")
def health():
    return {"status":"ok"}, 200
```

### 2. How can you stream application logs to the Elastic Stack?

**Answer:**

Configure the logging module with a LogstashFormatter and send to Logstash via TCP/UDP handler.

### 3. How do you provide real-time progress updates via WebSockets in FastAPI?

**Answer:**

Use fastapi.WebSocket endpoint:

```python
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text("progress: 10%")
```

### 4. How do you protect your Flask API behind Kong with JWT?

**Answer:**

Configure Kong’s JWT plugin on your service/route.

Clients must include Authorization: Bearer <token>.

### 5. How do you deploy a FastAPI app to Azure Functions?

**Answer:**

Wrap your ASGI app with Mangum adapter and deploy using the Azure Functions Python worker:

```python
from mangum import Mangum
handler = Mangum(app)
```

## Kaggle Competitions & Real-world Projects

### 1. How do you log experiment parameters and metrics with MLflow?

**Answer:**

```python
import mlflow
mlflow.start_run()
mlflow.log_param("lr", 0.01)
mlflow.log_metric("accuracy", 0.95)
mlflow.end_run()
```

### 2. How do you perform time-series cross-validation with scikit-learn?

**Answer:**

Use TimeSeriesSplit:

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X): ...
```

### 3. How do you version notebooks and data together in Git and DVC?

**Answer:**

Track code and small files in Git; track large datasets with dvc add data.csv and commit the .dvc file.

### 4. How do you automate Kaggle submissions in a CI pipeline?

**Answer:**

Invoke kaggle competitions submit -c <comp> -f submission.csv -m "msg" as a build step after model training.

### 5. How do you serve model artifacts using an artifact repository (e.g., Nexus, Artifactory)?

**Answer:**

Upload model files via CI, then download them at runtime in your deployment script using the repository’s REST API.
