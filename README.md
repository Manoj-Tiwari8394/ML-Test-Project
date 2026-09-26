# Beginner Iris ML Project

A small, CPU-friendly classification project for learning the basic machine-learning workflow. It uses the Iris dataset bundled with scikit-learn and a logistic-regression model. The project includes local DVC and MLflow workflows, pytest coverage, a Dockerized prediction API, GitHub Actions CI, a Kubeflow Pipelines definition, and a small Kubernetes deployment.

## Dataset and model

Iris contains 150 flower observations, each with four numeric measurements: sepal length, sepal width, petal length, and petal width. The target is the flower species (`setosa`, `versicolor`, or `virginica`), so this is a three-class classification task.

The project uses logistic regression. It learns a weighted combination of the measurements to estimate a probability for each species. `C` controls regularization: smaller values apply stronger regularization. The default is `C=1.0`; the goal here is to understand the flow, not to tune for peak accuracy.

The model uses `StandardScaler` before logistic regression so measurements with different scales are comparable. Keeping both steps in one scikit-learn `Pipeline` ensures prediction data receives the same scaling learned from training data.

SciPy is pinned because scikit-learn uses it internally for optimization; controlling its version keeps the training run compatible and avoids solver-option warnings with newer SciPy releases.

## Environment

The project was set up for Python 3.12 in Ubuntu 24.04 on WSL2. All project packages belong in `.venv`; do not install them into system Python.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python --version
which python
```

The executable should point to this project’s `.venv/bin/python`. To leave the environment, run `deactivate`.

`requirements.txt` contains the pinned runtime packages; `requirements-dev.txt` includes those plus pytest. Pins make installations reproducible across project setup sessions.

| Package | Why it is used |
|---|---|
| pandas | Represent the dataset as a labeled DataFrame and select columns |
| NumPy | Numerical array operations used by the scientific Python stack |
| scikit-learn | Load Iris, split data, preprocess measurements, train and score the model |
| SciPy | Numerical optimization used internally by scikit-learn |
| joblib | Save and load the trained model pipeline |
| MLflow | Record training parameters, metrics, model artifacts, and CI registry versions |
| SQLAlchemy | SQLite-backed MLflow tracking and Model Registry |
| DVC | Local data and pipeline versioning |
| pytest | Verify dataset validation, preprocessing, training, and predictions |
| FastAPI | Not required; the prediction endpoint uses Python's standard-library HTTP server |
| Kubeflow Pipelines / `kfp` | Optional pipeline orchestration; the SDK is only needed to compile or submit the pipeline |

The Iris dataset comes from `sklearn.datasets.load_iris(as_frame=True)`, rather than `pd.read_csv(...)`: this small dataset ships with scikit-learn, so there is no external download or CSV file to manage in the first workflow. For CSV datasets, `pd.read_csv("path/to/file.csv")` reads the file and returns a pandas DataFrame.

Logistic regression learns model coefficients from labeled examples. In this project, `fit(X_train, y_train)` learns from flower measurements and known species. `predict(X_test)` applies the learned pipeline to held-out measurements. The test set estimates performance on examples that were not used for fitting. More complicated models are unnecessary for this introductory dataset and machine.

## Basic workflow

```bash
python -m src.train
python -m src.evaluate
python -m src.predict \
  --sepal-length 5.1 \
  --sepal-width 3.5 \
  --petal-length 1.4 \
  --petal-width 0.2
```

Training saves the fitted pipeline to `models/iris_model.joblib`. Evaluation loads it and measures performance on the same deterministic held-out test split. Prediction takes one flower’s measurements and prints the predicted species.

## How the Python files connect

```mermaid
flowchart TD
    A["scikit-learn bundled Iris data"] --> B["src/data_loader.py: DataFrame and validation"]
    B --> C["src/preprocess.py: X/y split and stratified train/test split"]
    C --> D["src/train.py: StandardScaler + LogisticRegression"]
    D --> E["models/iris_model.joblib"]
    D --> M["MLflow experiment: parameters, metrics, model artifact"]
    D --> F["src/evaluate.py: accuracy and macro F1"]
    E --> G["src/predict.py: one input row"]
    G --> H["Predicted species"]
    I["DVC: local data and pipeline versions"] -. tracks .-> A
    I -. tracks .-> D
    D --> J["Docker prediction API"]
    J --> K["Kubernetes Deployment + Service"]
    L["Git push / pull request"] --> N["GitHub Actions: tests + image build"]
    N -. validates .-> D
    O["Kubeflow Pipelines (optional platform)"] -. orchestrates .-> D
```

- `src/data_loader.py` calls `load_iris(as_frame=True)`, which supplies the feature table as a pandas DataFrame rather than reading a CSV. It adds readable species names and checks that the expected columns exist, values are present, and feature columns are numeric.
- `src/preprocess.py` selects the four input columns as `X` and the species column as `y`. It makes a reproducible, stratified train/test split so each species is represented in both portions. Stratification preserves each species’ share of the full dataset in both splits.
- `src/paths.py` defines the project root and default saved-model path once for the command-line modules.
- `src/train.py` builds the scaling-and-classification pipeline, fits it with `X_train` and `y_train`, saves it with joblib, then reports its held-out metrics. `C` is the inverse regularization strength: lower `C` constrains the coefficients more strongly.
- `src/evaluate.py` loads the saved pipeline and reports accuracy and macro F1. Since the split uses a fixed random seed, it evaluates the same test examples each time.
- `src/predict.py` turns four named measurements into a one-row DataFrame, loads the saved pipeline, and prints its species prediction.
- `src/serve.py` exposes the saved model through `GET /healthz`, `GET /readyz`, and `POST /predict`, using only Python's standard library for HTTP.
- `src/export_model.py` trains and saves the model without initializing MLflow; Docker uses it to bake a small model into the image.
- `src/pipeline_task.py` trains and evaluates a model and writes the model and metrics to Kubeflow-provided artifact paths.

In Python, a module is a `.py` file that can be imported or run with `python -m`. A function packages a named operation, such as loading, splitting, training, or evaluating. The data flow is: bundled dataset → pandas DataFrame → validation → features (`X`) and target (`y`) → train/test split → model fitting → evaluation or prediction.

## Local DVC pipeline

Git versions source code, configuration, and the small `data/raw/iris.csv.dvc` pointer; DVC stores the dataset contents in its local cache and runs the reproducible pipeline. No DVC cloud or network remote is configured. The project has its own DVC metadata under `.dvc/`; because this project is nested in a parent Git repository, initialize it with `dvc init --subdir`.

The bundled dataset is exported once to CSV for DVC tracking. Then DVC prepares a validated CSV, trains the model, and records evaluation metrics:

```bash
dvc add data/raw/iris.csv
dvc repro
dvc status
dvc dag
```

The raw CSV is 150 rows; the prepared CSV, model, and metrics are local DVC stage outputs. `params.yaml` stores the training configuration (`C`, test fraction, and random seed). For example, changing `train.c` and running `dvc repro` reruns the affected training and evaluation stages.

`pathspec` is pinned because DVC 3.59.1 imports an internal symbol removed by pathspec 1.x; pathspec 0.12.1 keeps the installed DVC version working. A remote can be configured later for shared or off-machine cache storage; the learning project does not require one.

## Local MLflow tracking

Each `python -m src.train` invocation creates an MLflow **run** in the `iris-logistic-regression` **experiment**. It logs the training parameters (including `C`), accuracy and macro F1 metrics, and the fitted scikit-learn pipeline as a model artifact with a signature and sample input. By default, interactive local runs use the file-backed `mlruns/` store, which is ignored by Git. The standalone joblib model remains at `models/iris_model.joblib` for evaluation and prediction.

Try three small runs with different regularization strengths:

```bash
python -m src.train --c 0.1
python -m src.train --c 1.0
python -m src.train --c 10.0
```

Start the file-backed local tracking UI in a separate terminal from the project root, with the virtual environment activated:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in a browser. Stop the UI with Ctrl+C. DVC tracks the data and pipeline; MLflow records the outcomes of training runs.

## Tests

Run the project tests from the repository root with the virtual environment active:

```bash
python -m pytest
```

The tests check that the dataset loader returns the expected rows and columns, reads a CSV correctly, and rejects invalid data. They verify that feature/target selection and the stratified split behave reproducibly, and that training, evaluation, and prediction return valid results. Tests help catch software regressions—such as a renamed column or an incorrect prediction shape—before they interfere with model training or serving.

## Docker image

The image uses `python:3.12-slim`, installs only the pinned packages needed for inference, and copies the source plus the already-trained model. It runs as a non-root user and starts the standard-library HTTP server on port 8000. DVC, MLflow, pytest, and the Kubeflow SDK are not installed in the inference image. `docker-compose.yml` is intentionally omitted.

Build and run it locally:

```bash
python -m src.export_model --output models/iris_model.joblib
docker build -t iris-ml:local .
docker run --rm -p 8000:8000 iris-ml:local
```

In another terminal, send a sample prediction:

```bash
curl http://127.0.0.1:8000/healthz
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}'
```

The API returns JSON containing the predicted species. The Docker port mapping connects host port 8000 to container port 8000.

## Automatic tracking, registry, and deployment

Every push and pull request runs the GitHub-hosted `validate` job: it tests the code, builds and smoke-tests the inference image, and compiles the KFP workflow. A successful push to `master` additionally starts `register-and-deploy` on a repository self-hosted runner labelled `iris-mlops`.

On each successful `master` push, that job:

1. Trains and evaluates the model.
2. Creates an MLflow run in the `iris-push-deploy` experiment. Tags record the Git commit SHA, commit author and message, changed file paths, branch, GitHub actor, event type, commit time, and CI start time. Parameters and held-out metrics are logged too.
3. Registers the trained model as a new version of `iris-classifier` in MLflow Model Registry.
4. Builds a Docker image containing that same trained model, labels it with the commit and model version, loads it into Kind, and updates the Deployment.
5. Waits for Kubernetes readiness and smoke-tests the deployed prediction endpoint.

The Registry uses a local SQLite database and artifact files under `$HOME/.local/share/iris-mlops/` on the WSL machine. These are persistent local state, not part of Git. Back up this directory if the experiment history and model versions matter. No credentials, tokens, or model artifacts are committed.

**Required one-time setup:** in the GitHub repository, open **Settings → Actions → Runners → New self-hosted runner**, select Linux x64, and follow GitHub’s displayed setup instructions on the WSL machine. Configure the runner with the custom label `iris-mlops`, install/run it as a service, and keep WSL, Docker, and the Kind cluster available. The runner’s Linux account must have access to the Docker daemon and the intended Kind `kubectl` context. The ephemeral runner registration token is shown by GitHub during setup; keep it private and never add it to this repository.

Self-hosted runners execute repository code with access to the local machine. The workflow deliberately limits deployment to pushes to `master`; pull requests run only on GitHub-hosted runners. Protect `master` and limit who can push to it. Do not allow untrusted pull-request code to run on the self-hosted runner.

Until that one-time runner setup is complete, the deployment job will remain queued waiting for a matching runner. The GitHub-hosted validation job continues to run independently. Do not push repeated commits while deployment is queued; the job needs this self-hosted runner to finish the pipeline.

### Inspect the push experiment and registered models

The CI run records GitHub actor (who pushed), commit author/message and changed paths (what and why), commit time, CI start time, branch, and commit SHA as MLflow tags. The source code also records hyperparameters, test metrics, and a model artifact. Every successful `master` push creates a new version under the `iris-classifier` registered model.

After the self-hosted runner is set up and has completed a run, start the registry-backed UI in a WSL terminal:

```bash
mlflow server \
  --backend-store-uri "sqlite:///$HOME/.local/share/iris-mlops/mlflow.db" \
  --default-artifact-root "file://$HOME/.local/share/iris-mlops/artifacts" \
  --host 127.0.0.1 \
  --port 5000
```

Open `http://127.0.0.1:5000`; choose the `iris-push-deploy` experiment to inspect runs and the Models page to inspect registry versions. This UI reads the same SQLite database and artifact folder used by CI. The earlier file-backed `mlruns/` runs remain separate.

The deployed pod template carries `git.commit` and `mlflow.model-version` annotations; inspect them with:

```bash
kubectl -n iris-ml get pods \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations.git\.commit}{"\t"}{.metadata.annotations.mlflow\.model-version}{"\n"}{end}'
```

To roll back the Kubernetes Deployment to its previous ReplicaSet (this does not delete the MLflow version):

```bash
kubectl -n iris-ml rollout undo deployment/iris-inference
kubectl -n iris-ml rollout status deployment/iris-inference
```

```text
Developer -> git push / pull request -> GitHub Actions
                                      -> GitHub-hosted validation and image smoke test
                                      -> (trusted master push) WSL self-hosted runner
                                           -> MLflow run + model registry version
                                           -> versioned image -> Kind rollout
```

## Kubeflow Pipelines (optional)

Kubeflow Pipelines (KFP) orchestrates ML tasks on Kubernetes; it is not a replacement for GitHub Actions CI. The source definition in `pipelines/iris_training_pipeline.py` describes a training task with configurable `C` and model/metrics artifacts. Compile it with the optional pinned SDK:

```bash
python -m pip install -r requirements-kfp.txt
python pipelines/iris_training_pipeline.py
```

This writes `pipelines/iris-training-pipeline.yaml`. To run it, you need a working Kubeflow Pipelines installation and a training image available to its worker pods. In this setup there is **no KFP installation**. Installing the full Kubeflow platform into the current single-node Kind cluster is not recommended for a machine with approximately 4 GB RAM. A compiled pipeline is ready for a suitable KFP endpoint later; it has not been submitted or run.

## Kubernetes model-serving demo

`k8s/iris-inference.yaml` contains a dedicated `iris-ml` namespace, one small inference Deployment, and an internal ClusterIP Service. The container has liveness/readiness probes, a non-root security context, and a 512 MiB memory limit. This deploys only the prediction API; it does not install Kubeflow, KServe, or an ingress controller.

With Docker and the existing Kind cluster running, build and load the image, then deploy:

```bash
python -m src.export_model --output models/iris_model.joblib
docker build -t iris-ml:local .
kind load docker-image iris-ml:local --name kubernetes-demo-cluster
kubectl apply -f k8s/iris-inference.yaml
kubectl -n iris-ml rollout status deployment/iris-inference
kubectl -n iris-ml port-forward service/iris-inference 8000:8000
```

While port-forwarding is active, use the same `curl` requests from the Docker example. Stop port-forwarding with Ctrl+C. To remove only this demo deployment:

```bash
kubectl delete -f k8s/iris-inference.yaml
```

Kubernetes runs the container and restarts it when needed; the Service gives the pod a stable in-cluster address. The model is bundled in the image, so no PVC, object storage, model registry, or cluster-wide service is needed. KServe is left optional for a later phase: the current single-node cluster has no Kubeflow installation, and the plain Deployment is easier to understand and lighter to run.

## Resource limits and optional platforms

The main learning project is designed for a CPU-only machine with approximately 4 GB RAM: the dataset has only 150 rows, training is a small scikit-learn operation, the prediction image is about 475 MB, and the local Kubernetes Deployment has a 512 MiB memory limit.

The active Kind cluster currently runs the Iris prediction Deployment, not Kubeflow. The Kubeflow Pipelines definition is compiled by CI but is not submitted to a pipeline server. A KFP server adds multiple platform services, and KServe adds serving controllers and related components; neither is installed because that extra footprint is not justified on this machine. The plain Kubernetes Deployment provides the model-serving learning objective with fewer moving parts. To run the KFP workflow later, use an existing KFP endpoint and make the built training image accessible to its worker pods.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `python: command not found` before activation | On this Ubuntu/WSL setup, create the environment with `python3 -m venv .venv`, then activate it with `source .venv/bin/activate`. |
| Imports fail or packages appear missing | Activate `.venv` and install the pinned dependencies with `python -m pip install -r requirements-dev.txt`. Check `which python` points inside `.venv`. |
| DVC reports a changed output | Run `dvc repro` to regenerate the dependent stages, then `dvc status`. The project uses only its local DVC cache; there is no cloud remote. |
| The prediction CLI says the model is missing | Run `python -m src.train` first. Before building an inference image, run `python -m src.export_model --output models/iris_model.joblib`. |
| Docker cannot access the daemon | Check `docker info`; start the Docker Engine in Ubuntu or the Docker Desktop WSL integration configured for this distro. Compose is not required. |
| Docker reports host port 8000 is already allocated | Use a different host port, e.g. `docker run --rm -p 18000:8000 iris-ml:local`, and send requests to port 18000. |
| Kubernetes reports `ImagePullBackOff` for `iris-ml:local` | Build the image and load it into this Kind cluster: `kind load docker-image iris-ml:local --name kubernetes-demo-cluster`. |
| Kubernetes port-forward cannot bind port 8000 | Choose an unused local port, e.g. `kubectl -n iris-ml port-forward service/iris-inference 18000:8000`. |
| Kubeflow pipeline compiles, but cannot be run | Compilation creates a package only. A KFP server and a worker-accessible training image are also required; neither is installed/configured in this lightweight setup. |
| The deployment job is queued | The WSL self-hosted runner must be registered with the `iris-mlops` label, running as a service, and able to access Docker and Kind. |
| MLflow Registry is unavailable in local training | Registry operations require a database-backed tracking store. Automatic CI configures SQLite; ordinary local runs use the file-backed `mlruns/` store unless `MLFLOW_TRACKING_URI` is set. SQLAlchemy is pinned to 2.0 because MLflow 2.22.2 is incompatible with SQLAlchemy 2.1. |

## Production comparison

This repository demonstrates lifecycle concepts locally; it is not a production deployment. A larger real system might add:

| Production capability | Why it may be added |
|---|---|
| Object storage and a remote DVC cache | Share versioned datasets and artifacts across developers and CI |
| Hosted MLflow tracking and a model registry | Centralize runs, model versions, approvals, and lineage |
| CI/CD and a container registry | Build, scan, version, and promote immutable images |
| Kubernetes with KServe | Operate scalable model endpoints with rollout and traffic management |
| Data validation and model monitoring | Detect broken inputs, drift, and changes in model behavior |
| Central logging and observability | Diagnose service errors and resource/performance problems |
| Secrets management and cloud infrastructure | Protect credentials and provision controlled environments |

These services require operational work, credentials, infrastructure, and more memory. They are deliberately excluded from the beginner project except for a small local Docker image and the existing lightweight Kind deployment.

## Learning map

| Technology | What you learned | Why it exists |
|---|---|---|
| Python | Organize a small ML application into modules and functions | Implements data handling, training, evaluation, and prediction |
| Virtualenv (`venv`) | Isolate project dependencies in `.venv` | Avoids changing system Python and keeps package installs project-specific |
| pandas | Work with labeled tabular data, columns, and DataFrames | Makes feature and target preparation readable |
| NumPy / SciPy | Use the numerical foundations of the Python ML stack | Support array operations and model optimization |
| scikit-learn | Split data, scale features, fit logistic regression, and evaluate predictions | Provides simple, consistent classical ML components |
| Git / GitHub | Version source, configuration, and collaboration changes | Provides the history and shared repository for the project |
| DVC | Track data pointers, pipeline stages, and reproducible outputs locally | Keeps data/pipeline lineage distinct from source-code history |
| MLflow | Compare experiment parameters, metrics, and model artifacts | Makes training runs inspectable and repeatable |
| pytest | Test data validation, preprocessing, training, and predictions | Catches software errors independently of model scores |
| Docker | Build and run the inference API as a portable image | Packages the runtime and application together |
| GitHub Actions | Run tests, build/smoke-test Docker, and compile KFP on pushes/PRs | Automates basic continuous integration |
| Kubernetes | Deploy a container as a Deployment and expose it with a Service | Orchestrates containers and supplies health checks/restarts |
| Kubeflow Pipelines | Define a parameterized training task and produce model/metric artifacts | Orchestrates multi-step ML workflows when a KFP platform is available |
| KServe | Not installed in this project | An optional Kubernetes-native model-serving layer for more advanced deployments |

Mental model:

```text
Python             -> ML code
Git                -> Code and configuration versioning
DVC                -> Data and pipeline versioning
MLflow             -> Experiment and model tracking
pytest             -> Software correctness
GitHub Actions     -> Automatic CI
Docker             -> Portable application image
Kubernetes         -> Container orchestration
Kubeflow Pipelines -> ML workflow orchestration (compiled here, server optional)
KServe             -> Model serving (optional, not installed here)
```

The order of learning matters: **understanding → reproducibility → simplicity → correctness**. Keep the optional platform infrastructure separate from the lightweight project until there is a real learning or deployment need for it.
