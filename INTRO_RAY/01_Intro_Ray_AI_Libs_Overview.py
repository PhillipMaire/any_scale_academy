
# %% [markdown]
# # Introduction to the Ray AI Libraries: An example of using Ray data, Ray Train, Ray Tune, Ray Serve to implement a XGBoost regression model
#
# © 2025, Anyscale. All Rights Reserved

# %% [markdown]
# 💻 **Launch Locally**: You can run this notebook locally, but performance will be reduced.
#
# 🚀 **Launch on Cloud**: A Ray Cluster with 4 GPUs (Click [here](http://console.anyscale.com/register) to easily start a Ray cluster on Anyscale) is recommended to run this notebook.

# %% [markdown]
# Let's start with a quick end-to-end example to get a sense of what the Ray AI Libraries can do.
# <div class="alert alert-block alert-info">
# <b> Here is the roadmap for this notebook:</b>
# <ul>
#     <li>Overview of the Ray AI Libraries</li>
#     <li>Quick end-to-end example</li>
#     <ul>
#       <li>Vanilla XGBoost code</li>
#       <li>Hyperparameter tuning with Ray Tune</li>
#       <li>Distributed training with Ray Train</li>
#       <li>Serving an ensemble model with Ray Serve</li>
#       <li>Batch inference with Ray Data</li>
#     </ul>
# </ul>
# </div>

# %% [markdown]
# **Imports**

# %%
# (Optional): If you get an XGBoostError at import, you might have to `brew install libomp` before importing xgboost again
# !brew install libomp

# %%
import asyncio
import fastapi
import pandas as pd
import requests
# macos: If you get an XGBoostError at import, you might have to `brew install libomp` before importing xgboost again
import xgboost
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

import ray
import ray.tune
import ray.train
from ray.train.xgboost import XGBoostTrainer as RayTrainXGBoostTrainer
from ray.train import RunConfig
import ray.data
import ray.serve

# %% [markdown]
# ## 1. Overview of the Ray AI Libraries
#
# <img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Ray_AI_Libraries/Ray+AI+Libraries.png" width="700px" loading="lazy">
#
# Built on top of Ray Core, the Ray AI Libraries inherit all the performance and scalability benefits offered by Core while providing a convenient abstraction layer for machine learning. These Python-first native libraries allow ML practitioners to distribute individual workloads, end-to-end applications, and build custom use cases in a unified framework.
#
# The Ray AI Libraries bring together an ever-growing ecosystem of integrations with popular machine learning frameworks to create a common interface for development.
#
# |<img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Introduction_to_Ray_AIR/e2e_air.png" width="100%" loading="lazy">|
# |:-:|
# |Ray AI Libraries enable end-to-end ML development and provides multiple options for integrating with other tools and libraries from the MLOps ecosystem.|
#
#

# %% [markdown]
# ## 2. Quick end-to-end example
#
# For this classification task, you will apply a simple [XGBoost](https://xgboost.readthedocs.io/en/stable/) (a gradient boosted trees framework) model to the June 2021 [New York City Taxi & Limousine Commission's Trip Record Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). 
#
# The full dataset contains millions of samples of yellow cab rides, and the goal is to predict the tip amount.
#
# **Dataset features**
# * **`passenger_count`**
#     * Float (whole number) representing number of passengers.
# * **`trip_distance`** 
#     * Float representing trip distance in miles.
# * **`fare_amount`**
#     * Float representing total price including tax, tip, fees, etc.
# * **`tolls_amount`**
#     * Float representing the total paid on tolls if any.
#
# **Target**
# * **`trip_amount`**
#     * Float representing the total paid as tips

# %% [markdown]
# ### 2.1 Vanilla XGboost code
#
# Let's start with the vanilla XGBoost code to predict the tip amount for a NYC taxi cab data.

# %%
features = [
    "passenger_count", 
    "trip_distance",
    "fare_amount",
    "tolls_amount",
]

label_column = "tip_amount"


# %% [markdown]
# Define a function to load the data and split into train and test

# %%
# def load_data():
#     path = "s3://anyscale-public-materials/nyc-taxi-cab/yellow_tripdata_2021-03.parquet"
#     df = pd.read_parquet(path, columns=features + [label_column])
#     X_train, X_test, y_train, y_test = train_test_split(
#         df[features], df[label_column], test_size=0.2, random_state=42
#     )
#     dtrain = xgboost.DMatrix(X_train, label=y_train)
#     dtest = xgboost.DMatrix(X_test, label=y_test)
#     return dtrain, dtest
def load_data():
    path = "s3://anyscale-public-materials/nyc-taxi-cab/yellow_tripdata_2021-03.parquet"
    df = pd.read_parquet(
        path,
        columns=features + [label_column],
        storage_options={"anon": True}
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[label_column], test_size=0.2, random_state=42
    )
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest  = xgboost.DMatrix(X_test, label=y_test)
    return dtrain, dtest



# %% [markdown]
# Define a function to run `xgboost.train` given some hyperparameter dictionary `params`

# %%
storage_folder = "/Users/phil/Documents/GITHUB/any_scale_academy/INTRO_RAY/data/01_Intro_Ray_AI_Libs_Overview/" # Modify this path to your local folder if it runs on your local environment

# %%
from pathlib import Path
model_path = Path(storage_folder) / "model.ubj"

def my_xgboost_func(params):    
    evals_result = {}
    dtrain, dtest = load_data()
    bst = xgboost.train(
        params, 
        dtrain, 
        num_boost_round=10, 
        evals=[(dtest, "eval")], 
        evals_result=evals_result,
    )
    # Use Path
    bst.save_model(model_path)
    print(f"{evals_result['eval']}")
    return {"eval-rmse": evals_result["eval"]["rmse"][-1]}

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.1,
}
my_xgboost_func(params)

# %% [markdown]
# ### 2.2 Hyperparameter tuning with Ray Tune
#
# Let's use Ray Tune to run distributed hyperparameter tuning for the XGBoost model.

# %%
# tuner = ray.tune.Tuner(  # Create a tuner
#     my_xgboost_func,  # Pass it the training function which Ray Tune calls Trainable.
#     param_space={  # Pass it the parameter space to search over
#         "objective": "reg:squarederror",
#         "eval_metric": "rmse",
#         "tree_method": "hist",
#         "max_depth": 6,
#         "eta": ray.tune.uniform(0.01, 0.3),
#     },
#     run_config=RunConfig(storage_path=storage_folder),
#     tune_config=ray.tune.TuneConfig(  # Tell it which metric to tune
#         metric="eval-rmse",
#         mode="min",
#         num_samples=10,
#     ),
# )

tuner = ray.tune.Tuner(
    my_xgboost_func,
    param_space={
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": ray.tune.uniform(0.01, 0.3),
    },
    tune_config=ray.tune.TuneConfig(
        metric="eval-rmse",
        mode="min",
        num_samples=10,
    ),
    run_config=ray.train.RunConfig(
        storage_path=storage_folder,
    ),
)


results = tuner.fit()  # Run the tuning job
print(results.get_best_result().config)  # Get back the best hyperparameters



# %% [markdown]
# Here is a diagram that shows what Tune does:
#
# It is effectively scheduling many trials and returning the best performing one.
#
# <img src="https://bair.berkeley.edu/static/blog/tune/tune-arch-simple.png" width="700px" loading="lazy">

# %% [markdown]
# ### 2.3. Distributed training with Ray Train
#
# In case your training data is too large, your training might take a long time to complete.
#
# To speed it up, shard the dataset across training workers and perform distributed XGBoost training.
#
# Let's redefine `load_data` to now load a different slice of the data given the worker index/rank.

# %%
# def load_data():
#     # find out which training worker is running this code
#     train_ctx = ray.train.get_context()
#     worker_rank = train_ctx.get_world_rank()
#     print(f"Loading data for worker {worker_rank}...")

#     # build path based on training worker rank
#     month = (worker_rank + 1) % 12
#     year = 2021 + (worker_rank + 1) // 12
#     path = f"s3://anyscale-public-materials/nyc-taxi-cab/yellow_tripdata_{year}-{month:02}.parquet"

#     # same as before
#     df = pd.read_parquet(path, columns=features + [label_column])
#     X_train, X_test, y_train, y_test = train_test_split(
#         df[features], df[label_column], test_size=0.2, random_state=42
#     )
#     dtrain = xgboost.DMatrix(X_train, label=y_train)
#     dtest = xgboost.DMatrix(X_test, label=y_test)
#     return dtrain, dtest

def load_data():
    train_ctx = ray.train.get_context()
    worker_rank = train_ctx.get_world_rank()
    print(f"Loading data for worker {worker_rank}...")

    month = (worker_rank + 1) % 12
    year = 2021 + (worker_rank + 1) // 12
    path = f"s3://anyscale-public-materials/nyc-taxi-cab/yellow_tripdata_{year}-{month:02}.parquet"

    df = pd.read_parquet(
        path,
        columns=features + [label_column],
        storage_options={"anon": True}   # <-- required
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[label_column], test_size=0.2, random_state=42
    )
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)
    return dtrain, dtest



# %% [markdown]
# Now we can run distributed XGBoost training using Ray Train's XGBoostTrainer - similar trainers exist for other popular ML frameworks.

# %%
trainer = RayTrainXGBoostTrainer(  # Create a trainer
    my_xgboost_func,  # Pass it the training function
    scaling_config=ray.train.ScalingConfig(
        num_workers=2, use_gpu=False
    ),  # Define how many training workers
    train_loop_config=params,  # Pass it the hyperparameters
)

trainer.fit()  # Run the training job

# %% [markdown]
# Here is a diagram that shows what Train does:
#
# A train controller will create training workers and execute the training function on each worker.
#
# Ray Train delegates the distributed training to the underlying XGBoost framework.
#
# <img src="https://docs.ray.io/en/latest/_images/overview.png" width="700px" loading="lazy">

# %% [markdown]
# ### 2.4 Serving an ensemble model with Ray Serve
#
# Ray Serve allows for distributed serving of models and complex inference pipelines.
#
# Here is a diagram showing how to deploy an ensemble model with Ray Serve:
#
# <img src="https://images.ctfassets.net/xjan103pcp94/3DJ7vVRxYIvcFO7JmIUMCx/77a45caa275ffa46f5135f4d6726dd4f/Figure_2_-_Fanout_and_ensemble.png" width="700px" loading="lazy">
#
# Here is how the resulting code looks like:

# %%
app = fastapi.FastAPI()

class Payload(BaseModel):
    passenger_count: int
    trip_distance: float
    fare_amount: float
    tolls_amount: float


@ray.serve.deployment
@ray.serve.ingress(app)
class Ensemble:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    @app.post("/predict")
    async def predict(self, data: Payload) -> dict:
        model1_prediction, model2_prediction = await asyncio.gather(
            self.model1.predict.remote([data.model_dump()]),
            self.model2.predict.remote([data.model_dump()]),
        )
        out = {"prediction": float(model1_prediction + model2_prediction) / 2}
        return out


@ray.serve.deployment
class Model:
    def __init__(self, path: str):
        self._model = xgboost.Booster()
        self._model.load_model(path)

    def predict(self, data: list[dict]) -> list[float]:
        # Make prediction
        dmatrix = xgboost.DMatrix(pd.DataFrame(data))
        model_prediction = self._model.predict(dmatrix)
        return model_prediction


# Run the deployment
handle = ray.serve.run(
    Ensemble.bind(
        model1=Model.bind(model_path),
        model2=Model.bind(model_path),
    ),
    route_prefix="/ensemble"
)

# %% [markdown]
# Let's make an HTTP request to the Ray Serve instance.

# %%
requests.post(
    "http://localhost:8000/ensemble/predict",
    json={  # Use json parameter instead of params
        "passenger_count": 1,
        "trip_distance": 2.5,
        "fare_amount": 10.0,
        "tolls_amount": 0.5,
    },
).json()


# %% [markdown]
# ### 2.5 Batch inference with Ray Data
#
# Ray Data allows for distributed data processing through streaming execution across a heterogeneous cluster of CPUs and GPUs.
#
# This makes Ray Data ideal for workloads like compute-intensive data processing, data ingestion, and batch inference.

# %%
class OfflinePredictor:
    def __init__(self):
        # Load expensive state
        self._model = xgboost.Booster()
        self._model.load_model(model_path)

    def predict(self, data: list[dict]) -> list[float]:
        # Make prediction in batch
        dmatrix = xgboost.DMatrix(pd.DataFrame(data))
        model_prediction = self._model.predict(dmatrix)
        return model_prediction

    def __call__(self, batch: dict) -> dict:
        batch["predictions"] = self.predict(batch)
        return batch


# # Apply the predictor to the validation dataset
# prediction_pipeline = (
#     ray.data.read_parquet(
#         "s3://anyscale-public-materials/nyc-taxi-cab/yellow_tripdata_2021-03.parquet",
#     )
#     .select_columns(features)
#     .map_batches(OfflinePredictor, concurrency=(2, 10))
# )

import s3fs

fs = s3fs.S3FileSystem(anon=True)

prediction_pipeline = (
    ray.data.read_parquet(
        "s3://anyscale-public-materials/nyc-taxi-cab/yellow_tripdata_2021-03.parquet",
        filesystem=fs,       # <-- fsspec FS object
    )
    .select_columns(features)
    .map_batches(OfflinePredictor, concurrency=(2, 10))
)



# %% [markdown]
# After defining the pipeline, we can execute it in a distributed manner by writing the output to a sink

# %%
prediction_pipeline.write_parquet(f"{storage_folder}/xgboost_predictions")

# %% [markdown]
# Let's inspect the produced predictions.

# %%
# !ls {storage_folder}/xgboost_predictions/

# %% [markdown]
# ### 2.6 Clean up

# %%
# Run this cell for file cleanup 
# !rm -rf {storage_folder}/xgboost_predictions/
# !rm {model_path}

# %%
