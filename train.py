import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse


if __name__ == "__main__":
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print(f"Score: {score}")
    mlflow.log_metric("score", score)
    predictions = lr.predict(X)
    signature = infer_signature(X, predictions)
    #mlflow.sklearn.log_model(lr, "model", signature=signature)
    #print(f"Model saved in run {mlflow.active_run().info.run_uuid}")
    
    # modification for getting result in dahshun
    remote_server_uri="https://dagshub.com/nathabee123/mlflowexperiments.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print(mlflow.get_tracking_uri())
    print(tracking_url_type_store)

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(
            lr, "model", registered_model_name="LogisticRegression", signature=signature
        )
    else:
        mlflow.sklearn.log_model(lr, "model", signature=signature)