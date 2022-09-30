import shutil
import time
from pathlib import Path
from urllib.parse import urlparse
import os 

import mlflow
import mlflow.keras
import tensorflow as tf

from BirdClassifier.entity import EvaluationConfig
from BirdClassifier.utils import (get_best_model_s3, load_json,
                                  s3_download_model, save_json, upload_file)


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    def _test_generator(self):

        data_generator_kwargs = dict(rescale=1.0 / 255,)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear")

        test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                                **data_generator_kwargs)

        self.test_generator = test_data_generator.flow_from_directory(
            directory=self.config.test_data, **dataflow_kwargs)

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    @property
    def _get_trained_model_path(self):
        trained_model_name = f"model_{self.timestamp}.h5"
        trained_model_path = os.path.join(self.config.evaluation_model_dir, trained_model_name)
        shutil.copy(src=self.config.path_of_model, dst=trained_model_path)
        return trained_model_path

    def evaluation(self):
        self.train_model_path = self._get_trained_model_path
        self.best_model_path = get_best_model_s3(self.config.best_model_path)
        self.best_model = None
        if self.best_model_path is not None:
            self.best_model = self.load_model(self.best_model_path)
        self.model = self.load_model(self.train_model_path)
        self._test_generator()
        self.log_into_mlflow()

    def _evaluate_model(self, model):
        scores = model.evaluate(self.test_generator)
        response_ = {"loss": scores[0], "accuracy": scores[1]}
        return response_

    def update_best_model(self):
        s3_model_result = self.result.s3_best_model_score
        local_model_result = self.result.local_model_score
        if s3_model_result is None:
            upload_file(file_name=str(self.train_model_path), object_name="Best_model")
        elif local_model_result.accuracy > s3_model_result.accuracy:
            self.best_model = self.model
            upload_file(file_name=str(self.train_model_path) , object_name="Best_model")
            upload_file(file_name=str(self.best_model_path), object_name=f"{self.timestamp}_model")
        else:
            upload_file(file_name=str(self.train_model_path), object_name=f"{self.timestamp}_model")

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            self.scores_ = dict()
            self.scores_["local_model_score"] = self._evaluate_model(model=self.model)
            self.scores_["s3_best_model_score"] = None
            if self.best_model is not None:
                self.scores_["s3_best_model_score"] = self._evaluate_model(
                    model=self.best_model
                )
            score_file_name = f"{self.timestamp}_score.json"
            score_file_path = os.path.join(self.config.score_dir, score_file_name)
            save_json(path=Path(score_file_path), data=self.scores_)
            self.result = load_json(path=Path(score_file_path))
            self.update_best_model()
            local_model_result = self.result.local_model_score

            mlflow.log_metrics({"loss": local_model_result.loss,
                    "accuracy": local_model_result.accuracy,})
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(
                    self.model, "model", registered_model_name="VGG16Model"
                )
            else:
                mlflow.keras.log_model(self.model, "model")
            mlflow.end_run()
