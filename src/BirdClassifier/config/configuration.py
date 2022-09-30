import os
from pathlib import Path

from BirdClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from BirdClassifier.entity import (DataIngestionConfig, EvaluationConfig,
                                   PrepareBaseModelConfig,
                                   PrepareCallbacksConfig, TrainingConfig)
from BirdClassifier.logger import logger
from BirdClassifier.utils import create_directories, read_yaml


class ConfigurationManager:
    """Configuration Manger class which take
    Config_file_path : configuration Data
    params File path : Params data
    """

    def __init__(
        self, config_file_path=CONFIG_FILE_PATH, param_file_path=PARAMS_FILE_PATH
    ) -> None:
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(param_file_path)
        logger.info(f"Artifacts dir : {self.config.artifacts_root}")
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_file_name=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )
        return prepare_base_model_config

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_checkpoint_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([config.tensorboard_root_log_dir, model_checkpoint_dir])
        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
            early_stopping_patience=self.params.EARLY_STOPPING_PATIENCE,
            early_stopping__monitor=self.params.EARLY_STOPPING_MONITOR,
        )
        return prepare_callback_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=training.trained_model_path,
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=training.training_data_dir,
            validation_data=training.validation_data_dir,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
        )
        return training_config

    def get_validation_config(self) -> EvaluationConfig:

        eval = self.config.evaluation
        params = self.params
        create_directories([eval.evaluation_model_dir, eval.score_dir])
        eval_config = EvaluationConfig(
            path_of_model=eval.path_of_model,
            test_data=eval.test_data,
            score_dir=eval.score_dir,
            evaluation_model_dir=eval.evaluation_model_dir,
            best_model_path=eval.best_model_path,
            mlflow_uri=eval.mlflow_uri,
            all_params=params,
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
        return eval_config
