from pathlib import Path

from pydantic import BaseModel, DirectoryPath, FilePath


class DataIngestionConfig(BaseModel):
    root_dir: DirectoryPath
    source_url: str
    local_file_name: str
    unzip_dir: DirectoryPath


class PrepareBaseModelConfig(BaseModel):
    root_dir: DirectoryPath
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


class TrainingConfig(BaseModel):
    root_dir: DirectoryPath
    trained_model_path: Path
    updated_base_model_path: FilePath
    training_data: DirectoryPath
    validation_data: DirectoryPath
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list


class PrepareCallbacksConfig(BaseModel):
    root_dir: DirectoryPath
    tensorboard_root_log_dir: DirectoryPath
    checkpoint_model_filepath: Path
    early_stopping_patience: int
    early_stopping__monitor: str


class EvaluationConfig(BaseModel):
    path_of_model: FilePath
    evaluation_model_dir: DirectoryPath
    test_data: DirectoryPath
    best_model_path: Path
    all_params: dict
    mlflow_uri: str
    score_dir: DirectoryPath
    params_image_size: list
    params_batch_size: int
