from pydantic import BaseModel, FilePath, DirectoryPath
from pathlib import Path


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
