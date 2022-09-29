from pydantic import BaseModel , FilePath , FileUrl , DirectoryPath , AnyUrl
from pydantic.dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    root_dir : DirectoryPath
    source_url : str
    local_file_name : str
    unzip_dir : DirectoryPath