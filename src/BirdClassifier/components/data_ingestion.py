import os
from zipfile import ZipFile
import subprocess
from BirdClassifier.logger import logger
from BirdClassifier.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """Data Ingestion class to download the data . please make sure
        download kaggle.json file on your machine.
        reference : https://www.kaggle.com/general/156610

        Args:
            config (DataIngestionConfig):   root_dir : DirectoryPath
                                            source_url : str
                                            local_file_name : str
                                            unzip_dir : DirectoryPath
        """
        self.config = config
        logger.info(f'{"#"*10} STAGE ONE DATA INGESTION STARTED {"#"*10}')

    def download_file(self):
        if not os.path.exists(self.config.local_file_name):
            logger.info(f" downloading data from kaggle {self.config.source_url}")
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    self.config.source_url,
                    "-p",
                    self.config.root_dir,
                ]
            )

    def unzip_and_clean(self):
        with ZipFile(file=self.config.local_file_name, mode="r") as zf:
            list_of_files = zf.namelist()
            logger.info(f"Total files downloaded : {len(list_of_files)}")
            zf.extractall(path=self.config.root_dir)
