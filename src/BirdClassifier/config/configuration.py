from BirdClassifier.logger import logger
from BirdClassifier.entity import DataIngestionConfig
from BirdClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from BirdClassifier.utils import read_yaml, create_directories


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

        data_ingestion_config = DataIngestionConfig(root_dir=config.root_dir,
                                                    source_url=config.source_url,
                                                    local_file_name=config.local_data_file,
                                                    unzip_dir=config.unzip_dir,
                                                    )
        return data_ingestion_config
