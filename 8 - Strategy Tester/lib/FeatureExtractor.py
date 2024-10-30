import pandas as pd
import importlib.util
from lib.Logger import Logger


class FeatureExtractor:
    def __init__(self, strategy_name: str, data_path: str):
        self.logger = Logger().get_logger(self.__class__.__name__)
        self.strategy_name = strategy_name
        self.data_path = data_path
        self.strategy_function = self._load_strategy_function()
        self.logger.info("Initialized with strategy: %s", strategy_name)

    def extract_features(self):
        self.logger.info("Starting feature extraction with data file: %s", self.data_path)
        df = pd.read_parquet(self.data_path)
        extracted_df = self.strategy_function(df)
        self.logger.info("Feature extraction completed for strategy: %s", self.strategy_name)
        return extracted_df

    def _load_strategy_function(self):
        module_path = f"./strategies/{self.strategy_name}.py"
        spec = importlib.util.spec_from_file_location("strategy_module", module_path)
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)

        if not hasattr(strategy_module, 'strategy'):
            self.logger.error("The strategy file does not contain a 'strategy' function.")
            raise AttributeError("Strategy function is missing in the specified file.")
        
        return strategy_module.strategy
