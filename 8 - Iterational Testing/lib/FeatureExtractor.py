import pandas as pd
import numpy as np
import importlib.util
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractor:
    def __init__(self, strategy_file: str):
        self.strategy_file = strategy_file
        self.og_data_path = None
        logging.info("FeatureExtractor initialized with strategy file: %s", strategy_file)
        
    def extract_features(self, data_path: str, save_to: str):
        self.og_data_path = data_path
        logging.info("Starting feature extraction with data file: %s", data_path)
        
        # Load the data
        df = self._read_dataframe()
        logging.info("Data successfully loaded from %s", data_path)
        
        # Dynamically import the strategy function
        strategy = self._import_strategy_function()
        logging.info("Strategy function successfully imported from %s", self.strategy_file)
        
        # Apply the strategy function to the dataframe
        extracted_df = strategy(df)
        logging.info("Strategy function applied to the dataframe.")
        
        # Save the modified dataframe
        self._save_extracted_df(extracted_df, save_to)
        logging.info("Extracted features saved to %s", save_to)

    def _read_dataframe(self):
        if not self.og_data_path:
            raise ValueError("First set the data path!")
        df = pd.read_parquet(self.og_data_path)
        return df

    def _import_strategy_function(self):
        # Build the path to the strategy file
        module_path = f"./extraction_scripts/{self.strategy_file}"
        logging.info("Attempting to load strategy module from %s", module_path)
        
        # Load the module from the specified path
        spec = importlib.util.spec_from_file_location("strategy_module", module_path)
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        
        # Retrieve the strategy function from the loaded module
        if not hasattr(strategy_module, 'strategy'):
            logging.error("The specified strategy file does not contain a 'strategy' function.")
            raise AttributeError("The specified strategy file does not contain a 'strategy' function.")
        
        logging.info("Strategy function found in module.")
        return strategy_module.strategy

    def _save_extracted_df(self, extracted_df: pd.DataFrame, save_to: str):
        extracted_df.to_parquet(save_to)
