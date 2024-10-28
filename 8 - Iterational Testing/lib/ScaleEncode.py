import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScaleEncode:
    def __init__(self, data_path=None):
        self.og_data_path = data_path
        logging.info("ScaleEncode initialized with data path: %s", data_path)
        
    def scale_and_encode(self, save_to: str, target_col=None, scaler=None, specific_columns_to_encode=None):
        # Load the data
        df = self._read_dataframe()
        logging.info("Data successfully loaded from %s", self.og_data_path)
        
        # Encode categorical columns
        if specific_columns_to_encode:
            logging.info("Encoding specified categorical columns: %s", specific_columns_to_encode)
            df = self._encode_specific(df, specific_columns_to_encode)
        
        logging.info("Automatically encoding the rest of the categorical columns.")
        df = self._auto_encode(df)
        
        # Check for unsafe numeric columns
        unsafe_columns = self._find_unsafe_num_columns(df)
        if unsafe_columns:
            logging.warning("Found unsafe numeric columns with infinite values: %s", unsafe_columns)
        
        # Scale numeric columns if a scaler is provided
        if scaler:
            logging.info("Scaling numeric columns using %s", scaler.__class__.__name__)
            df = self._scale(df, target_col, scaler)
        
        # Save the modified dataframe
        self._save_extracted_df(df, save_to)
        logging.info("Scaled and encoded dataframe saved to %s", save_to)

    def _read_dataframe(self):
        if not self.og_data_path:
            raise ValueError("Data path must be specified!")
        df = pd.read_parquet(self.og_data_path)
        return df

    def _auto_encode(self, df: pd.DataFrame):
        columns_to_encode = df.select_dtypes(include=['category', 'object']).columns
        logging.info("Auto-encoding columns: %s", columns_to_encode)
        if len(columns_to_encode) > 0:
            df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
        return df

    def _encode_specific(self, df: pd.DataFrame, columns_to_encode):
        if isinstance(columns_to_encode, str):
            columns_to_encode = [columns_to_encode]
        logging.info("Encoding specific columns: %s", columns_to_encode)
        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
        return df

    def _find_unsafe_num_columns(self, df: pd.DataFrame):
        numeric_columns = df.select_dtypes(include=['number']).columns
        unsafe_cols = [col for col in numeric_columns if not np.isfinite(df[col]).all()]
        return unsafe_cols

    def _scale(self, df: pd.DataFrame, target_col: str, scaler):
        if target_col:
            target = df[target_col]
            numeric_columns = df.select_dtypes(include=['number']).columns.drop(target_col)
        else:
            numeric_columns = df.select_dtypes(include=['number']).columns

        # Apply scaling
        df_scaled = df.copy()
        df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])
        
        # Re-attach target column if it was separated
        if target_col:
            df_scaled[target_col] = target
        return df_scaled

    def _save_extracted_df(self, extracted_df: pd.DataFrame, save_to: str):
        extracted_df.to_parquet(save_to)