import pandas as pd
import numpy as np
from lib.Logger import Logger
from sklearn.preprocessing import StandardScaler


class ScaleEncode:
    def __init__(self, data_path=None, data=None, scaler=None):
        """
        Initialize the ScaleEncode class.

        Parameters:
        - data_path (str): Path to the input data file (optional if data is provided).
        - data (pd.DataFrame): DataFrame to be used directly (optional if data_path is provided).
        - scaler: Scaler instance from sklearn (default is StandardScaler if None).
        """
        self.logger = Logger().get_logger(self.__class__.__name__)
        self.data_path = data_path
        self.data = data
        self.scaler = scaler or StandardScaler()
        if data_path:
            self.logger.info("Initialized with data path: %s", data_path)
        elif isinstance(data, pd.DataFrame):
            self.logger.info("Initialized with provided DataFrame.")
        else:
            raise ValueError("Either data_path or data must be specified.")

    def scale_and_encode(self, save_to=None, target_col=None, specific_columns_to_encode=None):
        """
        Scale and encode the input data, and optionally save the modified DataFrame.

        Parameters:
        - save_to (str): Path to save the modified DataFrame.
        - target_col (str): Name of the target column to exclude from scaling.
        - specific_columns_to_encode (list): List of categorical columns to specifically encode (optional).
        
        Returns:
        - pd.DataFrame: Scaled and encoded DataFrame.
        """
        df = self._read_dataframe()
        self.logger.info("Data successfully loaded.")

        # Handle specified or automatic encoding
        if specific_columns_to_encode:
            self.logger.info("Encoding specified categorical columns: %s", specific_columns_to_encode)
            df = self._encode_specific(df, specific_columns_to_encode)
        else:
            self.logger.info("Automatically encoding categorical columns.")
            df = self._auto_encode(df)

        # Handle NaN and infinite values in numeric columns
        unsafe_columns = self._find_and_handle_unsafe_columns(df)
        if unsafe_columns:
            self.logger.warning("Handled unsafe numeric columns: %s", unsafe_columns)

        # Scale the data
        if self.scaler:
            self.logger.info("Scaling numeric columns using %s", self.scaler.__class__.__name__)
            df = self._scale(df, target_col)

        # Save if path provided
        if save_to:
            self._save_extracted_df(df, save_to)
            self.logger.info("Scaled and encoded dataframe saved to %s", save_to)
        
        return df

    def _read_dataframe(self):
        """
        Reads the DataFrame either from a file or directly if provided.

        Returns:
        - pd.DataFrame: Loaded DataFrame.
        """
        if self.data is not None:
            return self.data
        elif self.data_path:
            return pd.read_parquet(self.data_path)
        else:
            raise ValueError("No data source available.")

    def _auto_encode(self, df):
        """
        Automatically encodes all categorical columns with one-hot encoding, dropping columns with high cardinality.

        Parameters:
        - df (pd.DataFrame): DataFrame to encode.

        Returns:
        - pd.DataFrame: Encoded DataFrame.
        """
        columns_to_encode = df.select_dtypes(include=['category', 'object']).columns
        columns_to_drop = []

        for col in columns_to_encode:
            if df[col].nunique() > 5:
                self.logger.warning("Dropping high-cardinality column '%s'", col)
                columns_to_drop.append(col)

        df = df.drop(columns=columns_to_drop)
        df = pd.get_dummies(df, columns=[col for col in columns_to_encode if col not in columns_to_drop], drop_first=True)
        self.logger.info("Auto-encoding completed. Encoded columns: %s", list(columns_to_encode))
        return df

    def _encode_specific(self, df, columns_to_encode):
        """
        Specifically encodes provided categorical columns using one-hot encoding.

        Parameters:
        - df (pd.DataFrame): DataFrame to encode.
        - columns_to_encode (list): List of columns to encode.

        Returns:
        - pd.DataFrame: Encoded DataFrame.
        """
        if isinstance(columns_to_encode, str):
            columns_to_encode = [columns_to_encode]
        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
        self.logger.info("Specific columns encoded: %s", columns_to_encode)
        return df

    def _find_and_handle_unsafe_columns(self, df):
        """
        Identifies columns with NaN or infinite values, replacing them with median or mode values as appropriate.

        Parameters:
        - df (pd.DataFrame): DataFrame to process.

        Returns:
        - dict: A dictionary with column names as keys and issue types ('inf', 'nan') as values.
        """
        numeric_columns = df.select_dtypes(include=['number']).columns
        handled_columns = {}

        # Handle infinite values
        for col in numeric_columns:
            if np.isinf(df[col]).any():
                median_value = df[~np.isinf(df[col])][col].median()
                df[col] = df[col].replace([np.inf, -np.inf], median_value)
                handled_columns[col] = 'inf'
                self.logger.info("Replaced inf values in column '%s' with median: %f", col, median_value)

        # Handle NaN values
        for col in df.columns[df.isna().any()]:
            if col in numeric_columns:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                self.logger.info("Replaced NaN values in numeric column '%s' with median: %f", col, median_value)
            else:
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                self.logger.info("Replaced NaN values in non-numeric column '%s' with mode: %s", col, mode_value)
            handled_columns[col] = 'nan'

        return handled_columns

    def _scale(self, df, target_col):
        """
        Scales numeric columns in the DataFrame, excluding the target column if specified.

        Parameters:
        - df (pd.DataFrame): DataFrame to scale.
        - target_col (str): Target column to exclude from scaling.

        Returns:
        - pd.DataFrame: Scaled DataFrame.
        """
        numeric_columns = df.select_dtypes(include=['number']).columns
        if target_col and target_col in numeric_columns:
            numeric_columns = numeric_columns.drop(target_col)

        df_scaled = df.copy()
        df_scaled[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        if target_col:
            df_scaled[target_col] = df[target_col]  # Reattach target column if needed
        self.logger.info("Scaling completed.")
        return df_scaled

    def _save_extracted_df(self, df, save_to):
        """
        Saves the DataFrame to a specified file path in Parquet format.

        Parameters:
        - df (pd.DataFrame): DataFrame to save.
        - save_to (str): Path to save the DataFrame.
        """
        df.to_parquet(save_to)
        self.logger.info("Data saved to %s", save_to)
