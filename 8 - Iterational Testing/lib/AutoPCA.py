import pandas as pd
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutoPCA:
    def __init__(self, data_path):
        self.df_path = data_path
        logging.info("AutoPCA initialized with data path: %s", data_path)

    def decompose(self, out_dim: int, save_to: str, exclude_cols: list = None):
        logging.info("Starting PCA decomposition with output dimensions: %d", out_dim)
        
        # Read the dataframe
        df = self._read_dataframe()
        logging.info("Data successfully loaded from %s", self.df_path)
        
        # Drop columns if specified
        if exclude_cols:
            logging.info("Excluding columns from PCA: %s", exclude_cols)
            X = df.drop(exclude_cols, axis=1)
        else:
            X = df
        logging.info("Data shape after excluding columns: %s", X.shape)
        
        # Apply PCA
        pca = PCA(n_components=out_dim)
        X_pca = pca.fit_transform(X)
        logging.info("PCA applied. Reduced data shape: %s", X_pca.shape)
        
        # Create a new DataFrame with principal components
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        
        # Reattach excluded columns if any
        if exclude_cols:
            pca_df[exclude_cols] = df[exclude_cols]
            logging.info("Excluded columns reattached to the PCA dataframe.")
        
        # Save the result
        self._save_extracted_df(pca_df, save_to)
        logging.info("PCA-transformed dataframe saved to %s", save_to)

    def _read_dataframe(self):
        if not self.df_path:
            raise ValueError("Data path must be specified!")
        df = pd.read_parquet(self.df_path)
        return df

    def _save_extracted_df(self, extracted_df: pd.DataFrame, save_to: str):
        extracted_df.to_parquet(save_to)
