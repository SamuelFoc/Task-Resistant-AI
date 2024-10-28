# lib

---

## `FeatureExtractor` Class

### Purpose

The `FeatureExtractor` class is designed to dynamically import and apply a feature extraction function (`strategy`) from an external Python file. This function modifies a given DataFrame according to user-defined logic, making it flexible for various data processing needs.

### Initialization

```python
FeatureExtractor(strategy_file: str)
```

- **`strategy_file`**: The filename (located in `./extraction_scripts/`) containing a `strategy` function. This file should define a function called `strategy(df: pd.DataFrame) -> pd.DataFrame`, which receives a DataFrame as input and returns a modified DataFrame.

### Methods

#### `extract_features`

```python
extract_features(data_path: str, save_to: str)
```

- **`data_path`**: Path to the `.parquet` file containing the data to be processed.
- **`save_to`**: Path where the transformed DataFrame will be saved after the `strategy` function is applied.

This method:

1. Loads the data from `data_path`.
2. Dynamically imports the specified `strategy` function from the `strategy_file`.
3. Applies the `strategy` function to modify the DataFrame.
4. Saves the transformed DataFrame to `save_to`.

#### Example Usage

Assuming a `strategy` function is defined in `extraction_scripts/my_strategy_file.py`, which modifies the DataFrame as required:

```python
# Initialize FeatureExtractor with the strategy file name
extractor = FeatureExtractor("my_strategy_file.py")

# Perform feature extraction on data, saving the output to specified path
extractor.extract_features(data_path="data/my_data.parquet", save_to="output/processed_data.parquet")
```

### Logging

- Logs initialization and confirmation of the specified strategy file.
- Logs successful data loading and the start of the feature extraction process.
- Logs successful import of the `strategy` function and its application to the DataFrame.
- Logs successful saving of the processed DataFrame.

### Key Notes

- **Dynamic Import of Strategy Function**: The class uses `importlib` to import the `strategy` function at runtime, allowing users to change feature extraction logic by swapping out the strategy file.
- **Error Handling**: If the `strategy` function is missing in the specified file, an `AttributeError` is raised, logging an error message for easier debugging.
- **Flexibility**: The `strategy` function can be tailored to any processing needs, making `FeatureExtractor` a versatile tool for data preparation.

---

## `ScaleEncode` Class

### Purpose

The `ScaleEncode` class is designed to:

1. Encode categorical columns in a dataframe using one-hot encoding.
2. Scale numerical columns, excluding specified target columns.

### Initialization

```python
ScaleEncode(data_path: str)
```

- **`data_path`**: Path to the dataset (in `.parquet` format) to be processed.

### Methods

#### `scale_and_encode`

```python
scale_and_encode(save_to: str, target_col: str = None, scaler = None, specific_columns_to_encode: list = None)
```

- **`save_to`**: Path where the transformed data will be saved as a `.parquet` file.
- **`target_col`**: Name of the target column to exclude from scaling. Defaults to `None`.
- **`scaler`**: An instance of a scaler (e.g., `StandardScaler`, `MinMaxScaler`) from `sklearn.preprocessing`. If `None`, no scaling is applied.
- **`specific_columns_to_encode`**: A list of categorical columns to encode. If not specified, all categorical columns are encoded.

#### Example Usage

```python
from sklearn.preprocessing import StandardScaler

# Initialize ScaleEncode with data path
se = ScaleEncode(data_path="data/my_data.parquet")

# Scale and encode with StandardScaler, saving to specified path
se.scale_and_encode(
    save_to="output/scaled_encoded_data.parquet",
    target_col="target",
    scaler=StandardScaler(),
    specific_columns_to_encode=["Category1", "Category2"]
)
```

### Logging

- Logs initialization, data loading, encoding and scaling processes, and saving.
- Warnings are issued if numeric columns with invalid values (e.g., infinity) are detected.

### Key Notes

- Automatically encodes all categorical columns if `specific_columns_to_encode` is not specified.
- Scales numeric columns only if a `scaler` is provided.

---

## `AutoPCA` Class

### Purpose

The `AutoPCA` class applies Principal Component Analysis (PCA) to a dataset to reduce dimensionality while preserving important variance. It optionally excludes specific columns from PCA (e.g., categorical or identifier columns).

### Initialization

```python
AutoPCA(data_path: str)
```

- **`data_path`**: Path to the dataset (in `.parquet` format) for PCA processing.

### Methods

#### `decompose`

```python
decompose(out_dim: int, save_to: str, exclude_cols: list = None)
```

- **`out_dim`**: The number of principal components to retain.
- **`save_to`**: Path where the PCA-transformed data will be saved as a `.parquet` file.
- **`exclude_cols`**: A list of column names to exclude from PCA transformation. These columns will be retained in the final DataFrame after PCA transformation.

#### Example Usage

```python
# Initialize AutoPCA with data path
pca_processor = AutoPCA(data_path="data/my_data.parquet")

# Apply PCA to reduce dimensions to 5, saving output and retaining 'ID' and 'Label' columns
pca_processor.decompose(out_dim=5, save_to="output/pca_data.parquet", exclude_cols=["ID", "Label"])
```

### Logging

- Logs initialization, data loading, excluded columns, PCA application, and saving.
- Indicates the original and reduced data shapes to verify transformation.

### Key Notes

- PCA transformation is applied to all numeric columns, excluding specified columns.
- The transformed data retains any excluded columns in the final output.

---
