# Fraud Detection 🔍💸

This project addresses the problem of fraud detection through a systematic, step-by-step pipeline, where each phase is organized in its own directory. Below is an in-depth look at each directory’s function and the framework’s class structure, which collectively form a comprehensive tool for handling, analyzing, and testing fraud detection strategies.

---

### 0 - Data 🔗 [LinkIcon](/0%20-%20Data/)

The `Data` directory holds every dataset iteration used throughout the process, broken into six sub-directories:

- 🔗 [0 - original](/0%20-%20Data/0%20-%20original/): The initial raw datasets collected from various sources.
- 🔗 [1 - merge](/0%20-%20Data/1%20-%20merge/): Merged versions of the original datasets, ready for cleaning.
- 🔗 [2 - clean](/0%20-%20Data/2%20-%20clean/): Cleaned datasets with invalid values (e.g., NaN, Inf) removed or corrected, ensuring consistency.
- 🔗 [3 - featured](/0%20-%20Data/3%20-%20featured/): Feature-enhanced datasets created by applying specific feature extraction strategies to clean data.
- 🔗 [4 - scaled](/0%20-%20Data/4%20-%20scaled/): Datasets encoded and scaled, where categorical features are one-hot encoded and numerical features are normalized.
- 🔗 [5 - pcas](/0%20-%20Data/5%20-%20pcas/): Dimensionally reduced datasets, obtained by applying PCA or an autoencoder.

Each stage of the `Data` pipeline ensures an optimized dataset that best reflects patterns of potential fraud.

### 1 - Data Processing 🔗 [LinkIcon](/1%20-%20Data%20Processing/)

The `Data Processing` directory contains Jupyter Notebooks that merge data using two libraries:

- 🔗 [Using Pandas](/1%20-%20Data%20Processing/Data%20Processing%20in%20Pandas/): A Pandas-based approach for data handling.
- 🔗 [Using Polars](/1%20-%20Data%20Processing/Data%20Processing%20In%20Polar/): Uses Polars, an alternative library offering enhanced performance with large datasets.

The choice between Pandas and Polars allows for flexibility in data handling, depending on dataset size and computational resources.

### 2 - Data Cleaning 🔗 [LinkIcon](/2%20-%20Data%20Cleaning/)

In the `Data Cleaning` directory, date columns are converted to datetime formats, and numerical columns (e.g., currency) are stripped of symbols and standardized to float or integer formats. This step also encodes categorical columns to UTF-8 and replaces null values, reducing the risk of errors during later analysis.

### 3 - Data Exploration 🔗 [LinkIcon](/3%20-%20Data%20Exploration/)

Exploratory analysis is carried out here to investigate correlations between input features and the target variable, providing insight into potential fraud markers. The analysis highlights dataset imbalances, guiding decisions on preprocessing and model tuning to handle skewed classes.

### 4 - Feature Extraction 🔗 [LinkIcon](/4%20-%20Feature%20Extraction/)

This directory explores different methods for extracting new features, such as identifying time-dependent features, calculating card usage cycles, and analyzing transaction amounts. Each method aims to enhance the dataset’s representation of fraudulent and non-fraudulent transactions.

### 5 - Scaling & Encoding 🔗 [LinkIcon](/5%20-%20Scaling%20&%20Encoding/)

Categorical features are one-hot encoded here, while numeric features undergo scaling using a standard scaler, which normalizes values to improve model performance and avoid bias from varied feature scales.

### 6 - PCA or ED 🔗 [LinkIcon](/6%20-%20PCA%20or%20ED/)

This section documents attempts to reduce the dataset’s dimensionality using Principal Component Analysis (PCA) and autoencoders. These methods aim to streamline data structure without significant information loss, although initial attempts did not yield strong results for fraud detection.

### 7 - Models 🔗 [LinkIcon](/7%20-%20Models/)

This directory contains implementations of several machine learning models, including XGBoost, CATBoost, and a deep neural network. XGBoost emerged as the most effective, with GPU-accelerated support for enhanced training speed, while the Random Forest model was evaluated but limited by its slower, CPU-bound processing.

### 8 - Strategy Tester 🔗 [LinkIcon](/8%20-%20Strategy%20Tester/)

This directory contains a testing framework that iterates through different feature extraction, encoding, and scaling strategies, fine-tuning the model’s hyperparameters to optimize fraud detection. It automates repetitive testing processes, reducing the time needed for experimentation.

**Folder Structure**

- `data`: Contains cleaned data from [Data Cleaning](/2%20-%20Data%20Cleaning/).
- `lib`: Core framework files for feature extraction, preprocessing, encoding, and logging.
- `strategies`: Stores individual strategy files for feature extraction.
- `tests`: Logs, evaluations, and parameter settings for each test strategy.

**Framework Architecture**

The framework is built around modular classes that handle data processing, feature extraction, model training, and evaluation.

---

### Classes

**FeatureExtractor**

The `FeatureExtractor` class loads and applies a feature extraction strategy:

- **Constructor Inputs**:
  - `strategy_name`: Name of the feature extraction strategy to apply.
  - `data_path`: Path to the data file.
- **Functionality**: Initializes with the specified strategy and dynamically loads the function from a corresponding file in the `strategies` directory. When `extract_features()` is called, it reads the dataset, applies the strategy function, and returns the transformed data.

- **Communication**: The `FeatureExtractor` class connects with `Logger` for logging and `FeatureExtractorFactory` for instance creation.

**FeatureExtractorFactory**

This factory function, `create_feature_extractor`, provides a standardized way to instantiate `FeatureExtractor` objects. It accepts `strategy_name` and `data_path` as parameters and returns an initialized `FeatureExtractor`.

**Logger**

The `Logger` class manages centralized, rotating logs for each test session:

- **Constructor Inputs**:

  - `test_name`: Name of the test to ensure unique log files.

- **Functionality**: Configures a rotating file handler to store logs for each session in a unique directory under `tests`. Logging instances are reused across the framework’s classes, maintaining consistency in output.

- **Communication**: `Logger` interacts with other classes, like `StrategyTester`, `FeatureExtractor`, `Preprocessor`, `ScaleEncode`, and `useModel`, for uniform logging.

**Preprocessor**

The `Preprocessor` class splits data into training and testing sets based on a specified date:

- **Constructor**: Initializes `Logger` for tracking operations.

- **Main Method**: `split_by_date`:

  - **Inputs**:

    - `X`: Feature DataFrame.
    - `y`: Target variable.
    - `split_date`: Date to divide the dataset into training and testing.
    - `datetime_col`: Column containing date information.

  - **Functionality**: Auto-detects the datetime column if not provided, converts it to datetime format, and splits data accordingly. Returns training and testing sets for both features and targets.

- **Communication**: `Preprocessor` collaborates with `Logger` and `ScaleEncode` for encoding and scaling, enhancing data consistency across tests.

**ScaleEncode**

The `ScaleEncode` class handles scaling and encoding of data, with options for auto-detecting categorical columns or specifying columns for encoding:

- **Constructor Inputs**:
  - `data_path` or `data`: Path to the data file or a DataFrame.
  - `scaler`: Scaler instance (e.g., `StandardScaler`).
- **Main Methods**:

  - `scale_and_encode`: Encodes specified categorical columns and scales numeric data, handling missing and infinite values.
  - `_auto_encode`: Automatically one-hot encodes categorical columns.
  - `_find_and_handle_unsafe_columns`: Replaces NaN and infinite values with median/mode as necessary.

- **Communication**: `ScaleEncode` interacts with `Logger` for logging and with `Preprocessor` for ensuring processed data consistency.

**useModel**

The `useModel` class adapts to multiple model types, handling training and evaluation:

- **Constructor Inputs**:

  - `model_type`: Model selection, currently supporting "XGBoost" and "CatBoost".
  - `params`: Model-specific parameters.

- **Main Methods**:

  - `train`: Trains the selected model on training data.
  - `predict`: Generates predictions on test data.
  - `evaluate`: Outputs precision, recall, and F1 scores, saving evaluation metrics to a designated file.

- **Communication**: `useModel` integrates with `StrategyTester` for evaluating strategies and with `Logger` for consistent logging across sessions.

**StrategyTester**

The `StrategyTester` class orchestrates the testing pipeline, encompassing feature extraction, scaling, encoding, and model evaluation:

- **Constructor Inputs**:
  - `test_name`: Name of the test for unique identification.
  - `data_path`: Path to the dataset.
  - `split_date`: Date to split the dataset.
  - `target_col`: Target column (e.g., "Is Fraud").
  - `

datetime_col`: Datetime column for split.

- `model_type`: Type of model to use.
- `model_params`: Parameters for the selected model.
- `num_rounds`: Training rounds (e.g., for XGBoost).
- `scaler`: Scaler instance.
- `evaluation_file`: Path to save evaluation metrics.

- **Functionality**: Saves test parameters, iterates through strategies in the `strategies` directory, and executes each one. The pipeline involves:

  1. Initializing `FeatureExtractor` through `FeatureExtractorFactory`.
  2. Applying `ScaleEncode` for encoding/scaling.
  3. Splitting data via `Preprocessor`.
  4. Training and evaluating models using `useModel`.

- **Communication**: The `StrategyTester` is the central coordinating class, interacting with `FeatureExtractorFactory`, `ScaleEncode`, `Preprocessor`, `useModel`, and `Logger`.

---

**Diagram of Communication**

![architecture](/img/diagram.png)

### Results

Using the `Strategy Tester`, I evaluated multiple strategies available [here](/8%20-%20Strategy%20Tester/strategies/) with various model configurations. The tests were conducted on both XGBoost and CatBoost models. For detailed information on specific tests, such as `test-1`, refer to the [tests/test-1](/8%20-%20Strategy%20Tester/tests/test-1/) directory. This directory contains the following files:

- `evaluations.csv` – Evaluation metrics for each strategy in the current test.
- `strategy_test.log` – Logs from the test execution.
- `test_params.json` – Model configurations used in the test.

These files are auto-generated by the `Strategy Tester` for each test.

**Overall Results**

The [results](/8%20-%20Strategy%20Tester/results/) directory includes a [Jupyter Notebook](/8%20-%20Strategy%20Tester/results/ntb.ipynb) that, when executed, automatically aggregates data from each test in the tests directory. It provides a comprehensive analysis of the tests and identifies the best overall strategy and model settings.
