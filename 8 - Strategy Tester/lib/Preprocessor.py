import pandas as pd
from lib.Logger import Logger
from imblearn.over_sampling import SMOTE


class Preprocessor:
    def __init__(self):
        """
        Initialize the Preprocessor class.
        """
        self.logger = Logger().get_logger(self.__class__.__name__)
        self.logger.info("Preprocessor initialized.")

    def split_by_date(self, X, y, split_date, datetime_col=None, smote=False, sampling_strategy=0.5):
        """
        Splits data into training and testing sets based on a specified date.

        Parameters:
        - X (pd.DataFrame): Feature DataFrame containing a datetime column.
        - y (pd.Series): Target Series aligned with X.
        - split_date (str): Date string (YYYY-MM-DD) to use as the split point.
        - datetime_col (str, optional): Column name for datetime data in X.
          If not provided, attempts to auto-detect.

        Returns:
        - X_train (pd.DataFrame): Training feature set.
        - X_test (pd.DataFrame): Testing feature set.
        - y_train (pd.Series): Training target set.
        - y_test (pd.Series): Testing target set.
        """
        self.logger.info("Splitting data by date: %s", split_date)

        # Identify datetime column if not specified
        if not datetime_col:
            datetime_col = self._detect_datetime_column(X)
            if not datetime_col:
                self.logger.error("No datetime column provided or found in DataFrame.")
                raise ValueError("Datetime column not provided and could not be auto-detected.")

        # Ensure the datetime column is in datetime format
        X = X.copy()
        X[datetime_col] = pd.to_datetime(X[datetime_col], errors='coerce')
        if X[datetime_col].isnull().all():
            self.logger.error("Datetime column could not be parsed as dates.")
            raise ValueError("Provided datetime column could not be parsed as dates.")

        # Split the data based on the split_date
        train_data = X[X[datetime_col] < split_date]
        test_data = X[X[datetime_col] >= split_date]

        # Drop the datetime column after splitting
        X_train = train_data.drop(columns=[datetime_col])
        X_test = test_data.drop(columns=[datetime_col])

        # Align target variable (y) with the indices of train and test sets
        y_train = y[train_data.index]
        y_test = y[test_data.index]

        # Log the results of the split
        self.logger.info("Data split complete: %d training samples, %d test samples.", len(X_train), len(X_test))

        # SMOTE alg
        if smote:
            self.logger.info("Applying SMOTE to address class imbalance.")
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            
            # Log the class distribution before applying SMOTE
            class_counts_before = y_train.value_counts().to_dict()
            self.logger.info(f"Class distribution before SMOTE: {class_counts_before}")
            
            # Apply SMOTE
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            # Log the class distribution after applying SMOTE
            class_counts_after = y_train.value_counts().to_dict()
            self.logger.info(f"Class distribution after SMOTE: {class_counts_after}")
        
        return X_train, X_test, y_train, y_test

    def _detect_datetime_column(self, df):
        """
        Attempts to auto-detect a datetime column in the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame to inspect.

        Returns:
        - str or None: Name of the detected datetime column, or None if not found.
        """
        datetime_candidates = df.select_dtypes(include=['datetime', 'object']).columns
        for col in datetime_candidates:
            try:
                if pd.to_datetime(df[col], errors='coerce').notna().sum() > 0:
                    self.logger.info("Auto-detected datetime column: %s", col)
                    return col
            except Exception:
                continue
        return None
