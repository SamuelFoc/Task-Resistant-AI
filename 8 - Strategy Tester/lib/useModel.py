import os
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from lib.Logger import Logger


class useModel:
    def __init__(self, model_type='XGBoost', params=None):
        """
        Initializes the useModel class for a specified model type.

        Parameters:
        - model_type (str): Type of model ('XGBoost' or 'CatBoost').
        - params (dict): Parameters for the model.
        """
        self.logger = Logger().get_logger(self.__class__.__name__)
        self.model_type = model_type
        self.model_strategy = self._set_model_strategy(params)
        self.logger.info("useModel initialized for %s.", model_type)

    def _set_model_strategy(self, params):
        if self.model_type == 'XGBoost':
            return XGBoostStrategy(params)
        elif self.model_type == 'CatBoost':
            return CatBoostStrategy(params)
        else:
            raise ValueError("Unsupported model type: choose 'XGBoost' or 'CatBoost'.")

    def train(self, X_train, y_train, num_rounds=100):
        self.model_strategy.train(X_train, y_train, num_rounds)

    def predict(self, X_test):
        return self.model_strategy.predict(X_test)

    def evaluate(self, y_test, y_pred, file_path, data_label):
        self.logger.info("Evaluating model.")
        
        # Generate classification report as a dictionary
        report = classification_report(y_test, y_pred, output_dict=True)

        # Extract metrics for both classes '0' and '1'
        precision_0 = report['0']['precision']
        recall_0 = report['0']['recall']
        f1_score_0 = report['0']['f1-score']
        precision_1 = report['1']['precision']
        recall_1 = report['1']['recall']
        f1_score_1 = report['1']['f1-score']

        # Prepare the data line for writing
        data_line = f"{data_label},{precision_0},{recall_0},{f1_score_0},{precision_1},{recall_1},{f1_score_1}"

        # Write header if the file does not exist
        header = "data_label,precision_0,recall_0,f1_score_0,precision_1,recall_1,f1_score_1"
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(header + "\n")

        # Append data line to the file
        with open(file_path, 'a') as f:
            f.write(data_line + "\n")
        self.logger.info("Evaluation results saved to %s.", file_path)



class XGBoostStrategy:
    def __init__(self, params=None):
        """
        Initializes XGBoost model with given parameters.
        """
        self.params = params or {'objective': 'binary:logistic'}
        self.model = None

    def train(self, X_train, y_train, num_boost_round=100):
        """
        Trains the XGBoost model.
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain, num_boost_round=num_boost_round)

    def predict(self, X_test, threshold=0.5):
        """
        Predicts using the trained XGBoost model and applies a threshold to convert
        probabilities to binary class labels.

        Parameters:
        - X_test (pd.DataFrame): Test data features.
        - threshold (float): Probability threshold for binary classification.

        Returns:
        - List[int]: Binary predictions.
        """
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = self.model.predict(dtest)
        y_pred = [1 if prob > threshold else 0 for prob in y_pred_proba]
        return y_pred


class CatBoostStrategy:
    def __init__(self, params=None):
        """
        Initializes CatBoost model with given parameters.
        """
        self.params = params or {'iterations': 100}
        self.model = CatBoostClassifier(**self.params)

    def train(self, X_train, y_train, iterations=100):
        """
        Trains the CatBoost model.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts using the trained CatBoost model.

        Parameters:
        - X_test (pd.DataFrame): Test data features.

        Returns:
        - List[int]: Binary predictions.
        """
        return self.model.predict(X_test).tolist()
