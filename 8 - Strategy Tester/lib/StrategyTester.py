import os
import json
from lib.FeatureExtractorFactory import create_feature_extractor
from lib.ScaleEncode import ScaleEncode
from lib.useModel import useModel
from lib.Preprocessor import Preprocessor
from lib.Logger import Logger  # Import the Logger class
from sklearn.preprocessing import StandardScaler

class StrategyTester:
    def __init__(
        self,
        test_name,
        data_path,
        split_date,
        apply_smote=False,
        smote_sampling_strategy=0.5,
        target_col="Is Fraud",
        datetime_col="Datetime",
        model_type="XGBoost",
        model_params=None,
        num_rounds=100,
        scaler=None,
        evaluation_file="evaluations.csv"
    ):
        """
        Initializes the StrategyTester class with configuration details.

        Parameters:
        - data_path (str): Path to the data file.
        - split_date (str): Date string to split the data for train-test.
        - target_col (str): Target column in the data.
        - datetime_col (str): Datetime column in the data for splitting.
        - model_type (str): Type of model to use ("XGBoost" or "CatBoost").
        - model_params (dict): Parameters for the model.
        - scaler: Scaler instance (default is StandardScaler).
        - evaluation_file (str): Path to save the evaluation results.
        """
        # Initialize the singleton logger with the test name
        self.logger = Logger(test_name).get_logger(self.__class__.__name__)
        self.test_name = test_name
        self.data_path = data_path
        self.split_date = split_date
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.model_type = model_type
        self.model_params = model_params or {}
        self.num_rounds = num_rounds
        self.scaler = scaler or StandardScaler()
        self.evaluation_file = f"tests/{test_name}/{evaluation_file}"
        self.preprocessor = Preprocessor()
        self.apply_smote = apply_smote
        self.smote_sampling = smote_sampling_strategy
        
        # Save the test parameters to JSON
        self.save_test_params()

    def get_strategies(self, directory="strategies"):
        """Returns a list of strategy files in the specified directory, excluding __pycache__."""
        return [file.split(".")[0] for file in os.listdir(directory) if file.endswith(".py") and file != "__pycache__"]

    def save_test_params(self):
        """Saves the test configuration parameters to a JSON file."""
        test_dir = f"tests/{self.test_name}"
        os.makedirs(test_dir, exist_ok=True)  # Create directory if it doesn't exist
        params_file = os.path.join(test_dir, "test_params.json")

        params = {
            "test_name": self.test_name,
            "data_path": self.data_path,
            "split_date": self.split_date,
            "target_col": self.target_col,
            "datetime_col": self.datetime_col,
            "model_type": self.model_type,
            "model_params": self.model_params,
            "evaluation_file": self.evaluation_file,
            "smote": self.apply_smote,
            "smote_sampling": self.smote_sampling
        }

        with open(params_file, 'w') as f:
            json.dump(params, f, indent=4)
        self.logger.info(f"Test parameters saved to {params_file}")

    def run(self):
        """Runs the testing pipeline for each strategy."""
        for strategy_name in self.get_strategies():
            self.logger.info(f"Testing strategy: {strategy_name}")
            # Initialize and extract features using the strategy
            fe = create_feature_extractor(strategy_name, self.data_path)
            extracted_df = fe.extract_features()
            
            # Scale and encode data
            se = ScaleEncode(data=extracted_df, scaler=self.scaler)
            scaled_encoded_df = se.scale_and_encode(target_col=self.target_col)

            # Separate features (X) and target (y)
            X = scaled_encoded_df.drop(columns=[self.target_col])
            y = scaled_encoded_df[self.target_col]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = self.preprocessor.split_by_date(
                X, y, self.split_date, datetime_col=self.datetime_col, smote=self.apply_smote, sampling_strategy=self.smote_sampling
            )

            # Initialize and train the model
            model = useModel(model_type=self.model_type, params=self.model_params)
            model.train(X_train, y_train, self.num_rounds)

            # Make predictions and evaluate
            y_pred = model.predict(X_test)
            model.evaluate(y_test, y_pred, file_path=self.evaluation_file, data_label=f"{strategy_name}_test")
