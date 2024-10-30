from lib.StrategyTester import StrategyTester

# Configuration
test_name = "test-12"
data_path = "data/clean_transactions.pq"
split_date = '2019-10-01 00:00:00'
target_col = "Is Fraud"
datetime_col = "Datetime"
apply_smote = True
smote_sampling = 0.01
model_type = "CatBoost"
num_rounds = 75 
model_params = {
    "iterations": 1000,  # Increased for better exploration of patterns
    "learning_rate": 0.03,  # Reduced for finer adjustments per iteration
    "depth": 8,  # Keep moderate depth, adjust based on tests
    "loss_function": 'Logloss',
    "eval_metric": 'AUC',
    "verbose": 0,  # Print progress for better tracking
    "random_state": 42,
    "task_type": 'GPU',
    "devices": '0',
    "class_weights": [1, 10],  # Adjusted if true ratio differs
    "early_stopping_rounds": 100,  # Stop early if no improvement
    "l2_leaf_reg": 3,  # Regularization to prevent overfitting
}

evaluation_file = "evaluations.csv"

# Initialize and run the StrategyTester
tester = StrategyTester(
    test_name=test_name,
    data_path=data_path,
    split_date=split_date,
    target_col=target_col,
    datetime_col=datetime_col,
    apply_smote=apply_smote,
    smote_sampling_strategy=smote_sampling,
    model_type=model_type,
    model_params=model_params,
    num_rounds=num_rounds,
    evaluation_file=evaluation_file
)

tester.run()
