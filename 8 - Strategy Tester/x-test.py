from lib.StrategyTester import StrategyTester

# Configuration
test_name = "test-7"
data_path = "data/clean_transactions.pq"
split_date = '2019-10-01 00:00:00'
target_col = "Is Fraud"
datetime_col = "Datetime"
num_rounds = 100
model_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 30,
    'tree_method': 'hist',
    'device': 'cuda'
}
evaluation_file = "evaluations.csv"

# Initialize and run the StrategyTester
tester = StrategyTester(
    test_name=test_name,
    data_path=data_path,
    split_date=split_date,
    target_col=target_col,
    datetime_col=datetime_col,
    model_type="XGBoost",
    model_params=model_params,
    num_rounds=num_rounds,
    evaluation_file=evaluation_file
)

tester.run()
