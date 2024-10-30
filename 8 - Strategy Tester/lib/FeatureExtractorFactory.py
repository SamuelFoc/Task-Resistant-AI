from lib.FeatureExtractor import FeatureExtractor

def create_feature_extractor(strategy_name, data_path):
    return FeatureExtractor(strategy_name=strategy_name, data_path=data_path)
