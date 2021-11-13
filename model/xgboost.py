import xgboost as xgb
import pickle

def xgboost_construction(is_training, log_dir):
    model = xgb.XGBRegressor()

    if is_training:
        return model
    else:
        print("Load model from: {}".format(log_dir))
        filename = log_dir + "best_model.sav"
        return pickle.load(open(filename, 'rb'))