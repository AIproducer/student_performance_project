import os
import sys
from dataclasses import dataclass

#from catboost import CatBoostClassifier
from sklearn.ensemble import(
    RandomForestClassifier,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass

class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Entered the model trainer component")
            logging.info("Spilt training and test input data")
            X_train,y_train,X_test,y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]

            )

            models= {
                'LinearRegression':LinearRegression(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'XGBRegressor':XGBRegressor(),
                'KNeighborsRegressor':KNeighborsRegressor()
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            ## To get bset model score from dict

            best_model_score = max(sorted(model_report.values()))
            ## to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_name = models[best_model_name]
            if best_model_score<.06:
                raise CustomException("No best model found")
            logging.info(f"Best found model on the both training and test dataset")


            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model_name
            )
            predicted=best_model_name.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)
