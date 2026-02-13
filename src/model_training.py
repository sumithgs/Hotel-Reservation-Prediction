import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from src.logger import get_logger
from src.custom_exception import CustomException

from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data

from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = get_logger("__name__")

class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
    
    def load_and_split_data(self):
        try:
            logger.info(f"Loading the data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading the data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns="booking_status")
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns="booking_status") 
            y_test = test_df["booking_status"]

            logger.info("Data splitted successfully for model training")
            return X_train,y_train,X_test,y_test
        
        except Exception as e:
            logger.error("Error while splitting the data")
            raise CustomException("Error during the data splitting step",e)


    def trainlgbm(self,X_train,y_train):
        try:
            logger.info("Initializing the model")
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])

            logger.info("Starting our Hyperparameter tuning")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )
            logger.info("Starting our Hyperparameter tuning")
            
            random_search.fit(X_train,y_train)
            
            logger.info("Hyperparametr tuning completed!")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best parameters are : {best_params}")    
            return best_lgbm_model
            
        except Exception as e:
            logger.error("Error while training LGBM model")
            raise CustomException("Failed to train the model",e)
    
    def evaluatemodel(self, model,X_test,y_test):
        try:
            logger.info("Evaluating the LGBM Model")
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)

            logger.info(f"Accuracy Score : {accuracy}")
            logger.info(f"Precision Score : {precision}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"F1 Score : {f1}")

            return {
                "accuracy" : accuracy,
                "precision" : precision,
                "recall" : recall,
                "f1" : f1
            }
        
        except Exception as e:
            logger.error("Error while evaluating")
            raise CustomException("Failed to evaluate the model",e)
        
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            logger.info("Saving the model")
            joblib.dump(model,self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")
        
        except Exception as e:
            logger.error("Error while saving the model")
            raise CustomException("Failed to save the model", e)

    def run(self):
        try:
            # Ensure MLflow artifact directory exists
            os.makedirs(MLFLOW_ARTIFACTS_DIR, exist_ok=True)

            # Set MLflow tracking URI to a safe workspace path
            mlflow.set_tracking_uri(f"file://{os.path.abspath(MLFLOW_ARTIFACTS_DIR)}")

            with mlflow.start_run():
                logger.info("Starting the Model Training Step")
                logger.info("Starting our MLflow experimentation")

                # Log datasets safely
                logger.info("Logging the training and testing dataset to MLflow")
                os.makedirs(os.path.join(MLFLOW_ARTIFACTS_DIR, "datasets"), exist_ok=True)
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                # Load data
                X_train, y_train, X_test, y_test = self.load_and_split_data()

                # Train model
                best_lgbm_model = self.trainlgbm(X_train, y_train)

                # Evaluate model
                metrics = self.evaluatemodel(best_lgbm_model, X_test, y_test)

                # Save model
                self.save_model(best_lgbm_model)

                # Log the model artifact
                logger.info("Logging the model to MLflow")
                mlflow.log_artifact(self.model_output_path, artifact_path="models")

                # Log parameters and metrics
                logger.info("Logging params and metrics into MLflow!")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model training successfully completed!")

        except Exception as e:
            logger.error("Error while training the model", exc_info=True)
            raise CustomException("Failed to train the model", e)


if __name__=="__main__":
    obj = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    obj.run()