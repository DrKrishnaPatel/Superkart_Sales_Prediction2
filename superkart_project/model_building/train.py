# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder # Added OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow


import mlflow
# This tells MLflow to save logs to a folder named 'mlruns' in the runner
mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment("mlops-training-experiment")


# NOTE: In a production environment, ngrok should not be used as it exposes a local server.
# MLflow tracking URI should be set to a persistent MLflow server.
# For the purpose of this Colab demonstration, we'll keep the ngrok setup similar to the dev environment.
# However, for actual production deployment, you would point to a dedicated MLflow server.


api = HfApi()


Xtrain_path = "hf://datasets/drkrishnapatel/SuperkartSalesPrediction/Xtrain.csv"
Xtest_path = "hf://datasets/drkrishnapatel/SuperkartSalesPrediction/Xtest.csv"
ytrain_path = "hf://datasets/drkrishnapatel/SuperkartSalesPrediction/ytrain.csv"
ytest_path = "hf://datasets/drkrishnapatel/SuperkartSalesPrediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# One-hot encode categorical features and scale numeric features
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
]
categorical_features = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Id',
    'Store_Establishment_Year',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
    ]

# Define your categorical and ordinal lists
# Note: You must manually specify which columns are ordinal vs nominal
categorical_cols = ['Product_Type', 'Store_Id', 'Store_Establishment_Year', 'Store_Type']
ordinal_cols = ['Product_Sugar_Content', 'Store_Size', 'Store_Location_City_Type']

# ... (all your imports remain the same) ...

# Define the correct category orders for the OrdinalEncoder
# Make sure these match the unique values in your CSV exactly
sugar_order = ['Low Sugar', 'Medium Sugar', 'High Sugar']
size_order = ['Small', 'Medium', 'High']
city_order = ['Tier 3', 'Tier 2', 'Tier 1']

# Create the preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    # Pass the categories list here to maintain the rank
    (OrdinalEncoder(categories=[sugar_order, size_order, city_order], 
                    handle_unknown='use_encoded_value', 
                    unknown_value=-1), ordinal_cols)
)

# Define base XGBoost model
xgb_regressor = xgb.XGBRegressor(random_state=42)

# make_pipeline automatically names 'XGBRegressor' as 'xgbregressor' (lowercase)
model_pipeline = make_pipeline(preprocessor, xgb_regressor)

# Define hyperparameter grid
# Ensure these names match the lowercase name of the class
param_grid = {
    'xgbregressor__n_estimators': [100, 200],
    'xgbregressor__max_depth': [3, 5],
    'xgbregressor__learning_rate': [0.01, 0.1]
}


# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i] # This is typically validation score
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_validation_score", mean_score)
            mlflow.log_metric("std_validation_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    def get_adj_r2(y_true, y_pred, n_samples, n_features):
        """Calculates Adjusted R-Squared."""
        r2 = r2_score(y_true, y_pred)
        adjr2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        return adjr2

    # Calculate regression metrics
    train_rmse = mean_squared_error(ytrain, y_pred_train)**0.5
    train_mae = mean_absolute_error(ytrain, y_pred_train)
    train_r2 = r2_score(ytrain, y_pred_train)
    train_adjr2 = get_adj_r2(ytrain, y_pred_train, Xtrain.shape[0], Xtrain.shape[1])
    train_mape = mean_absolute_percentage_error(ytrain, y_pred_train)


    test_rmse = mean_squared_error(ytest, y_pred_test)**0.5
    test_mae = mean_absolute_error(ytest, y_pred_test)
    test_r2 = r2_score(ytest, y_pred_test)
    test_adjr2 = get_adj_r2(ytest, y_pred_test, Xtest.shape[0], Xtest.shape[1])
    test_mape = mean_absolute_percentage_error(ytest, y_pred_test)


    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "train_adjr2": train_adjr2,
        "train_mape": train_mape,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "test_adjr2": test_adjr2,
        "test_mape": test_mape
    })


    # Save the model locally
    model_path = "best_superkart_product_sales_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "drkrishnapatel/SuperkartSalesPrediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type="model", private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_superkart_product_sale_model_v1.joblib",
        path_in_repo="best_superkart_product_sales_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
