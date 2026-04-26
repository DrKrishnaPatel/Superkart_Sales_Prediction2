# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation and scale numerical data
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/drkrishnapatel/SuperkartSalesPrediction/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['Product_Id'], inplace=True)

# Convert 'year_of_establishment' to categorical
df['Store_Establishment_Year'] = df['Store_Establishment_Year'].astype('category')

# Get Numerical Columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

# Define columns
ordinal_cols = ['Product_Sugar_Content', 'Store_Size', 'Store_Location_City_Type']

# Check status safely
for col in ordinal_cols:
    if col in df.columns:
        # Optional: Convert to category first if you want to use .cat
        df[col] = df[col].astype('category')
        print(f"{col} is ordered: {df[col].cat.ordered}")

# Encode the columns properly
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for col in ordinal_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

print("Encoding complete for ordinal columns.")

target_col = 'Product_Store_Sales_Total'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="drkrishnapatel/SuperkartSalesPrediction",
        repo_type="dataset",
    )
