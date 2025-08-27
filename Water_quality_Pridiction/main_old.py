import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the dataset
df = pd.read_csv("water_quality_dataset_100k_new.csv")


df["ph_cat"] = pd.cut(df["pH"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
df["ph_cat"] = df["ph_cat"].fillna(df["ph_cat"].mode()[0])


# Assume income_cat is a column in the dataset created from median_income
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df["ph_cat"]):
    strat_train_set = df.loc[train_index].drop("ph_cat",axis=1)
    strat_test_set = df.loc[test_index].drop("ph_cat",axis=1)

# We will work on the copy of training data 
sample = strat_train_set.copy()

# 3. Seperate features and labels
sample= strat_train_set.drop("Target", axis=1)
sample_labels = strat_train_set["Target"].copy()



# Separate numeric and object columns
numeric_cols = sample.select_dtypes(include=['float64', 'int64']).columns
object_cols = sample.select_dtypes(include=['object', 'category']).columns

# 5. Lets make the pipeline 

# For numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# For categorical columns
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Combine numerical and categorical pipelines using ColumnTransformer
full_pipeline = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), object_cols)
])

# Fit and transform the sample data
sample_prepared = full_pipeline.fit_transform(sample)
print(sample_prepared.shape)