import os
import joblib 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("water_quality_dataset_100k_new.csv")
    df["ph_cat"] = pd.cut(df["pH"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
    df["ph_cat"] = df["ph_cat"].fillna(df["ph_cat"].mode()[0])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["ph_cat"]):
        strat_train_set = df.loc[train_index].drop("ph_cat",axis=1)
        strat_test_set = df.loc[test_index].drop("ph_cat",axis=1)
        
        sample = strat_train_set.drop("Target", axis=1)
        sample_labels = strat_train_set["Target"].copy()

        numeric_cols = sample.select_dtypes(include=['float64', 'int64']).columns
        object_cols = sample.select_dtypes(include=['object', 'category']).columns

        pipeline = build_pipeline(numeric_cols, object_cols)
        sample_prepared = pipeline.fit_transform(sample)
        
        model = RandomForestRegressor(random_state=42)
        model.fit(sample_prepared, sample_labels)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(pipeline, PIPELINE_FILE)
        print("Model is trained. Congrats!")
