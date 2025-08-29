import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,  ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


# Load data
df= pd.read_csv("water_quality_dataset_100k_preprocessed.csv")

num_features=df.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove('Target')
cat_features=df.select_dtypes(include=['object','category']).columns.tolist()
non_feature_cols = ['Target','Month','Day','Time of Day',]


#normalize of standerize numerical features
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
#convert categorical features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
#combain transformer into a preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])
#apply the transformations to the dataframe
df_preprocessed = preprocessor.fit_transform(df)
#which sahpe of transform data
print("Shape of preprocessed data:", df_preprocessed.shape)

#apply the transformations to the dataframe
df_preprocessed = preprocessor.fit_transform(df)
#which sahpe of transform data
print("Shape of preprocessed data:", df_preprocessed.shape)
#feature Engineering
df['water_tem_to_Air_ratio']=df['Water Temperature']/df['Air Temperature']
df['total_metals']=df[['Iron', 'Lead', 'Copper', 'Zinc', 'Manganese']].sum(axis=1)
num_features.extend(['water_tem_to_Air_ratio', 'total_metals'])



# Apply transformations
X = df.drop('Target', axis=1)
y = df['Target']
X_transformed_df=X[num_features].copy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed_df, y, test_size=0.2, random_state=42)
