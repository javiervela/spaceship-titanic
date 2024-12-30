#!/usr/bin/env python
# coding: utf-8

# # Spaceship Titanic - Notebook
# 
# <!-- TODO -->
# 

# Import the necessary libraries. We will use:
# 
# - `pandas` to load the data and manipulate it.
# - `scikit-learn` to build the model.
# <!-- TODO - `matplotlib` and `seaborn` to plot the data. -->
# 

# In[ ]:


import os
import json
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from featuretools import EntitySet, dfs

import optuna
from optuna.samplers import TPESampler

import signal


# 

# In[3]:


# Define constants
CURRENT_DIR = os.getcwd()

DATA_DIR = f"{CURRENT_DIR}/data"
TRAIN_DATA_FILE = f"{DATA_DIR}/train.csv"
TEST_DATA_FILE = f"{DATA_DIR}/test.csv"

TARGET_COLUMN = "Transported"
ID_COLUMN = "PassengerId"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2

MISSING_VALUE = "Missing"


# In[4]:


# Load the data files into pandas dataframes
train_data = pd.read_csv(TRAIN_DATA_FILE)
test_data = pd.read_csv(TEST_DATA_FILE)


# ## Data Exploration
# 

# In[ ]:


print("First few rows of data:")
print(train_data.head())


# In[ ]:


print("Data columns and types:")
print(train_data.dtypes)


# In[7]:


NUMERICAL_COLUMNS = train_data.select_dtypes(include=[np.number]).columns.tolist()
CATEGORICAL_COLUMNS = train_data.select_dtypes(include=["object"]).columns.tolist()


leftover_columns = [
    col
    for col in train_data.columns
    if col not in NUMERICAL_COLUMNS
    and col not in CATEGORICAL_COLUMNS
    and col != TARGET_COLUMN
]
assert not leftover_columns


# In[ ]:


print(f"Numerical columns: {NUMERICAL_COLUMNS}")
print(f"Categorical columns: {CATEGORICAL_COLUMNS}")
print(f"Target column: {TARGET_COLUMN}")


# In[ ]:


print("\nSummary statistics:")
print(train_data.describe())

print("\nMissing values:")
print(train_data.isnull().sum())

# print("\nCorrelation matrix:")
# sns.heatmap(train_data[NUMERICAL_COLUMNS].corr(), annot=True)
# # plt.show()

print("\nValue counts for categorical variables:")
for col in CATEGORICAL_COLUMNS:
    print(f"\n{col} value counts:")
    print(train_data[col].value_counts())


# In[ ]:


train_data


# ## Clean Dataset
# 
# We need to clean the train and test datasets the same way
# 

# In[11]:


def clean_data(data: pd.DataFrame):

    data = data.copy()

    # Convert columns to integer (with missing values)
    for col in [
        "CabinNumber",
        "CryoSleep",
        "VIP",
        "Transported",
        "PassengerGroupId",
        "PassengerIntraGroupId",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype("Int64")

    # Make PassengerId the index
    data.set_index(ID_COLUMN, inplace=True)

    # Drop columns
    for col in [
        "Name",
        "Cabin",
        "PassengerGroupId",
        "PassengerIntraGroupId",
    ]:
        if col in data.columns:
            data.drop(columns=col, inplace=True)

    return data


# In[12]:


# train_data = clean_data(train_data)
# test_data = clean_data(test_data)


# ## Create Features
# 

# In[13]:


CREATED_FEATURES = [
    "AmountSpentTotal",
    "CabinDeck",
    "CabinNumber",
    "CabinSide",
    "CabinMates",
    "PassengerGroupSize",
]


def create_features(
    data: pd.DataFrame,
    **kwargs,
):

    # Create new features:
    # - AmountSpentTotal: Total money spent in the ship's service
    # - CabinDeck: Deck of the cabin
    # - CabinNumber: Number of the cabin
    # - CabinSide: Side of the cabin
    # - CabinMates: Number of people in the same cabin
    # - PassengerGroupSize: Group Size

    # Get from kwargs the features to return
    selected_features = [
        feature for feature in CREATED_FEATURES if kwargs.get(f"use_{feature}", False)
    ]

    new_data = data.copy()
    data = data.copy()

    # Create new feature: Total money spent in the ship's service
    new_data["AmountSpentTotal"] = data[
        ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    ].sum(axis=1, skipna=True)

    # Create new feature: Mean money spent in the ship's service
    # TODO is the same as the other one
    # new_data["AmountSpentMean"] = data[
    #     ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    # ].mean(axis=1, skipna=True)

    # Create new features: Convert Cabin to three different columns (Deck, Number, Side)
    new_data[["CabinDeck", "CabinNumber", "CabinSide"]] = data["Cabin"].str.split(
        "/", expand=True
    )

    # Create new feature: Number of people in the same cabin
    new_data["CabinMates"] = data.groupby("Cabin")["Cabin"].transform("count")

    # Create new features: Group Id, Group Size, Intra Group Id,
    new_data[["PassengerGroupId", "PassengerIntraGroupId"]] = data[ID_COLUMN].str.split(
        "_", expand=True
    )
    new_data["PassengerGroupSize"] = new_data.groupby("PassengerGroupId")[
        "PassengerGroupId"
    ].transform("count")

    # Only return the old features and the selected new ones
    return pd.concat([data, new_data[selected_features]], axis=1)


# In[14]:


# train_data = create_features(train_data)
# test_data = create_features(test_data)


# In[15]:


pipeline = Pipeline(
    [
        ("create_features", FunctionTransformer(create_features)),
        ("clean_data", FunctionTransformer(clean_data)),
    ]
)


# In[ ]:


train_data_transformed_df = pipeline.fit_transform(train_data)
print(train_data_transformed_df.dtypes)


# ## Data Preprocessing Pipeline
# 
# ### Handle Missing Values
# 
# - Input Data
# - Mark as "Missing"
# 
# ### Data Preprocessing
# 
# - Make Categorical Columns Numerical
#   - One-Hot encoding
#   - Ordinal encoding
# - Scale Numerical Columns
# 

# In[17]:


MAX_CARDINALITY = 4


def select_high_cardinality_categorical_features(df: pd.DataFrame):
    hi_c_cat = df.select_dtypes(include=["object"]).nunique() > MAX_CARDINALITY
    features = hi_c_cat[hi_c_cat].index.tolist()
    return features


def select_low_cardinality_categorical_features(df: pd.DataFrame):
    lo_c_cat = df.select_dtypes(include=["object"]).nunique() <= MAX_CARDINALITY
    features = lo_c_cat[lo_c_cat].index.tolist()
    return features


def select_numerical_features(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


# Combine handling missing values and preprocessing into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat_low_cardinality",
            Pipeline(
                steps=[
                    (
                        "impute",
                        # SimpleImputer(strategy="most_frequent"),
                        SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),
                    ),
                    (
                        "to_num",
                        OneHotEncoder(),
                        # OrdinalEncoder(),
                        # LabelEncoder(),
                    ),
                ]
            ),
            select_low_cardinality_categorical_features,  # make_column_selector(dtype_include='object'),
        ),
        (
            "cat_high_cardinality",
            Pipeline(
                steps=[
                    (
                        "impute",
                        SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),
                        # SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "to_num",
                        # OneHotEncoder(),
                        OrdinalEncoder(),
                        # LabelEncoder(),
                    ),
                ]
            ),
            select_high_cardinality_categorical_features,  # make_column_selector(dtype_include='object'),
        ),
        (
            "num",
            Pipeline(
                steps=[
                    (
                        "impute",
                        # KNNImputer(n_neighbors=1),
                        # KNNImputer(n_neighbors=3),
                        KNNImputer(n_neighbors=5),
                        # SimpleImputer(strategy="mean"),
                        # SimpleImputer(strategy="median"),
                    ),
                    (
                        "scale",
                        StandardScaler(),
                        # MinMaxScaler(),
                        # RobustScaler(),
                    ),
                ]
            ),
            select_numerical_features,  # make_column_selector(dtype_include='number'),
        ),
    ],
    remainder="passthrough",
    # sparse_threshold=0,
)

# preprocessor.set_output(transform="pandas")


# In[18]:


pipeline = Pipeline(
    steps=[
        ("create_features", FunctionTransformer(create_features)),
        ("clean_data", FunctionTransformer(clean_data)),
        ("preprocessor", preprocessor),
    ]
)


# In[19]:


def transform_data(data: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    # Fit and transform the data using the pipeline
    data_transformed = pipeline.fit_transform(X=X, y=y)

    # Extract feature names from the preprocessor step
    if "preprocessor" in pipeline.named_steps:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    else:
        feature_names = data.columns

    # Extract the selected feature indices from the feature engineering step
    if "feature_engineering" in pipeline.named_steps:
        feature_selector = pipeline.named_steps["feature_engineering"].named_steps[
            "feature_selection"
        ]
        if isinstance(feature_selector, SelectFromModel):
            support_mask = feature_selector.get_support()
        elif isinstance(feature_selector, RFE):
            support_mask = feature_selector.support_
        else:
            support_mask = np.ones(len(feature_names), dtype=bool)
        selected_feature_names = [
            name for name, selected in zip(feature_names, support_mask) if selected
        ]
    else:
        selected_feature_names = feature_names

    # Remove prefixes from the column names
    selected_feature_names = [name.split("__")[-1] for name in selected_feature_names]

    # Convert the transformed data back to a DataFrame
    data_transformed_df = pd.DataFrame(data_transformed, columns=selected_feature_names)

    data_transformed_df[TARGET_COLUMN] = y.values

    return data_transformed_df


# In[20]:


# Use the function to transform the train_data
train_data_transformed_df = transform_data(train_data, pipeline)


# ### Check Preprocessing Works
# 
# - Check if no missing values after preprocessing
# - Check if all columns are numerical after preprocessing
# 

# In[ ]:


# Check for missing values

print("Number of missing values in transformed data:")
print(pd.DataFrame(train_data_transformed_df.isna().sum()).T)

assert train_data_transformed_df.isna().sum().sum() == 0


# In[ ]:


# Check all columns are numerical

all_columns_numerical = train_data_transformed_df.select_dtypes(
    include=[np.number]
).columns.tolist()
all_columns = train_data_transformed_df.columns.tolist()

columns_not_numerical = (
    set(all_columns) - set(all_columns_numerical) - set([TARGET_COLUMN])
)
print(f"Columns not numerical: {columns_not_numerical}")

# Output the types of the non-numerical columns
for col in columns_not_numerical:
    print(
        f"Column: {col}, Type: {train_data_transformed_df[col].dtype}, First Value: {train_data_transformed_df[col].iloc[0]}"
    )

assert columns_not_numerical == set()


# ## Feature Engineering
# 

# In[23]:


feature_engineering = Pipeline(
    steps=[
        # ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
        # ("feature_selection", RFE(estimator=RandomForestClassifier(random_state=RANDOM_SEED))),
        # ("feature_selection", RFE(estimator=RandomForestClassifier(random_state=RANDOM_SEED), n_features_to_select=10)),
        (
            "feature_selection",
            SelectFromModel(LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=10000)),
        ),
        # ("feature_selection", SelectKBest(f_classif, k=10)),
        # ("feature_selection", SelectKBest(mutual_info_classif, k=10)),
        # ("feature_selection", SelectKBest(f_classif, k=20)),
        # ("feature_selection", SelectKBest(mutual_info_classif, k=20)),
        # ("feature_selection", SelectFromModel(RandomForestClassifier(random_state=RANDOM_SEED), threshold="mean")),
    ]
)


# In[24]:


# Add the feature engineering pipeline to the main pipeline
pipeline = Pipeline(
    steps=[
        ("create_features", FunctionTransformer(create_features)),
        ("clean_data", FunctionTransformer(clean_data)),
        ("preprocessor", preprocessor),
        ("feature_engineering", feature_engineering),
    ]
)


# In[25]:


# Use the function to transform the train_data
train_data_transformed_df = transform_data(train_data, pipeline)


# In[ ]:


train_data_transformed_df.columns


# ## Analyze Correlation on Transformed Dataset
# 

# In[ ]:


corr_matrix = train_data_transformed_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    xticklabels=train_data_transformed_df.columns.tolist(),
    yticklabels=train_data_transformed_df.columns.tolist(),
)
plt.title("Correlation Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
# plt.show()


# In[ ]:


# Filter the correlation matrix to only include the Target Column
target_corr_matrix = corr_matrix[[TARGET_COLUMN]].sort_values(
    by=TARGET_COLUMN, ascending=False
)

plt.figure(figsize=(8, 12))
sns.heatmap(target_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title(f"Correlation with {TARGET_COLUMN}")
# plt.show()


# ## Tuning Grids
# 

# In[29]:


# Main pipeline

pipeline = Pipeline(
    steps=[
        ("create_features", FunctionTransformer(create_features)),
        ("clean_data", FunctionTransformer(clean_data)),
        ("preprocessor", preprocessor),
        ("feature_engineering", feature_engineering),
        ("classifier", "passthrough"),
    ]
)

# Note: "passthrough" is used as a placeholder for the model to be used


# ### Preprocessor Grids
# 
# 2 _ 2 _ 5 = 20
# Fitting 5 folds for each of 120 candidates, totalling 600 fits
# 11 min 51 s
# 

# In[30]:


# preprocessor_grid = {
#     "preprocessor__cat_low_cardinality__impute": [
#         # SimpleImputer(strategy="most_frequent"), #
#         SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),
#     ],
#     "preprocessor__cat_low_cardinality__to_num": [
#         OneHotEncoder(),
#         OrdinalEncoder(), #
#     ],
#     "preprocessor__cat_high_cardinality__impute": [
#         # SimpleImputer(strategy="most_frequent"), #
#         SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),
#     ],
#     "preprocessor__cat_high_cardinality__to_num": [
#         OneHotEncoder(),  # TODO works better?
#         OrdinalEncoder(), #
#     ],
#     "preprocessor__num__impute": [
#         # KNNImputer(n_neighbors=1), #
#         KNNImputer(n_neighbors=3), #
#         KNNImputer(n_neighbors=5),  #
#         SimpleImputer(strategy="mean"), #
#         SimpleImputer(strategy="median"),
#     ],
#     "preprocessor__num__scale": [
#         "passthrough",
#         StandardScaler(),
#         # MinMaxScaler(), #
#         # RobustScaler(), #
#     ],
# }


# In[31]:


preprocessor_grid = {
    "preprocessor__cat__impute": [
        SimpleImputer(strategy="most_frequent"),  #
        SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),
    ],
    "preprocessor__cat__to_num": [
        OneHotEncoder(),
        OrdinalEncoder(),  #
    ],
    "preprocessor__num__impute": [
        # KNNImputer(n_neighbors=1), #
        KNNImputer(n_neighbors=3),  #
        KNNImputer(n_neighbors=5),  #
        SimpleImputer(strategy="mean"),  #
        SimpleImputer(strategy="median"),
    ],
    "preprocessor__num__scale": [
        "passthrough",
        StandardScaler(),
        # MinMaxScaler(), #
        # RobustScaler(), #
    ],
}


# ### Feature Engineering Grid
# 

# In[32]:


LASSO_CV = 5

feature_engineering_grid = {
    "create_features": [
        FunctionTransformer(create_features),
        # "passthrough",
    ],
    "feature_engineering__feature_selection": [
        # RFE(estimator=RandomForestClassifier(random_state=RANDOM_SEED)),
        # RFE(estimator=RandomForestClassifier(random_state=RANDOM_SEED), n_features_to_select=10),
        SelectFromModel(LassoCV(cv=LASSO_CV, random_state=RANDOM_SEED)),
        # SelectKBest(f_classif, k=10),
        # SelectKBest(mutual_info_classif, k=10),
        # SelectKBest(f_classif, k=20),
        # SelectKBest(mutual_info_classif, k=20),
        # SelectFromModel(RandomForestClassifier(random_state=RANDOM_SEED), threshold="mean"),
        "passthrough",
    ],
}


# ### Model Grid
# 
# 6 + 3 + 6 + 6 + 8 = 29
# Fitting 5 folds for each of 29 candidates, totalling 145 fits
# 3 min 45s
# 

# In[33]:


model_grids = [
    # {
    #     # Logistic Regression
    #     "classifier": [LogisticRegression()],
    #     "classifier__C": [0.01, 0.1, 1, 10, 100],
    #     "classifier__penalty": ["l1", "l2"],
    #     "classifier__solver": ["liblinear", "saga"],
    # },
    # {
    #     # Decision Tree
    #     "classifier": [DecisionTreeClassifier(random_state=RANDOM_SEED)],
    #     "classifier__max_depth": [None, 10, 20, 30],
    #     "classifier__min_samples_split": [2, 5, 10],
    #     "classifier__min_samples_leaf": [1, 2, 4],
    # },
    # {
    #     # Random Forest
    #     "classifier": [RandomForestClassifier(random_state=RANDOM_SEED)],
    #     "classifier__n_estimators": [100, 200, 300],
    #     "classifier__max_depth": [None, 10, 20, 30],
    #     "classifier__min_samples_split": [2, 5, 10],
    #     "classifier__min_samples_leaf": [1, 2, 4],
    # },
    # {
    #     # K-Nearest Neighbors
    #     "classifier": [KNeighborsClassifier()],
    #     "classifier__n_neighbors": [3, 5, 7, 9, 11],
    #     "classifier__weights": ["uniform", "distance"],
    #     "classifier__metric": ["euclidean", "manhattan"],
    # },
    # {
    #     # Support Vector Machine
    #     "classifier": [SVC(probability=True)],
    #     "classifier__C": [0.01, 0.1, 1, 10],
    #     "classifier__kernel": ["linear", "rbf", "poly"],
    #     "classifier__gamma": ["scale", "auto"],
    # },
    # {
    #     # Gradient Boosting
    #     "classifier": [GradientBoostingClassifier(random_state=RANDOM_SEED)],
    #     "classifier__n_estimators": [100, 150, 200, 250, 300, 500],
    #     "classifier__learning_rate": [0.05, 0.1, 0.15, 0.2],
    #     "classifier__max_depth": [3, 5, 7],
    #     "classifier__subsample": [0.8, 1.0],
    # },
    # {
    #     # XGBoost
    #     "classifier": [XGBClassifier(random_state=RANDOM_SEED)],
    #     "classifier__n_estimators": [100, 250, 500],
    #     "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    #     "classifier__max_depth": [3, 6, 9],
    #     "classifier__subsample": [0.8, 1.0],
    #     "classifier__colsample_bytree": [0.8, 1.0],
    # },
    # {
    #     # LightGBM
    #     "classifier": [LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)],
    #     "classifier__n_estimators": [100, 250, 500],
    #     "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    #     "classifier__max_depth": [3, 6, 9],
    #     "classifier__subsample": [0.8, 1.0],
    #     "classifier__colsample_bytree": [0.8, 1.0],
    # },
]


# In[34]:


model_grids = [
    {
        # Gradient Boosting
        "classifier": [GradientBoostingClassifier(random_state=RANDOM_SEED)],
        "classifier__n_estimators": [200, 250, 300],
        "classifier__learning_rate": [0.05, 0.1, 0.15],
        "classifier__max_depth": [1, 3, 5],
        "classifier__subsample": [0.8, 1.0],
    },
]


# ### Final Grid Search
# 

# In[35]:


parameter_grids = []

for m in model_grids:
    grid = m
    grid.update(preprocessor_grid)
    grid.update(feature_engineering_grid)
    parameter_grids.append(grid)


# ## Model Training and Parameter Grid Search
# 

# In[36]:


# # Split the train data into training and validation sets
# X = train_data.drop(columns=[TARGET_COLUMN])
# y = train_data[TARGET_COLUMN]

# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED
# )


# In[37]:


# Split the train data into training and validation sets
X_train = train_data.drop(columns=[TARGET_COLUMN])
y_train = train_data[TARGET_COLUMN]


# In[38]:


# # Run experiments
# grid_search = GridSearchCV(
#     estimator=pipeline,
#     param_grid=parameter_grids,
#     cv=5,  # TODO parametrize
#     scoring="accuracy",
#     verbose=1,
# )

# grid_search.fit(X_train, y_train)


# In[39]:


pipeline = Pipeline(
    steps=[
        ("create_features", FunctionTransformer(create_features)),
        ("clean_data", FunctionTransformer(clean_data)),
        ("preprocessor", preprocessor),
        ("feature_engineering", feature_engineering),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_SEED)),
    ]
)


# In[40]:


classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=10000, random_state=RANDOM_SEED),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_SEED),
    "KNeighbors": KNeighborsClassifier(),
    "SVC": SVC(max_iter=10000, random_state=RANDOM_SEED, probability=True),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    "XGBoost": XGBClassifier(random_state=RANDOM_SEED),
    "LightGBM": LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
}


# In[41]:


transformers = {
    "constant": SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),
    "most_frequent": SimpleImputer(strategy="most_frequent"),
    "onehot": OneHotEncoder(),
    "ordinal": OrdinalEncoder(),
    "knn_3": KNNImputer(n_neighbors=3),
    "knn_5": KNNImputer(n_neighbors=5),
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler(),
    "create_features": FunctionTransformer(create_features),
    "lasso": SelectFromModel(LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=10000)),
    "passthrough": "passthrough",
}


# In[42]:


# def objective(trial):
#     # Define the hyperparameters to tune
#     params = {
#         "classifier__n_estimators": trial.suggest_int("classifier__n_estimators", 200, 300),
#         "classifier__learning_rate": trial.suggest_float("classifier__learning_rate", 0.05, 0.15),
#         "classifier__max_depth": trial.suggest_int("classifier__max_depth", 1, 5),
#         "classifier__subsample": trial.suggest_float("classifier__subsample", 0.8, 1.0),
#         "preprocessor__cat_low_cardinality__impute": transformers[trial.suggest_categorical(
#             "preprocessor__cat_low_cardinality__impute", ["constant", "most_frequent"]
#         )],
#         "preprocessor__cat_low_cardinality__to_num": transformers[trial.suggest_categorical(
#             "preprocessor__cat_low_cardinality__to_num", ["onehot", "ordinal"]
#         )],
#         "preprocessor__cat_high_cardinality__impute": transformers[trial.suggest_categorical(
#             "preprocessor__cat_high_cardinality__impute", ["constant", "most_frequent"]
#         )],
#         "preprocessor__cat_high_cardinality__to_num": transformers[trial.suggest_categorical(
#             "preprocessor__cat_high_cardinality__to_num", ["onehot", "ordinal"]
#         )],
#         "preprocessor__num__impute": transformers[trial.suggest_categorical(
#             "preprocessor__num__impute", ["knn_3", "knn_5", "mean", "median"]
#         )],
#         "preprocessor__num__scale": transformers[trial.suggest_categorical(
#             "preprocessor__num__scale", ["standard", "minmax", "robust", "passthrough"]
#         )],
#         "feature_engineering__feature_selection": transformers[trial.suggest_categorical(
#             "feature_engineering__feature_selection", ["lasso", "passthrough"]
#         )],
#     }

#     # Update the pipeline with the suggested hyperparameters
#     pipeline.set_params(**params)

#     # Perform cross-validation
#     scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
#     return scores.mean()


# In[43]:


# import signal

# class TimeoutException(Exception):
#     pass

# def timeout_handler(signum, frame):
#     raise TimeoutException

# def objective(trial):
#     # Set the timeout handler
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(120)  # Set the timeout to 2 minutes (120 seconds)

#     try:
#         # Define the models to tune
#         model_name = trial.suggest_categorical("classifier", ["LogisticRegression", "RandomForest", "GradientBoosting", "SVC", "KNeighbors"])

#         if model_name == "LogisticRegression":
#             classifier = LogisticRegression(
#                 C=trial.suggest_float("classifier__C", 0.01, 10.0),
#                 penalty=trial.suggest_categorical("classifier__penalty", ["l1", "l2"]),
#                 solver=trial.suggest_categorical("classifier__solver", ["liblinear", "saga"]),
#                 random_state=RANDOM_SEED
#             )
#         elif model_name == "RandomForest":
#             classifier = RandomForestClassifier(
#                 n_estimators=trial.suggest_int("classifier__n_estimators", 100, 300),
#                 max_depth=trial.suggest_int("classifier__max_depth", 1, 10),
#                 min_samples_split=trial.suggest_int("classifier__min_samples_split", 2, 10),
#                 min_samples_leaf=trial.suggest_int("classifier__min_samples_leaf", 1, 4),
#                 random_state=RANDOM_SEED
#             )
#         elif model_name == "KNeighbors":
#             classifier = KNeighborsClassifier(
#                 n_neighbors=trial.suggest_int("classifier__n_neighbors", 3, 11),
#                 weights=trial.suggest_categorical("classifier__weights", ["uniform", "distance"]),
#                 metric=trial.suggest_categorical("classifier__metric", ["euclidean", "manhattan"])
#             )
#         elif model_name == "SVC":
#             classifier = SVC(
#                 C=trial.suggest_float("classifier__C", 0.01, 10.0),
#                 kernel=trial.suggest_categorical("classifier__kernel", ["linear", "rbf", "poly"]),
#                 probability=True,
#                 max_iter=1000,
#                 random_state=RANDOM_SEED
#             )
#         elif model_name == "GradientBoosting":
#             classifier = GradientBoostingClassifier(
#                 n_estimators=trial.suggest_int("classifier__n_estimators", 100, 300),
#                 learning_rate=trial.suggest_float("classifier__learning_rate", 0.01, 0.2),
#                 max_depth=trial.suggest_int("classifier__max_depth", 1, 10),
#                 subsample=trial.suggest_float("classifier__subsample", 0.8, 1.0),
#                 random_state=RANDOM_SEED
#             )

#         # Define the hyperparameters for the preprocessor and feature engineering
#         params = {
#             "preprocessor__cat__impute": transformers[trial.suggest_categorical(
#                 "preprocessor__cat__impute", ["constant", "most_frequent"]
#             )],
#             "preprocessor__cat__to_num": transformers[trial.suggest_categorical(
#                 "preprocessor__cat__to_num", ["onehot", "ordinal"]
#             )],
#             "preprocessor__num__impute": transformers[trial.suggest_categorical(
#                 "preprocessor__num__impute", ["knn_3", "knn_5", "mean", "median"]
#             )],
#             "preprocessor__num__scale": transformers[trial.suggest_categorical(
#                 "preprocessor__num__scale", ["standard", "passthrough"]
#                 # "preprocessor__num__scale", ["standard", "minmax", "robust", "passthrough"]
#             )],
#             "create_features": transformers[trial.suggest_categorical(
#                 "create_features", ["create_features", "passthrough"]
#             )],
#             "feature_engineering__feature_selection": transformers[trial.suggest_categorical(
#                 "feature_engineering__feature_selection", ["lasso", "passthrough"]
#             )],
#         }

#         # Update the pipeline with the suggested hyperparameters
#         pipeline.set_params(classifier=classifier, **params)

#         # Print the parameters for the current trial
#         print(f"Trial {trial.number}: Starting")
#         for key, value in trial.params.items():
#             print(f"  {key}: {value}")

#         # Perform cross-validation
#         scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

#         # Cancel the alarm
#         signal.alarm(0)

#         return scores.mean()

#     except TimeoutException:
#         print(f"Trial {trial.number}: Timeout")
#         return 0.0  # Return a bad accuracy score if the trial times out


# In[44]:


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def objective(trial):
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)  # Set the timeout to 2 minutes (120 seconds)

    try:
        # Define the hyperparameters for the preprocessor and feature engineering
        params = {
            "classifier": classifiers[
                trial.suggest_categorical(
                    "classifier",
                    [
                        "LogisticRegression",
                        "RandomForest",
                        "GradientBoosting",
                        "SVC",
                        "KNeighbors",
                        "XGBoost",
                        # "LightGBM"
                    ],
                )
            ],
            "preprocessor__cat_low_cardinality__impute": transformers[
                trial.suggest_categorical(
                    "preprocessor__cat_low_cardinality__impute",
                    ["constant", "most_frequent"],
                )
            ],
            "preprocessor__cat_low_cardinality__to_num": transformers[
                trial.suggest_categorical(
                    "preprocessor__cat_low_cardinality__to_num", ["onehot", "ordinal"]
                )
            ],
            "preprocessor__cat_high_cardinality__impute": transformers[
                trial.suggest_categorical(
                    "preprocessor__cat_high_cardinality__impute",
                    ["constant", "most_frequent"],
                )
            ],
            "preprocessor__cat_high_cardinality__to_num": transformers[
                trial.suggest_categorical(
                    "preprocessor__cat_high_cardinality__to_num", ["onehot", "ordinal"]
                )
            ],
            "preprocessor__num__impute": transformers[
                trial.suggest_categorical(
                    "preprocessor__num__impute", ["knn_3", "knn_5", "mean", "median"]
                )
            ],
            "preprocessor__num__scale": transformers[
                trial.suggest_categorical(
                    "preprocessor__num__scale",
                    ["standard", "passthrough"],
                )
            ],
            # TODO get selected features selection here subset of selected features list
            "create_features": transformers[
                trial.suggest_categorical(
                    "create_features", ["create_features", "passthrough"]
                )
            ],
            "feature_engineering__feature_selection": transformers[
                trial.suggest_categorical(
                    "feature_engineering__feature_selection", ["lasso", "passthrough"]
                )
            ],
        }
        if params["create_features"] != "passthrough":
            params[f"create_features__kw_args"] = {
                f"use_{feature}": trial.suggest_categorical(
                    f"create_features__kw_args__use_{feature}", [False, True]
                )
                for feature in CREATED_FEATURES
            }

        # Update the pipeline with the suggested hyperparameters
        print(pipeline)
        print(params)
        pipeline.set_params(**params)

        # Print the parameters for the current trial
        print(f"Trial {trial.number}: Starting")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")

        # Perform cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

        # Cancel the alarm
        signal.alarm(0)

        return scores.mean()

    except TimeoutException:
        print(f"Trial {trial.number}: Timeout")
        return 0.0  # Return a bad accuracy score if the trial times out


# In[ ]:


# Set logging level to INFO
optuna.logging.set_verbosity(optuna.logging.DEBUG)

N_TRIALS_PIPELINE = 1
# N_TRIALS_PIPELINE = 100

# Create a study and optimize the objective function
study_pipeline = optuna.create_study(
    direction="maximize", sampler=TPESampler(seed=RANDOM_SEED)
)
study_pipeline.optimize(objective, n_trials=N_TRIALS_PIPELINE)


# In[ ]:


def print_model_parameters(params):
    for k, v in params.items():
        print(f"  {k:<50}: {v}")


# In[ ]:


# Show best pipeline
print("Best pipeline:")
print_model_parameters(study_pipeline.best_params)


# In[103]:


# Define the objective function for Optuna
def objective(trial, pipeline, classifier_name):
    classifier = classifiers[classifier_name]

    if classifier_name == "LogisticRegression":
        classifier.set_params(
            C=trial.suggest_float("classifier__C", 0.01, 10.0),
            penalty=trial.suggest_categorical("classifier__penalty", ["l1", "l2"]),
            solver=trial.suggest_categorical(
                "classifier__solver", ["liblinear", "saga"]
            ),
        )
    elif classifier_name == "RandomForest":
        classifier.set_params(
            n_estimators=trial.suggest_int("classifier__n_estimators", 100, 300),
            max_depth=trial.suggest_int("classifier__max_depth", 1, 10),
            min_samples_split=trial.suggest_int("classifier__min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("classifier__min_samples_leaf", 1, 4),
        )
    elif classifier_name == "KNeighbors":
        classifier.set_params(
            n_neighbors=trial.suggest_int("classifier__n_neighbors", 3, 11),
            weights=trial.suggest_categorical(
                "classifier__weights", ["uniform", "distance"]
            ),
            metric=trial.suggest_categorical(
                "classifier__metric", ["euclidean", "manhattan"]
            ),
        )
    elif classifier_name == "SVC":
        classifier.set_params(
            C=trial.suggest_float("classifier__C", 0.01, 10.0),
            kernel=trial.suggest_categorical(
                "classifier__kernel", ["linear", "rbf", "poly"]
            ),
            probability=True,
            max_iter=1000,
        )
    elif classifier_name == "GradientBoosting":
        classifier.set_params(
            n_estimators=trial.suggest_int("classifier__n_estimators", 100, 300),
            learning_rate=trial.suggest_float("classifier__learning_rate", 0.01, 0.2),
            max_depth=trial.suggest_int("classifier__max_depth", 1, 10),
            subsample=trial.suggest_float("classifier__subsample", 0.8, 1.0),
        )
    elif classifier_name == "XGBoost":
        classifier.set_params(
            n_estimators=trial.suggest_int("classifier__n_estimators", 100, 300),
            learning_rate=trial.suggest_float("classifier__learning_rate", 0.01, 0.2),
            max_depth=trial.suggest_int("classifier__max_depth", 1, 10),
            subsample=trial.suggest_float("classifier__subsample", 0.8, 1.0),
            colsample_bytree=trial.suggest_float(
                "classifier__colsample_bytree", 0.8, 1.0
            ),
        )
    elif classifier_name == "LightGBM":
        classifier.set_params(
            n_estimators=trial.suggest_int("classifier__n_estimators", 100, 300),
            learning_rate=trial.suggest_float("classifier__learning_rate", 0.01, 0.2),
            max_depth=trial.suggest_int("classifier__max_depth", 1, 10),
            subsample=trial.suggest_float("classifier__subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(
                "classifier__colsample_bytree", 0.8, 1.0
            ),
        )

    # Update the pipeline with the classifier
    pipeline.set_params(classifier=classifier)

    # Perform cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()


# In[104]:


def map_and_set_params(pipeline, study_params):
    mapped_params = {
        key: (
            transformers[value]
            if value in transformers
            else classifiers[value] if key == "classifier" else value
        )
        for key, value in study_params.items()
    }
    pipeline.set_params(**mapped_params)
    return pipeline


# In[ ]:


N_TRIALS_HYPERPARAMETERS = 1
# N_TRIALS_HYPERPARAMETERS = 100

# Get the best classifier name from the previous study
best_classifier_name = study_pipeline.best_params["classifier"]
best_pipeline = map_and_set_params(pipeline, study_pipeline.best_params)

# Create a study and optimize the objective function
study_hyperparameters = optuna.create_study(
    direction="maximize", sampler=TPESampler(seed=RANDOM_SEED)
)
study_hyperparameters.optimize(
    lambda trial: objective(trial, pipeline, best_classifier_name),
    n_trials=N_TRIALS_HYPERPARAMETERS,
)

# Print the best hyperparameters
print("Best hyperparameters from second study:")
for key, value in study_hyperparameters.best_params.items():
    print(f"{key}: {value}")


# #### Best so far
# 
# ```
# Best Model:
# classifier                                        : GradientBoostingClassifier(random_state=42)
# classifier__learning_rate                         : 0.1
# classifier__n_estimators                          : 250
# preprocessor__cat_onehot__impute                  : SimpleImputer(fill_value='Missing', strategy='constant')
# preprocessor__cat_onehot__onehot                  : OneHotEncoder()
# preprocessor__cat_ordinal__impute                 : SimpleImputer(fill_value='Missing', strategy='constant')
# preprocessor__cat_ordinal__ordinal                : OneHotEncoder()
# preprocessor__num__impute                         : SimpleImputer(strategy='median')
# preprocessor__num__scale                          : StandardScaler()
# ```
# 

# #### Best model for current execution
# 

# In[ ]:


# grid_search.best_estimator_
best_params = study_pipeline.best_params | study_hyperparameters.best_params
print("Best Model:")
print_model_parameters(best_params)


# #### All models for current execution
# 

# In[107]:


# # Assuming grid_search is your GridSearchCV object
# all_estimators_with_scores = list(
#     zip(grid_search.cv_results_["params"], grid_search.cv_results_["mean_test_score"])
# )

# # Sort the estimators by their scores in descending order
# all_estimators_with_scores.sort(key=lambda x: x[1], reverse=True)

# # Print all estimators with their scores and ranking
# for rank, (estimator, score) in enumerate(all_estimators_with_scores, start=1):
#     print(f"Rank: {rank}")
#     print(f"Accuracy: {score}")
#     print("Model:")
#     print_model_parameters(estimator)
#     print("\n")


# In[108]:


# Save the best (in train) model parameters to a JSON file
best_params_file = f"{DATA_DIR}/best_params_train.json"

# Convert non-serializable objects to their string representations
serializable_best_params = {k: str(v) for k, v in best_params.items()}

with open(best_params_file, "w") as f:
    json.dump(serializable_best_params, f, indent=4)


# ## Best Model Evaluation with Validation Set
# 

# In[109]:


def evaluate_model(pipeline, estimator, X_val, y_val):
    y_pred = pipeline.predict(X_val)
    y_pred_proba = (
        pipeline.predict_proba(X_val)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else None

    print(f"Accuracy: {accuracy}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc}")
    # print(classification_report(y_val, y_pred))
    print("Model:")
    print_model_parameters(estimator)

    return accuracy


# In[110]:


# # Evaluate all estimators in grid search with validation set
# for estimator, _ in all_estimators_with_scores:
#     pipeline.set_params(**estimator)
#     pipeline.fit(X_train, y_train)
#     score = evaluate_model(pipeline, estimator, X_val, y_val)


# ## Final Model Training and Submission
# 

# In[ ]:


best_pipeline = map_and_set_params(pipeline, best_params)
best_pipeline.fit(X_train, y_train)


# In[112]:


# Make predictions on the test data
X_test = test_data
y_pred = best_pipeline.predict(X_test)
test_data[TARGET_COLUMN] = y_pred.astype(bool)

# Make predictions on the test set with the best model
# best_model = max(best_models.items(), key=lambda x: cross_val_score(x[1], X_train, y_train, cv=5).mean())[1]
# test_predictions = best_model.predict(test_data)
# test_data[TARGET_COLUMN] = test_predictions.astype(bool)


# In[ ]:


# Create a DataFrame with only the ID_COLUMN and Predictions
predictions_df = test_data.reset_index()[[ID_COLUMN, TARGET_COLUMN]]

# Print predictions
print(predictions_df)

# Save predictions to a CSV file
predictions_df.to_csv(f"{DATA_DIR}/predictions.csv", index=False)


# ## TODO
# 
# - Exploratory Data Analysis:
#   - Data visualization
#   - Find missing values
#   - Find outliers, extreme or unusual values
#   - Correlation analysis between numerical attributes and the target variable
#   - For better data fitting, a more detailed analysis of each categorical variable is necessary.
# - Data Preparation:
#   - Data cleaning
#   - Handle: outliers, extreme or unusual values
#   - Missing values (removal, transformation, imputation, etc.)
#   - Transform categorical variables into numerical (e.g., one-hot encoding)
#   - (?) Transform numerical variables into categorical (e.g., discretization)
#   - Feature engineering:
#     - Create new attributes from existing ones to improve data description and reduce dimensionality.
#     - It may also be interesting to create new attributes based on the interaction of highly correlated variables.
#   - Feature selection
#   - Instance selection
# - Model Training:
#   - Cross-validation
#   - Grid search
# - Model Evaluation:
#   - Evaluation metrics
#   - Model comparison
# - Results Presentation:
#   - Results visualization
#   - Results interpretation
# 
