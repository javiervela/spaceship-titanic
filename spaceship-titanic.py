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
import time


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


# In[122]:


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


# In[123]:


# Load the data files into pandas dataframes
train_data = pd.read_csv(TRAIN_DATA_FILE)
test_data = pd.read_csv(TEST_DATA_FILE)


# In[124]:


# Make PassengerId the index
train_data.set_index(ID_COLUMN, inplace=True)
test_data.set_index(ID_COLUMN, inplace=True)


# ## Data Exploration
# 

# In[ ]:


print("First few rows of data:")
print(train_data.head())


# In[126]:


print("Data columns and types:")
print(train_data.dtypes)


# In[ ]:


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


# In[128]:


print(f"Numerical columns: {NUMERICAL_COLUMNS}")
print(f"Categorical columns: {CATEGORICAL_COLUMNS}")
print(f"Target column: {TARGET_COLUMN}")


# In[129]:


print("\nSummary statistics:")
print(train_data.describe())

print("\nMissing values:")
print(train_data.isnull().sum())

# print("\nCorrelation matrix:")
# sns.heatmap(train_data[NUMERICAL_COLUMNS].corr(), annot=True)
# plt.show()

print("\nValue counts for categorical variables:")
for col in CATEGORICAL_COLUMNS:
    print(f"\n{col} value counts:")
    print(train_data[col].value_counts())


# In[130]:


train_data


# ## Clean Dataset
# 
# We need to clean the train and test datasets the same way
# 

# In[ ]:


def clean_data(data: pd.DataFrame):

    # Convert Cabin to three different columns (Deck, Number, Side)
    data[["CabinDeck", "CabinNumber", "CabinSide"]] = data["Cabin"].str.split(
        "/", expand=True
    )

    # Convert columns to integer (with missing values)
    for col in ["CabinNumber", "CryoSleep", "VIP", "Transported"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype("Int64")

    # Drop columns
    data.drop(columns=["Name", "Cabin"], inplace=True)

    return data


# In[132]:


train_data = clean_data(train_data)
test_data = clean_data(test_data)


# In[ ]:


def create_features(data: pd.DataFrame):
    data["AmountSpentTotal"] = data[
        ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    ].sum(axis=1, skipna=True)
    data["AmountSpentMean"] = data[
        ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    ].mean(axis=1, skipna=True)
    # TODO add more features ?
    return data


# In[134]:


train_data = create_features(train_data)
test_data = create_features(test_data)


# In[ ]:


handle_missing_values = ColumnTransformer(
    transformers=[
        (
            "cat",
            SimpleImputer(strategy="most_frequent"),
            ["HomePlanet", "Destination", "CabinDeck", "CabinSide"],
        ),
        (
            "num",
            KNNImputer(n_neighbors=5),
            [
                "CryoSleep",
                "Age",
                "VIP",
                "RoomService",
                "FoodCourt",
                "ShoppingMall",
                "Spa",
                "VRDeck",
                "CabinNumber",
                "AmountSpentTotal",
                "AmountSpentMean",
            ],
        ),
    ],
    remainder="passthrough",
    # sparse_threshold=0,
)

# Other numerical inputers
# SimpleImputer(strategy="mean")
# SimpleImputer(strategy="median")
# KNNImputer(n_neighbors=5)

# Other categorical inputers
# SimpleImputer(strategy="most_frequent")
# SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),

# handle_missing_values.set_output(transform="pandas")


# ## Data Preprocessing Joint Pipeline
# 
# ### Handle Missing Values
# 
# - Input Data
# - Mark as "Missing"
# 
# - TODO:
#   - Look for other inputer strategies
# 
# ### Data Preprocessing
# 
# - Make Categorical Columns Numerical
#   - One-Hot encoding
#   - Ordinal encoding
# - Scale Numerical Columns
# 
# - TODO:
#   - Other scaling techniques
# 

# In[ ]:


# Other numerical inputers
# SimpleImputer(strategy="mean")
# SimpleImputer(strategy="median")
# KNNImputer(n_neighbors=5)


# Combine handling missing values and preprocessing into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat_onehot",
            Pipeline(
                steps=[
                    (
                        "impute",
                        # SimpleImputer(strategy="most_frequent"),
                        SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(),
                        # OrdinalEncoder(),
                        # LabelEncoder(),
                    ),
                ]
            ),
            ["HomePlanet", "Destination"],
        ),
        (
            "cat_ordinal",
            Pipeline(
                steps=[
                    (
                        "impute",
                        SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),
                        # SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "ordinal",
                        OneHotEncoder(),
                        # OrdinalEncoder(),
                        # LabelEncoder(),
                    ),
                ]
            ),
            ["CabinDeck", "CabinSide"],
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
            [
                "CryoSleep",
                "Age",
                "VIP",
                "RoomService",
                "FoodCourt",
                "ShoppingMall",
                "Spa",
                "VRDeck",
                "CabinNumber",
                "AmountSpentTotal",
                "AmountSpentMean",
            ],
        ),
    ],
    remainder="passthrough",
    # sparse_threshold=0,
)

# preprocessor.set_output(transform="pandas")


# In[ ]:


# Fit and transform the train_data using the preprocessor
train_data_transformed = preprocessor.fit_transform(train_data)

# Convert the transformed data back to a DataFrame
train_data_transformed_df = pd.DataFrame(
    train_data_transformed, columns=preprocessor.get_feature_names_out()
)

# Rename columns to remove prefixes
new_column_names = [col.split("__")[-1] for col in train_data_transformed_df.columns]
train_data_transformed_df.columns = new_column_names


# ### Check Preprocessing Works
# 
# - Check if no missing values after preprocessing
# - Check if all columns are numerical after preprocessing
# 

# In[138]:


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

columns_not_numerical = set(all_columns) - set(all_columns_numerical)
print(f"Columns not numerical: {columns_not_numerical}")

assert all_columns_numerical == all_columns


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
    xticklabels=all_columns,
    yticklabels=all_columns,
)
plt.title("Correlation Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
# plt.show()


# In[141]:


# Filter the correlation matrix to only include the Target Column
target_corr_matrix = corr_matrix[[TARGET_COLUMN]].sort_values(
    by=TARGET_COLUMN, ascending=False
)

plt.figure(figsize=(4, 6))
sns.heatmap(target_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title(f"Correlation with {TARGET_COLUMN}")
# plt.show()


# ## Tuning Grids
# 

# In[142]:


# Main pipeline

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
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

# In[ ]:


preprocessor_grids = [
    {
        "preprocessor__cat_onehot__impute": [
            # SimpleImputer(strategy="most_frequent"),
            SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),
        ]
    },
    {
        "preprocessor__cat_onehot__onehot": [
            OneHotEncoder(),
            # OrdinalEncoder(),
        ]
    },
    {
        "preprocessor__cat_ordinal__impute": [
            # SimpleImputer(strategy="most_frequent"),
            SimpleImputer(strategy="constant", fill_value=MISSING_VALUE),
        ]
    },
    {
        "preprocessor__cat_ordinal__ordinal": [
            OneHotEncoder(), # TODO works better?
            OrdinalEncoder(),
        ]
    },
    {
        "preprocessor__num__impute": [
            # KNNImputer(n_neighbors=1),
            # KNNImputer(n_neighbors=3),
            # KNNImputer(n_neighbors=5),
            # SimpleImputer(strategy="mean"),
            SimpleImputer(strategy="median"),
        ]
    },
    {
        "preprocessor__num__scale": [
            StandardScaler(),
            # MinMaxScaler(),
            # RobustScaler(),
        ]
    },
]


# ### Model Grid
# 
# 6 + 3 + 6 + 6 + 8 = 29
# Fitting 5 folds for each of 29 candidates, totalling 145 fits
# 3 min 45s
# 

# In[176]:


model_grids = [
    {
        # Random Forest
        "classifier": [RandomForestClassifier(random_state=RANDOM_SEED)],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
    },
    {
        # Logistic Regression
        "classifier": [LogisticRegression()],
        "classifier__C": [0.1, 1, 10],
    },
    {
        # Support Vector Machine
        "classifier": [SVC(probability=True)],
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["linear", "rbf"],
    },
    {
        # Gradient Boosting
        "classifier": [GradientBoostingClassifier(random_state=RANDOM_SEED)],
        "classifier__n_estimators": [100, 250, 500],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
    },
    {
        # K-Nearest Neighbors
        "classifier": [KNeighborsClassifier()],
        "classifier__n_neighbors": [3, 5, 7, 9],
        "classifier__weights": ["uniform", "distance"],
    },
]


# ### Final Grid Search
# 

# In[182]:


parameter_grids = []

for m in model_grids:
    for p in preprocessor_grids:
        grid = m
        grid.update(p)
        parameter_grids.append(grid)


# ## Model Training and Parameter Grid Search
# 

# In[183]:


# Split the train data into training and validation sets
X = train_data.drop(columns=[TARGET_COLUMN])
y = train_data[TARGET_COLUMN]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED
)


# In[ ]:

start_time = time.time()

# Run experiments
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grids,
    cv=5,  # TODO parametrize
    scoring="accuracy",
    verbose=2,
)

grid_search.fit(X_train, y_train)

end_time = time.time()

# Save elapsed time to a file
elapsed_time = end_time - start_time
elapsed_time_file = f"{DATA_DIR}/elapsed_time.txt"
with open(elapsed_time_file, "w") as f:
    f.write(f"Elapsed time: {elapsed_time} seconds")


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

# In[ ]:


# grid_search.best_estimator_
best_params = grid_search.best_params_
print(
    f"Best Model:\n{"\n".join([f"{k:<50}: {v}" for k,v in grid_search.best_params_.items()])}"
)


# In[ ]:


# Save the best model parameters to a file
best_params = grid_search.best_params_
best_params_file = f"{DATA_DIR}/best_params.txt"
with open(best_params_file, "w") as f:
	f.write(str(best_params))


# ## Best Model Evaluation with Validation Set
# 

# In[ ]:


model = grid_search


y_pred = model.predict(X_val)
y_pred_proba = (
    model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
)
accuracy = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else None

print(f"Best Model:\n", grid_search.best_params_)
print(f"Accuracy: {accuracy}")
if roc_auc is not None:
    print(f"ROC AUC: {roc_auc}")
print(classification_report(y_val, y_pred))


# ## Final Model Training and Submission
# 

# In[ ]:


# Retrain the best model on the full training data
best_model = grid_search.best_estimator_
best_model.fit(X, y)


# In[ ]:


# Make predictions on the test data
X_test = test_data
y_pred = best_model.predict(X_test)
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

# ## TODO 2
# 
# 1. Clean data
#    1. [x] Cabin split /
#    2. [x] Change types (boolean to int) -> True = 1, False = 0
#    3. [x] Class variable "Transported" to int
# 2. Missing values
#    - [ ] Remove
#    - [ ] Inpute
#      - Categorical:
#        - [ ] Mode
#        - [ ] Simple
#        - [ ] Constant MISSING_VALUE
#      - Numerical:
#        - [ ] Mean
#        - [ ] Median
#        - [ ] KNN
# 3. Feature engineering
#    - [ ] Create variable for expenses
#    - [ ] Create variable for name bag of words
# 4. Categorical variables to numerical
#    - [ ] One-hot encoding
#    - [ ] Ordinal encoding
# 5. Numerical values scaling
#    - [ ] Standardization
#    - [ ] Normalization
# 
