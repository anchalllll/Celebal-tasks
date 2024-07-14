import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# Function to load datasets
def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Load datasets
df_titanic = load_csv("titanic.csv")
df_application_train = load_csv("application_train.csv")

# Check the shape of the datasets
if df_titanic is not None:
    print(df_titanic.shape)  # (891, 12)
if df_application_train is not None:
    print(df_application_train.shape)  # (307511, 122)

# Function to calculate outlier thresholds
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Function to check for outliers
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Function to grab outliers
def grab_outliers(dataframe, col_name, outlier_index=False, f=5):
    low, up = outlier_thresholds(dataframe, col_name)
    outliers = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]
    if len(outliers) > 10:
        print(outliers.head(f))
    else:
        print(outliers)
    if outlier_index:
        return outliers.index

# Function to remove outliers
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

# Function to replace outliers with threshold values
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Function to grab column names
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

# Example usage for Titanic dataset
if df_titanic is not None:
    sns.boxplot(x=df_titanic["Age"])
    plt.show()

    print(outlier_thresholds(df_titanic, "Age"))

    # Check and remove outliers in Titanic dataset
    cat_cols, num_cols, cat_but_car = grab_col_names(df_titanic)
    num_cols.remove('PassengerId')

    for col in num_cols:
        df_titanic = remove_outlier(df_titanic, col)

    print(df_titanic.shape)

    for col in num_cols:
        replace_with_thresholds(df_titanic, col)

    for col in num_cols:
        print(col, check_outlier(df_titanic, col))

# Example usage for application_train dataset
if df_application_train is not None:
    cat_cols, num_cols, cat_but_car = grab_col_names(df_application_train)
    num_cols.remove('SK_ID_CURR')

    for col in num_cols:
        print(col, check_outlier(df_application_train, col))

    # Use Local Outlier Factor for anomaly detection
    df_diamonds = sns.load_dataset('diamonds')
    df_diamonds = df_diamonds.select_dtypes(include=['float64', 'int64']).dropna()
    print(df_diamonds.shape)

    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df_diamonds)
    df_scores = clf.negative_outlier_factor_

    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 20], style='.-')
    plt.show()

    th = np.sort(df_scores)[3]
    print(th)

    outliers = df_diamonds[df_scores < th]
    print(outliers)

    df_diamonds_cleaned = df_diamonds.drop(index=outliers.index)
    print(df_diamonds_cleaned.shape)
