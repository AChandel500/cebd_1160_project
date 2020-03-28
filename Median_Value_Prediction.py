import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LassoCV  # Select features using Embedded

parser = ArgumentParser()
parser.add_argument("figure_directory", nargs='?', default=os.getcwd(),
                    help="Directory to save output images. "
                         "Default: current working directory")
args = parser.parse_args()

def load_data():
    return pd.read_csv(f'{os.getcwd()}/data/housing.data', sep='\s+', header=None)

def prepare_data():
    housing_data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    print(f'\nNull values for data are:{housing_data.isnull().sum()}\n')


#####################################################
##    Data loading and Preparation                 ##
#####################################################

housing_data = load_data()
prepare_data()

RMSE_values = np.zeros((6, 3), dtype=float)  # Will store RMSE values for each feature selection
                                             # method.

#####################################################
##    Feature Selection Method 1: No criteria      ##
#####################################################
X = housing_data.drop('MEDV', axis=1)  # Select all features and copy to X
y = housing_data['MEDV']  # Copy median house values (target) to y

# Scale data
scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)

# Split data for training
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.35,
                                                    shuffle=True)
# Train models using method 1 and place RMSE result in RMSE_values
RMSE_idx = 0
for Model in [LinearRegression, GradientBoostingRegressor, ElasticNet, KNeighborsRegressor, Lasso, SVR]:
    model = Model()
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    RMSE_values[RMSE_idx, 0] = np.sqrt(metrics.mean_squared_error(y_test, predicted_values))
    RMSE_idx += 1

#####################################################
##    Feature Selection Method 2: Filter Method    ##
#####################################################

# Create correlation table for data
corr_table = pd.DataFrame(data=housing_data.corr(),
                          columns=housing_data.columns)

# Create heatmap of correlation data and save image
sns.set()
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corr_table, annot=True, cmap='autumn')
ax.set_xticklabels(housing_data.columns, rotation=45)
ax.set_yticklabels(housing_data.columns, rotation=45)
plt.title('Correlation map for Boston Housing Data', pad=25)

if (args.figure_directory == os.getcwd()):
    os.makedirs(f'{args.figure_directory}/figures', exist_ok=True)
    plt.savefig(f'{args.figure_directory}/figures/whole_corr_plot.png', format='png')
else:
    os.makedirs(f'{args.figure_directory}', exist_ok=True)
    plt.savefig(f'{args.figure_directory}/whole_corr_plot.png', format='png')


# Determine features which correlate highly with median house value
corr_y = abs(corr_table["MEDV"])
high_corr_features = corr_y[corr_y > 0.5]

# Determine correlations between features that have a high corr with MEDV and drop
print(f"\nFilter Method:\n\nCorrelation between features that have a high corr with MEDV:\n")
print(f"{housing_data[['RM', 'LSTAT']].corr()}\n\n{housing_data[['LSTAT', 'PTRATIO']].corr()}")
print(f'\nRM will be removed accordingly\n\n')

# Ignore RM, select LSTAT and PTRATIO features only
X = housing_data[['LSTAT', 'PTRATIO']]
y = housing_data['MEDV']

# Scale data
scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.35,

                                                    shuffle=True)
# Train models using method 2 and place RMSE result in RMSE_values
RMSE_idx = 0
for Model in [LinearRegression, GradientBoostingRegressor, ElasticNet, KNeighborsRegressor, Lasso, SVR]:
    model = Model()
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    RMSE_values[RMSE_idx, 1] = np.sqrt(metrics.mean_squared_error(y_test, predicted_values))
    RMSE_idx += 1

#####################################################
##    Feature Selection Method 3: Embedded Method  ##
#####################################################

X = housing_data.drop('MEDV', axis=1)  # Select all features and copy to X
y = housing_data['MEDV']  # Copy median house values (target) to y

# Determine which features relate most weakly to MEDV prediction (coefficients --> ~0)
reg = LassoCV()
reg.fit(X, y)
coef = pd.Series(reg.coef_, index=X.columns)

sns.set()
ax = plt.subplots(figsize=(12, 12))
sns.barplot(x=X.columns, y=coef)
plt.xticks(rotation=45)
plt.title('Embedded Method: Coefficient values by housing data feature', pad=25)
if (args.figure_directory == os.getcwd()):

    plt.savefig(f'{args.figure_directory}/figures/embed_method_coeffs.png', format='png')
else:
    plt.savefig(f'{args.figure_directory}/embed_method_coeffs.png', format='png')



# Drop features with zero value coefficients
X = housing_data.drop(['NOX', 'INDUS', 'CHAS'], axis=1)
y = housing_data['MEDV']

# Scale features
scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)

# Split data for training
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.35,
                                                    shuffle=True)
# Train models using method 3 and place RMSE result in RMSE_values
RMSE_idx = 0
for Model in [LinearRegression, GradientBoostingRegressor, ElasticNet, KNeighborsRegressor, Lasso, SVR]:
    model = Model()
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    RMSE_values[RMSE_idx, 2] = np.sqrt(metrics.mean_squared_error(y_test, predicted_values))
    RMSE_idx += 1

RMSE_df = pd.DataFrame(data=RMSE_values, index=['Linear Regression', 'GBoostingRegressor', 'ElasticNet',
                                                'KNeighborsRegressor', 'Lasso', 'SVR'])
RMSE_df.columns = ['No Criteria', 'Filter Method', 'Embedded Method']

print(f'All feature selection methods have been applied.\n\nRMSE table:\n\n{RMSE_df.to_string()}\n')

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(RMSE_df, annot=True, cmap='seismic')
ax.set_xticklabels(RMSE_df.columns)
ax.set_yticklabels(RMSE_df.index, rotation=45)
plt.title('Model Accuracy by Feature Selection Method (using RMSE value)', pad=25)

if (args.figure_directory == os.getcwd()):
    plt.savefig(f'{args.figure_directory}/figures/RMSE_heatmap', format='png')
else:
    plt.savefig(f'{args.figure_directory}/RMSE_heatmap.png', format='png')


