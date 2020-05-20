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

from sklearn.feature_selection import SelectFromModel  # Select features using Embedded method
from sklearn.linear_model import LassoCV  # Select features using Embedded method


def load_data():
    """Return housing data loaded from file"""
    return pd.read_csv(f'{os.getcwd()}/data/housing.data', sep='\s+', header=None)


def prepare_data(housing_dat):
    """Adds column names to housing dataframe and outputs null value sums."""
    housing_dat.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    print(f'\nNull values for data are:\n{housing_dat.isnull().sum()}\n')
    return housing_dat


def save_plot(workdir, filename, create_workdir=False):
    """Accepts workdir specified as argument to main script and filename
    - saves plot to disk."""
    if create_workdir:
        if workdir == os.getcwd():
            os.makedirs(f'{workdir}/figures', exist_ok=True)
            plt.savefig(f'{workdir}/figures/{filename}', format='png')
        else:
            os.makedirs(f'{workdir}', exist_ok=True)
            plt.savefig(f'{workdir}/{filename}', format='png')
    else:
        if workdir == os.getcwd():
            plt.savefig(f'{workdir}/figures/{filename}', format='png')
        else:
            plt.savefig(f'{workdir}/{filename}', format='png')


def scale_data(x):
    """Accepts features data, computes and applies scaling transformation.
    Returns scaled features"""
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler.transform(x)


def compute_rmse(X, y, rmse_matrix, col_select):
    """Accepts feature and target data, RMSE numpy matrix and matrix column number to store RMSE values.
    Trains all models and computes RMSE values from predictions."""
    # Scale data
    X = scale_data(X)

    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35,
                                                        shuffle=True, random_state=1)
    # Train models and place RMSE result in RMSE_values
    rmse_idx = 0
    for Model in [LinearRegression, GradientBoostingRegressor, ElasticNet, KNeighborsRegressor, Lasso, SVR]:
        model = Model()
        model.fit(X_train, y_train)
        predicted_values = model.predict(X_test)
        rmse_matrix[rmse_idx, col_select] = np.sqrt(metrics.mean_squared_error(y_test, predicted_values))
        rmse_idx += 1


def main():
    # Parse command-line argument for figure storage path
    parser = ArgumentParser()
    parser.add_argument("figure_directory", nargs='?', default=os.getcwd(),
                        help="Directory to save output images. "
                             "Default: current working directory")
    args = parser.parse_args()

    #####################################################
    ##    Data loading and Preparation                 ##
    #####################################################

    housing_data = load_data()
    housing_data = prepare_data(housing_data)

    rmse_values = np.zeros((6, 3), dtype=float)  # Will store RMSE values for each feature selection
    # method.

    #####################################################
    ##    Feature Selection Method 1: No criteria      ##
    #####################################################
    X = housing_data.drop('MEDV', axis=1)  # Select all features and copy to X
    y = housing_data['MEDV']  # Copy median house values (target) to y

    compute_rmse(X, y, rmse_values, 0)

    #####################################################
    ##    Feature Selection Method 2: Filter Method    ##
    #####################################################

    # Create correlation table for data
    corr_table = pd.DataFrame(data=housing_data.corr(),
                              columns=housing_data.columns)

    # Format correlation table
    mask = np.zeros(corr_table.shape, dtype=bool)
    mask[np.triu_indices(len(mask), 1)] = True

    # Create heatmap of correlation data and save image
    sns.set()
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(corr_table, annot=True, cmap='autumn', fmt='.3g', mask=mask, annot_kws={'size': 9})
    ax.set_xticklabels(housing_data.columns, rotation=45)
    ax.set_yticklabels(housing_data.columns, rotation=45)
    plt.title('Correlation map for Boston Housing Data', pad=25)

    save_plot(args.figure_directory, 'whole_corr_plot.png', create_workdir=True)

    # Determine features which correlate highly with median house value
    corr_y = abs(corr_table["MEDV"])
    high_corr_features = corr_y[(corr_y > 0.5) & (corr_y != 1)]
    print(f"\nFilter Method:\n\nFeatures highly correlated with MEDV (coefficient > 0.5):\n")
    print(f"{high_corr_features}")

    # Determine correlations between features that have a high corr with MEDV and drop
    print(f"\n\nCorrelation between features that have a high corr with MEDV:\n")
    print(f"{housing_data[['RM', 'LSTAT']].corr()}\n\n{housing_data[['LSTAT', 'PTRATIO']].corr()}")
    print(f'\nRM will be removed accordingly\n\n')

    # Ignore RM, select LSTAT and PTRATIO features only
    X = housing_data[['LSTAT', 'PTRATIO']]
    y = housing_data['MEDV']

    compute_rmse(X, y, rmse_values, 1)

    #####################################################
    ##    Feature Selection Method 3: Embedded Method  ##
    #####################################################

    X = housing_data.drop('MEDV', axis=1)  # Select all features and copy to X
    y = housing_data['MEDV']  # Copy median house values (target) to y

    columns = X.columns  # Store feature data column names

    # Scale X features
    X = scale_data(X)

    # Determine which features relate most weakly to MEDV prediction
    # using LassoCV regularization
    lsoCV = LassoCV().fit(X, y)
    feat_sel = SelectFromModel(lsoCV, prefit=True)
    X_new = feat_sel.transform(X)

    # Identify removed features and print
    print(f'Embedded method\n\nFrom LassoCV regularization, the feature(s) below were dropped:\n')

    coefs = pd.Series(lsoCV.coef_, index=columns)
    for idx, coeff in enumerate(coefs):
        if coeff == 0:
            print(f'{coefs.index[idx]}\n')

    print(f'Regularization coefficients:\n{coefs}\n')

    # Plot coefficients from LassoCV regularization
    sns.set()
    plt.subplots(figsize=(12, 12))
    sns.barplot(x=columns, y=coefs)
    plt.xticks(rotation=45)
    plt.title('Embedded Method: Coefficient values by housing data feature', pad=25)

    save_plot(args.figure_directory, 'embed_method_coeffs.png')
    plt.show()

    compute_rmse(X_new, y, rmse_values, 2)

    #####################################################
    ##    Final Output: RMSE Dataframe                 ##
    #####################################################

    rmse_df = pd.DataFrame(data=rmse_values, index=['Linear Regression', 'GBoostingRegressor', 'ElasticNet',
                                                    'KNeighborsRegressor', 'Lasso', 'SVR'])
    rmse_df.columns = ['No Criteria', 'Filter Method', 'Embedded Method']

    print(f'All feature selection methods have been applied.\n\nRMSE table:\n\n{rmse_df.to_string()}\n')

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(rmse_df, annot=True, cmap='seismic', fmt='.4g')
    ax.set_xticklabels(rmse_df.columns)
    ax.set_yticklabels(rmse_df.index, rotation=45)
    plt.title('Model Accuracy by Feature Selection Method (using RMSE value)', pad=25)

    save_plot(args.figure_directory, 'RMSE_heatmap.png')


if __name__ == "__main__":
    main()
