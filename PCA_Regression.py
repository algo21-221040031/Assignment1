import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from DataProcess import PlateConstruction, DataProcessor


def PrincipleComponentConstruction(data, k):
    """

    :param data: pandas DataFrame, the processed return data of plates
    :param k: top k eigenvalues
    :return: ComponentMatrix
    """

    covMatrix = data.cov()
    eVals, eVecs = np.linalg.eig(covMatrix)
    sortedIndices = np.argsort(eVals)
    # eSortedVals = eVals[sortedIndices[:-k-1:-1]]
    eSortedVecs = eVecs[:, sortedIndices[:-k-1:-1]].T
    PCAMatrix = pd.DataFrame()
    for i in range(k):
        eigenvectorI = eSortedVecs[i]
        componentI = np.matmul(data, eigenvectorI)
        PCAMatrix = pd.concat([PCAMatrix, componentI], axis=1)
    PCAMatrix.columns = ['Component' + str(i + 1) for i in range(k)]
    return PCAMatrix


def PCARegression(data, response, selected):
    """

    :param data: pandas DataFrame containing dependent and independent variables
    :param response: string, column name of response variable
    :param selected: list, column names of independent variables
    :return: model summary
    """
    formula = "{} ~ {} + 1".format(response, "+".join(selected))
    model = smf.ols(formula, data).fit()
    summary = model.summary()

    return summary

