import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from DataProcess import PlateConstruction, DataProcessor

# 构建回归数据集
# 将自变量和因变量拼接成一个数据框


def RegreDfConstruction(data, start, end, industry):
    # 选择对应时间区间内的数据
    timedData = data.loc[start:end, :]
    RegreDf = pd.DataFrame()
    # 因变量
    dependentVariable = timedData[industry].iloc[1:len(
        timedData)].reset_index(drop=True)
    RegreDf = pd.concat([RegreDf, dependentVariable], axis=1)
    RegreDf.columns = ['Response']
    # 自变量
    independentVariable = timedData.iloc[0:len(
        timedData)-1].reset_index(drop=True)
    # 构建矩阵
    RegreDf = pd.concat([RegreDf, independentVariable], axis=1)
    return RegreDf

# 逐步回归


def StepWiseRegression(data, response):
    """
    前向逐步回归法
    使用Adjusted R-Square来评判新加的参数是否提高回归中的统计显著性; 

    参数说明：
    -------
    data: pandas D  ataFrame, 包含所有自变量和因变量;
    response: string, 参数data中因变量的列名 

    返回值：
    -------
    model: 最优拟合的statsmodels下的线性模型
    """

    # 设定remaining集合，依次判断进入逐步回归的变量
    remaining = set(data.columns)
    # 移除因变量
    remaining.remove(response)
    # 定义一个空列表，储存最终进入模型的变量
    selected = []
    # 模型的拟合程度
    currentScore, bestNewScore = 0.0, 0.0
    while remaining and currentScore == bestNewScore:
        scoresWithCandidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           "+".join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scoresWithCandidates.append((score, candidate))
        scoresWithCandidates.sort()
        bestNewScore, bestCandidate = scoresWithCandidates.pop()
        if currentScore < bestNewScore:
            remaining.remove(bestCandidate)
            selected.append(bestCandidate)
            currentScore = bestNewScore
    formula = "{} ~ {} + 1".format(response, '+'.join(selected))

    model = smf.ols(formula, data).fit()

    return model
