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


if __name__ == "__main__":
    originData = pd.read_excel("ClosePriceIndustry.xlsx", index_col=0)

    # 周期上游：石油石化，煤炭，有色金属
    upCycleIndustry = ['801960.SI', '801950.SI', '801050.SI']
    upCycle = PlateConstruction(originData, upCycleIndustry)

    # 周期中游：公用事业、钢铁、基础化工、交通运输
    midCycleIndustry = ['801160.SI', '801040.SI', '801030.SI', '801170.SI']
    midCycle = PlateConstruction(originData, midCycleIndustry)

    # 周期下游：建筑、建材、汽车、机械
    underCycleIndustry = ['801720.SI', '801710.SI', '801880.SI', '801890.SI']
    underCycle = PlateConstruction(originData, underCycleIndustry)

    # 大金融：银行、非银金融、房地产
    financeIndustry = ['801780.SI', '801790.SI', '801180.SI']
    finance = PlateConstruction(originData, financeIndustry)

    # 消费：轻工制造、商贸零售、美容护理、家电、纺织服装、医药、食品饮料、农林牧渔
    consumeIndustry = ['801140.SI', '801200.SI', '801980.SI',
                       '801110.SI', '801130.SI', '801150.SI', '801120.SI', '801010.SI']
    consume = PlateConstruction(originData, consumeIndustry)

    # 成长：计算机、传媒、通信、电力设备、电子
    growthIndustry = ['801750.SI', '801760.SI',
                      '801770.SI', '801730.SI', '801080.SI']
    growth = PlateConstruction(originData, growthIndustry)

    # 生成六大周期板块
    majorPlate = pd.DataFrame(
        {'UpCycle': upCycle,
         'MidCycle': midCycle,
         'UnderCycle': underCycle,
         'Finance': finance,
         'Consume': consume,
         'Growth': growth})

    # 数据预处理
    processedMajorPlate = DataProcessor(majorPlate)
    processedMajorPlate.columns = ['UpCycle',
                                   'MidCycle',
                                   'UnderCycle',
                                   'Finance',
                                   'Consume',
                                   'Growth']
    timeIndex = pd.DataFrame(pd.date_range(
        start='2005-1-3', end='2022-2-28', freq='W-MON'), columns=['Date'])
    regreMajorPlate = pd.concat([processedMajorPlate, timeIndex], axis=1)
    regreMajorPlate.set_index(['Date'], inplace=True)
