from __future__ import division
import pandas as pd
import numpy as np
import math as m
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# 构建板块


def PlateConstruction(data, industry):
    plateData = data[industry]
    # 以等权的方法构建指数
    plateData['platePrice'] = plateData.mean(1)
    # 求对数收益率
    logReturn = []
    for i in range(1, len(plateData), 1):
        logReturnI = m.log(
            plateData['platePrice'][i]/plateData['platePrice'][i-1])
        logReturn.append(logReturnI)
    return logReturn

# 描述性统计及年化夏普率


def DesciptStats(plate):
    plateDescriptStats = plate.describe().T
    plateDescriptStats['Sharpe_Annual'] = plateDescriptStats['mean'] * \
        52 / (plateDescriptStats['std'] * m.sqrt(52))
    return plateDescriptStats

# 数据处理


def DataProcessor(plate):
    processedData = pd.DataFrame()
    for i in range(plate.shape[1]):
        tempList = []
        # 中位数法去极值
        x_median = plate.iloc[:, i].median()
        d_median = abs(plate.iloc[:, i] - x_median).median()
        for j in range(len(plate)):
            if plate.iloc[j, i] > x_median + len(plate) * d_median:
                x_minusExtreme = x_median + len(plate) * d_median
            elif plate.iloc[j, i] < x_median - len(plate) * d_median:
                x_minusExtreme = x_median - len(plate) * d_median
            else:
                x_minusExtreme = plate.iloc[j, i]
            tempList.append(x_minusExtreme)
        # 中心化和标准化
        x_mean = np.mean(tempList)
        x_std = np.std(tempList)
        processedList = (tempList - x_mean)/x_std
        processedListDf = pd.DataFrame(processedList)
        processedData = pd.concat([processedData, processedListDf], axis=1)
    return processedData

# 多重共线性及方差膨胀因子的确定


def VifCalculator(plateData):
    # 计算各板块间的相关性
    rSquare = plateData.corr()
    # 通过相关性矩阵计算方差膨胀因子
    cMatrix = np.linalg.inv(rSquare)
    vifList = []
    for i in range(len(cMatrix)):
        vifList.append(cMatrix[i][i])
    vifDic = {"VIF": vifList}
    vifDataFrame = pd.DataFrame(vifDic).T
    vifDataFrame.columns = ['UpCycle', 'MidCycle',
                            'UnderCycle', 'Finance', 'Consume', 'Growth']
    return vifDataFrame


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


def HierarchicalBacktesting(data, industry, selected):
    """
    :param data
    :param industry: list, containing industry string
    :param selected: list, containing principle component
    :return: pandas DataFrame describing the performance of strategy

    """
    # 总时间长度
    timePeriod = len(data)
    # 划分训练集和测试集
    splitLine = int(0.7*len(data))

    firstLayerList, secondLayerList, thirdLayerList = [], [], []

    for i in range(splitLine, timePeriod):
        # 构建训练集，每个时间截点要往前推一期
        trainData = data.iloc[i-splitLine:i, :]
        returnDic = {}
        for j in industry:
            # 分别对每个行业依次做主成分回归
            formula = "{} ~ {} + 1".format(j, "+".join(selected))
            model = smf.ols(formula, data).fit()
            params = model.params
            component = trainData.iloc[-1, -6:]
            intercept = pd.Series({'Intercept': 1})
            componentIntercept = component.append(intercept)
            predictReturn = np.matmul(params, componentIntercept)
            returnDic[j] = predictReturn
        returnDf = pd.DataFrame(returnDic, index=[0]).T
        returnDf.columns = ['Return']
        # 分3层回测
        sortedReturnDf = returnDf.sort_values(by='Return', ascending=False)
        # 第一层
        firstLayer = sortedReturnDf.iloc[0:2, :]['Return'].mean()
        firstLayerList.append(firstLayer)
        # 第二层
        secondLayer = sortedReturnDf.iloc[2:4, :]['Return'].mean()
        secondLayerList.append(secondLayer)
        # 第三层
        thirdLayer = sortedReturnDf.iloc[4:6, :]['Return'].mean()
        thirdLayerList.append(thirdLayer)

    def NetValue(array):
        # 计算单位净值
        x = 1
        netValue = [1]
        for i in range(len(array)):
            x = x * (1 + array[i])
            netValue.append(x)
        return netValue

    netValueFirst = NetValue(firstLayerList)
    netValueSecond = NetValue(secondLayerList)
    netValueThird = NetValue(thirdLayerList)

    def get_max_drawdown(array):
        # 计算最大回撤
        drawdowns = []
        for i in range(len(array)):
            max_array = max(array[:i+1])
            drawdown = max_array - array[i]
            drawdowns.append(drawdown)
        return max(drawdowns)

    drawdownFirst = get_max_drawdown(netValueFirst)
    drawdownSecond = get_max_drawdown(secondLayerList)
    drawdownThird = get_max_drawdown(netValueThird)

    drawdown = pd.DataFrame({'FirstLayer': drawdownFirst,
                             'SecondLyer': drawdownSecond,
                             'ThirdLayer': drawdownThird}, index=[0]).T
    drawdown.columns = ['Drawdown']
    Layers = pd.DataFrame({'FirstLayer': firstLayerList,
                           'SecondLyer': secondLayerList,
                           'ThirdLayer': thirdLayerList})
    resultMatrix = Layers.describe().T
    # 计算夏普比率
    resultMatrix['SharpeRatio'] = resultMatrix['mean'] * \
        52/(resultMatrix['std']*m.sqrt(52))
    finalMatrix = pd.concat([resultMatrix, drawdown], axis=1)

    return finalMatrix


def LongShortBackTesting(data, industry, selected):
    """
    :param data: pandas DataFrame, containing all variables and data
    :param industry: list, containing industry string
    :param selected: list, containing principle component
    :return: pandas DataFrame describing the performance of strategy

    """
    # 总时间长度
    timePeriod = len(data)
    # 划分训练集和测试集
    splitLine = int(0.7*len(data))

    returnList = []

    for i in range(splitLine, timePeriod):
        # 构建训练集，每个时间截点要往前推一期
        trainData = data.iloc[i-splitLine:i, :]
        returnDic = {}
        for j in industry:
            # 分别对每个行业依次做主成分回归
            formula = "{} ~ {} + 1".format(j, "+".join(selected))
            model = smf.ols(formula, data).fit()
            params = model.params
            component = trainData.iloc[-1, -6:]
            intercept = pd.Series({'Intercept': 1})
            componentIntercept = component.append(intercept)
            predictReturn = np.matmul(params, componentIntercept)
            returnDic[j] = predictReturn
        returnDf = pd.DataFrame(returnDic, index=[0]).T
        returnDf.columns = ['Return']
        # long-short
        sortedReturnDf = returnDf.sort_values(by='Return', ascending=False)
        firstReturn = sortedReturnDf['Return'].iloc[0]
        lastReturn = sortedReturnDf['Return'].iloc[len(sortedReturnDf)-1]
        netReturn = firstReturn - lastReturn
        returnList.append(netReturn)

    def NetValue(array):
        # 计算单位净值
        x = 1
        netValue = [1]
        for i in range(len(array)):
            x = x * (1 + array[i])
            netValue.append(x)
        return netValue

    netValue = NetValue(returnList)

    def get_max_drawdown(array):
        # 计算最大回撤
        drawdowns = []
        for i in range(len(array)):
            max_array = max(array[:i+1])
            drawdown = max_array - array[i]
            drawdowns.append(drawdown)
        return max(drawdowns)

    drawDown = get_max_drawdown(netValue)

    drawdown = pd.DataFrame({'LongShort': drawDown}, index=[0]).T
    drawdown.columns = ['Drawdown']
    longShortDf = pd.DataFrame({'LongShort': returnList})
    resultMatrix = longShortDf.describe().T
    # 计算夏普比率
    resultMatrix['SharpeRatio'] = resultMatrix['mean'] * \
        52/(resultMatrix['std']*m.sqrt(52))
    finalMatrix = pd.concat([resultMatrix, drawdown], axis=1)

    return finalMatrix


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
