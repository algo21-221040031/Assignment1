from __future__ import division
import pandas as pd
import numpy as np
import math as m
import statsmodels.formula.api as smf

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