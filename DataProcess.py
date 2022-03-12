from __future__ import division
import pandas as pd
import numpy as np
import math as m

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

