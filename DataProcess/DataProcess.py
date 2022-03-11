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


if __name__ == '__main__':

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
    
    # 数据的描述性统计
    majorDescriptStats = DesciptStats(majorPlate)
    # 数据预处理
    processedMajorPlate = DataProcessor(majorPlate)
    processedMajorPlate.columns = ['UpCycle',
                                   'MidCycle',
                                   'UnderCycle',
                                   'Finance',
                                   'Consume',
                                   'Growth']

    # 计算方差膨胀因子
    plateVIF = VifCalculator(processedMajorPlate)
