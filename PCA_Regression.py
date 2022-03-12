import pandas as pd
import numpy as np
from statsmodels.api import smf
from DataProcess import PlateConstruction, DataProcessor


def PrincipleComponentConstruction(data, k):
    """

    :param data: pandas DataFrame, the processed return data of plates
    :param k: top k eigenvalues
    :return:
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


def PCARegression(data):


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
