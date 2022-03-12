import pandas as pd
from DataProcess import *
from StepwiseRegression import *
from PCA_Regression import *
from BackTesting import *

if __name__ == "__main__":

    # 导入数据
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

    # 逐步回归的数据准备
    timeIndex = pd.DataFrame(pd.date_range(
        start='2005-1-3', end='2022-2-28', freq='W-MON'), columns=['Date'])
    regreMajorPlate = pd.concat([processedMajorPlate, timeIndex], axis=1)
    regreMajorPlate.set_index(['Date'], inplace=True)

    # 设置时间段为2013年至2015年的大牛市行情区间
    # 分行业进行逐步回归

    # 周期上游
    upCycleDf = RegreDfConstruction(regreMajorPlate, '2013', '2015', 'UpCycle')
    upCycleModel = StepWiseRegression(upCycleDf, 'Response')
    # upCycleModel.summary()

    # 周期中游
    midCycleDf = RegreDfConstruction(
        regreMajorPlate, '2013', '2015', 'MidCycle')
    midCycleModel = StepWiseRegression(midCycleDf, 'Response')
    # midCycleModel.summary()

    # 周期下游
    underCycleDf = RegreDfConstruction(
        regreMajorPlate, '2013', '2015', 'UnderCycle')
    underCycleModel = StepWiseRegression(underCycleDf, 'Response')
    # underCycleModel.summary()

    # 大金融
    financeDf = RegreDfConstruction(
        regreMajorPlate, '2013', '2015', 'Finance')
    financeModel = StepWiseRegression(financeDf, 'Response')
    # financeModel.summary()

    # 消费
    consumeDf = RegreDfConstruction(
        regreMajorPlate, '2013', '2015', 'Consume')
    consumeModel = StepWiseRegression(consumeDf, 'Response')
    # consumeModel.summary()

    # 成长
    growthDf = RegreDfConstruction(
        regreMajorPlate, '2013', '2015', 'Growth')
    growthModel = StepWiseRegression(growthDf, 'Response')
    # growthModel.summary()

    # 全样本做主成分回归
    # 生成主成分
    prePCADf = PrincipleComponentConstruction(regreMajorPlate, 6)
    prePCADf = prePCADf.iloc[0:len(prePCADf)-1, :].reset_index(
        drop=True).copy()
    PCAMajorPlate = regreMajorPlate.iloc[1:len(regreMajorPlate)].reset_index(
        drop=True).copy()
    PCADf = pd.concat([PCAMajorPlate, prePCADf], axis=1)
    # 基于主成分回归，做回测
    industry = ['UpCycle',
                'MidCycle',
                'UnderCycle',
                'Finance',
                'Consume',
                'Growth']
    selected = ['Component'+str(i) for i in range(1, 7)]
    hierarchicalResult = HierarchicalBacktesting(
        PCADf, industry, selected)  # 多层回测
    longShortResult = LongShortBackTesting(PCADf, industry, selected)  # 多空回测
