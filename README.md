# Assignment1
## Introduction
This project demonstrates a method based on general regression forecasting model to mine industry rotation laws and guide industry allocation. There is a significant cross-prediction relationship between the rates, and then a rotation strategy is constructed. The proposal of this model is mainly based on two considerations: 1. There are extensive and close connections between various industries. Through the regression model, the transmission law between industry returns can be quantitatively described; 2. The industry index itself is to observe the operating state of the macro economy. A good window of the industry index, and the return rate of the industry index depicts the dynamic changes of the economy, and can be used to guide the industry allocation by mining the transmission relationship between the return rates of various industries.

## Language Environment
* Python 3.9
* Modules: pandas, numpy, statsmodels.

## Files Description
* Folder Reference Paper: containing the reference paper, "华泰证券行业轮动系列之一：基于通用回归模型的行业轮动策略";
* ClosePriceIndustry.xlsx: source data, containing the close price of ShenWan Industry Index (Level I), weekly frequency, got from WIND.
* DataProcess.py: python code to preprocess data before modelling, including constructing industry plates, presenting descriptstatistics, constructing related pandas DataFrame, and calculate correlation and VIF, etc.;
* StepWiseRegression.py: python code to execute stepwise regression on different industry plates and time section, mining the transmission law between industry;
* PCA_Regression.py: python code to construct principle component of industry plates, making preparation for the latter strategy backtesting based on PCA regression;
* BackTesting.py: python code to backtest two strategies: Hierarchical Strategy and Long-Short Strategy, both based on PCA regression;
* main.py: python code to execute the whole files.

## Ideas
### Stepwise Regression
Collinearity means that there is an exact or approximate linear relationship between the explanatory variables. In regression model, the independent variable is the series of returns of various industries, and most of the time, the phenomenon of rising and falling at the same time in various industries is relatively significant, resulting in a high degree of correlation between the series of returns. By calculating , the correlation coefficients of consumption, mid-cycle and downstream of the cycle are the highest, all exceeding 0.9, and the correlation coefficient of the big finance and growth sectors is the lowest, also exceeding 0.6. The existence of collinearity will not affect the uniqueness and unbiasedness of the regression coefficient, but it will cause the variance of the estimated value of the regression coefficient to increase, making the confidence interval of the regression coefficient very wide, which on the one hand will reduce the accuracy of the estimator degree, and even change the sign of the coefficient; on the other hand, it may also make the estimator's 𝑇 value smaller, causing some explanatory variables that have a significant impact on the dependent variable to fail the hypothesis test. Therefore, whether collinearity is directly related to the goodness of the regression equation must be dealt with.

The basic idea of the stepwise regression is to introduce the variables into the model one by one. After each explanatory variable is introduced, the F test is performed, and the t-test is performed on the explanatory variables that have been selected one by one. If it is significant again, delete it to ensure that only significant variables are included in the regression equation each time a new variable is introduced. After stepwise regression, the explanatory variables that remained in the model were both significant and did not have severe multicollinearity. In the following text, the stepwise regression method will be used when modeling the typical interval of the historical market.

### PCA Regression
The principle of PCA Regression is to recombine the original explanatory variables into a new set of independent variables through linear transformation to replace the original explanatory factors and construct a regression equation. The advantage is that the collinearity effect in the original multiple linear regression model can be eliminated, but the disadvantage is that the newly synthesized explanatory factors do not necessarily have intuitive economic meanings. When constructing the industry rotation strategy, it will be based on the principal component regression method.

## Conclusion
### Stepwise Regression
Stepwise Regression is focused on the modeling the bull market from 2013 to 2015. The results show that the Growth Plate has dominated the bull market, while gains in the upper-cycle and large financial sectors have lagged far behind, as macroeconomic fundamentals have been in a downward trend since the 2008 global financial crisis, and deflation expectations have prompted the government to take action. Interest rate cuts and other easing policies, but the deterioration of fundamentals has led to serious damage to the profits of most companies. The abundant funds have no suitable destination, and can only flow into the growth sectors that speculate on expectations and look at the future. It can also be seen from the regression coefficient of the model that the rate of return of the Growth Plate positively promotes the rate of return of the mid-cycle, undercycle and consume plates, which also confirms that the growth sector is the power node and leading sector of this round of market.
### PCA Regression

#### HierarchicalBacktesting
Hierarchical backtesting is the most intuitive means to demonstrate the predictive ability of the model. The construction method is as follows:
* Optional targets: Upcycle, midcycle, undercycle, finance, consume, growth;
* Backtracking interval: 30%;
* Parameter setting: The length of the training window is 70%, and the regression model is constructed considering all principal component vectors;
* Portfolio construction: On the last trading day of each week, the principal component regression model is trained based on the weekly logarithmic return data of the six major plates in the past 70% time window, the pricing equation is constructed, and the return for the next week is predicted by extension. The yield ranking builds a tiered portfolio, and swaps positions at the closing price on the first trading day of next week;
* Evaluation methods: Backtest annualized return, Sharpe ratio, maximum drawdown.

#### LongShortBacktest
Assuming that all sector indices can be short-sold, a long-short strategy portfolio can be constructed, that is, at each section, buy the sector with the highest predicted yield and short sell the sector with the lowest predicted yield to hedge market risks and obtain alpha income.
The results can be obtained from main.py.
