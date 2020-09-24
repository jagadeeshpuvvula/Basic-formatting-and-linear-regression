'''
Dataset: Etsy - Table lamps (n = 10,053) scraped on Apr 28, 2020
Analysis: Descriptive analysis and Inferential statistics using multiple linear regression.
Objective: To understand the impact of different variables avaiable in the dataset on the product price
'''
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sm
import patsy as pt

data = pd.read_csv("C:/Users/jagad/Desktop/Python_spring_2020/PJ_proj_result.csv", sep='\t', header=(0), engine='python') #reading the dataset
print('Column names:', data.columns) #column names
print('No. of rows, columns:', data.shape) #dimensions of the dataset

#creating list of categorical and continous variables for the descriptive analysis
numerical = [ ' startPrice', ' rating', ' nreview']
categorical = [' made', ' rare', ' exaggeration', ' almostgone', ' bestseller', ' freeshipping']
data=data[numerical+categorical]

data.dropna() #removing all the missing values. In this dataset we dont have any missing observations
data.describe() # summary statistics of continous variables

#creating new variable to represent the store value
data['storeValue'] = np.log(data[' rating']*data[' nreview'])

#contingency tables to understand the categorical variables
#Table 1
d1= data[[" made", " rare"]]
table = sm.stats.Table.from_data(d1)
print(table.table_orig)

#Table 2
d2= data[[" exaggeration", " rare"]]
table2 = sm.stats.Table.from_data(d2)
print(table2.table_orig)

#Table 3
d3= data[[" almostgone", " rare"]]
table3 = sm.stats.Table.from_data(d3)
print(table3.table_orig)

#Table 4
d4= data[[" almostgone", " exaggeration"]]
table4 = sm.stats.Table.from_data(d4)
print(table4.table_orig)

#Panel plot by price and all the item attributes (categorical)
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for var, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x=var, y=' startPrice', data=data, ax=subplot)
fig.show()
fig.savefig("C:/Users/jagad/Desktop/01.png", format='png', dpi=300)

#scatter plot between price and store value
sns.jointplot(x=data[' startPrice'], y=data['storeValue']);
plt.show()
plt.savefig("C:/Users/jagad/Desktop/02.png", format='png', dpi=300)


#trendline between price and store value
sns.relplot(x=" startPrice", y="storeValue", kind="line", ci="sd", data=data);
plt.show()

#trend line between price and store value stratified by 
sns.relplot(x=" startPrice", y="storeValue", hue=" freeshipping", kind="line", data=data);
plt.show()

#correlation plot of all the variables in the dataset
corr = data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

#converting Boolean values to numeric
data[[' made', ' rare', ' exaggeration', ' almostgone', ' bestseller', ' freeshipping']]*=1

#correlation matrix table
cor_mat = data.corr()
print(cor_mat)

#embedding correlation coefficient to heatmap
sns.heatmap(cor_mat, annot=True)
plt.show()

#multiple linear regression

y = data[[ ' startPrice']]
X = data[[' made', ' rare', ' exaggeration', ' almostgone', ' bestseller', ' freeshipping', 'storeValue']]

y, X = pt.dmatrices("y ~ X", data=data)
valueReg = sm.OLS(y, X).fit()
valueReg.summary()