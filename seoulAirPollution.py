#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np, pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import graphviz
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from preprocessing import Preprocessing


# In[3]:


#*****Data preprocessing*****

#load Air Pollution Datasets in Beijing
aotizhongxin = pd.read_csv("PRSA_Data_Aotizhongxin_20130301-20170228.csv")
changping = pd.read_csv("PRSA_Data_Changping_20130301-20170228.csv")
dingling = pd.read_csv("PRSA_Data_Dingling_20130301-20170228.csv")
dongsi = pd.read_csv("PRSA_Data_Dongsi_20130301-20170228.csv")
guanyuan = pd.read_csv("PRSA_Data_Guanyuan_20130301-20170228.csv")
gucheng = pd.read_csv("PRSA_Data_Gucheng_20130301-20170228.csv")
huairou = pd.read_csv("PRSA_Data_huairou_20130301-20170228.csv")
nongzhanguan = pd.read_csv("PRSA_Data_Nongzhanguan_20130301-20170228.csv")
shunyi = pd.read_csv("PRSA_Data_Shunyi_20130301-20170228.csv")
tiantan = pd.read_csv("PRSA_Data_Tiantan_20130301-20170228.csv")
wanliu = pd.read_csv("PRSA_Data_Wanliu_20130301-20170228.csv")
wanshouxigong = pd.read_csv("PRSA_Data_Wanshouxigong_20130301-20170228.csv")

#load Air Pollution and Weather Datasets in Seoul
seoul = pd.read_csv("seoul_air_20130301-20170228.csv")
weather = pd.read_csv("weather.csv")


# In[4]:


#*Data restructuring
#Merging Beijing Datasets
beijing = pd.concat([aotizhongxin,changping,dingling,dongsi,
                     guanyuan,gucheng,huairou,nongzhanguan,
                     shunyi,tiantan,wanliu,wanshouxigong],ignore_index = True)

beijing.info()
beijing.describe()


#Drop No column
beijing.drop(columns="No",inplace = True)

beijing


# In[5]:


#change year, month, day, hour columns to date format 
tt = pd.DataFrame(pd.date_range("2013-03-01","2017-03-01",freq="h"))
tt = tt[:-1]
tt = pd.concat([tt,tt,tt,tt,tt,tt,tt,tt,tt,tt,tt,tt],ignore_index = True)
beijing["Date"] = tt
#drop year, month, day, hour columns 
beijing.drop(columns=["year","month","day","hour"],inplace=True)


# In[6]:


beijing


# In[7]:


beijing.info()
seoul.info()
weather.info()


# In[8]:


weather
seoul


# In[9]:


#change MSRDT column to date format 
date_s = seoul["MSRDT"].astype(str)
seoul["MSRDT"] = date_s.apply(lambda x: datetime.strptime(x, "%Y%m%d%H%M"))
seoul


# In[10]:


weather["tm"] = weather["tm"].apply(lambda _: datetime.strptime(_,"%Y-%m-%d"))
weather.info()


# In[11]:


#*Data value changes
#find nan values
seoul.isnull().sum()
weather.isnull().sum()
beijing.isnull().sum()


# In[12]:


#weather - If sumRn is nan value, it's a non-rainy day, so fill it at 0.0.
weather["sumRn"].fillna(0.0,inplace=True)


# In[13]:


weather.isnull().sum()


# In[14]:


#Fill weather missing values with predictions after learning the model with regression 
#after determining which column has the highly correlation.
#If a value of about 0.5 is obtained: a strong positive (+) correlation. As the variable x increases, the variable y increases.
weather.corr()


# In[15]:


#corr() heatmap
sns.heatmap(weather.corr(), annot=True, fmt='0.2f')
plt.title('weather', fontsize=20)
plt.show()


# In[16]:


#sumGsr-avgTs : 0.5
#sumGsr-minRhm : -0.55
#sumGsr-avgTs : 0.4
#sumGsr-avgTs --> accuray = 0.2..
#sumGsr-avgTsm,minRhm --> accuray = 0.6
#sumGsr-avgTsm,minRhm,avgTa --> accuracy = 0.72
#linear regression 
linear,score = Preprocessing().reg_score(weather,["avgTs","minRhm","avgTa"],"sumGsr")
score
pred_y = linear.predict(weather.loc[:, ["avgTs","minRhm","avgTa"]])
weather['sumGsr'].fillna(pd.Series(pred_y), inplace=True)


# In[17]:


weather.isnull().sum()


# In[18]:


#maxWd-sumGsr : 0.2
#Since maxwd does not show significant correlation with other features, 
#maxwd fills it with data from the previous day and the next day.
weather[weather["maxWd"].isnull() == True]


# In[19]:


weather["maxWd"].fillna(method='ffill', limit=1,inplace=True)


# In[20]:


weather[weather["maxWd"].isnull() == True]


# In[21]:


weather["maxWd"].fillna(method='bfill', limit=1,inplace=True)


# In[22]:


#cleaned data
weather.isnull().sum()


# In[23]:


#beijing data's nan value
beijing.isnull().sum()


# In[24]:


#Proceed with the same process as Weather.
beijing.corr()


# In[25]:


sns.heatmap(beijing.corr(), annot=True, fmt='0.2f')
plt.title('Beijing', fontsize=20)
plt.show()


# In[26]:


#onehotencoding for station, wd columns
ohe = OneHotEncoder()
station_oh = ohe.fit_transform(beijing["station"].values.reshape(-1,1)).toarray()
beijing_oh = beijing.copy()
station_oh = pd.DataFrame(station_oh,columns=ohe.categories_[0])


# In[27]:


station_oh


# In[28]:


beijing_oh.drop(columns="station",inplace=True)


# In[29]:


beijing_oh = pd.concat([beijing_oh,station_oh],axis=1)


# In[30]:


beijing_oh


# In[31]:


#The wind direction is filled with max values.
beijing_oh["wd"].fillna(beijing_oh["wd"].value_counts().index[beijing_oh["wd"].value_counts().argmax()],inplace=True)


# In[32]:


beijing_oh["wd"].isnull().sum()


# In[33]:


#onehotencoding after fill nan values.
wd_oh = ohe.fit_transform(beijing_oh["wd"].values.reshape(-1,1)).toarray()
wd_oh = pd.DataFrame(wd_oh,columns=ohe.categories_[0])
beijing_oh = pd.concat([beijing_oh,wd_oh],axis=1)
beijing_oh.drop(columns="wd",inplace=True)


# In[34]:


beijing_oh.columns


# In[35]:


beijing_oh


# In[36]:


beijing_oh.isnull().sum()


# In[37]:


#Proceed to fill missing values with the same process as weather data
beijing.corr()


# In[38]:


sns.heatmap(beijing.corr(), annot=True, fmt='0.2f')
plt.title('Beijing', fontsize=20)
plt.show()


# In[41]:


#predict PM2.5 - using PM10, NO2, CO 
linear,score = Preprocessing().reg_score(beijing_oh,["PM10","SO2","NO2","CO"],"PM2.5",0.2,1)
score
pred_y = linear.predict(beijing_oh[["PM10","SO2","NO2","CO"]].dropna(how="any"))
beijing_oh["PM2.5"].fillna(pd.Series(pred_y), inplace=True)


# In[42]:


beijing_oh.isnull().sum()


# In[44]:


#predict PM10 -also using PM2.5,SO2 NO2, CO
#linear regression 
linear,score = Preprocessing().reg_score(beijing_oh,["PM2.5","SO2","NO2","CO"],"PM10",0.2,2)
score
pred_y = linear.predict(beijing_oh[["PM2.5","SO2","NO2","CO"]].dropna(how="any"))
beijing_oh["PM10"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum()


# In[46]:


#predict SO2 -also using PM2.5, PM10, NO2, CO
#linear regression 
linear,score = Preprocessing().reg_score(beijing_oh,["PM2.5","PM10","NO2","CO"],"SO2",0.2,3)
score
#Low score, SO2 fills nan values with median instead of regression
beijing_oh["SO2"].fillna(beijing_oh["SO2"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[48]:


#predict NO2 -also using PM2.5,PM10,SO2,CO
#linear regression 
linear,score = Preprocessing().reg_score(beijing_oh,["PM2.5","PM10","SO2","CO"],"NO2",0.2,4)
score
#Low score, NO2 fills nan values with median instead of regression
beijing_oh["NO2"].fillna(beijing_oh["NO2"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[50]:


#predict CO -also using PM2.5,PM10,SO2,NO2
#linear regression 
linear,score = Preprocessing().reg_score(beijing_oh,["PM2.5","PM10","SO2","NO2"],"CO",0.2,5)
score
pred_y = linear.predict(beijing_oh[["PM2.5","PM10","SO2","NO2"]].dropna(how="any"))
beijing_oh["CO"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum()


# In[51]:


#predict O3 - using NO2,TEMP,PRES
#linear regression 
linear,score = Preprocessing().reg_score(beijing_oh,["NO2","TEMP","PRES"],"O3",0.2,6)
score
##Low score, O3 fills nan values with median instead of regression
beijing_oh["O3"].fillna(beijing_oh["O3"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[53]:


#predict TEMP - using O3,PRES,DEWP
#linear regression 
linear,score = Preprocessing().reg_score(beijing_oh,["O3","PRES","DEWP"],"TEMP",0.2,7)
score
pred_y = linear.predict(beijing_oh[["O3","PRES","DEWP"]].dropna(how="any"))
beijing_oh["TEMP"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[55]:


#predict PRES - using O3,TEMP,DEWP
#linear regression 
linear,score = Preprocessing().reg_score(beijing_oh,["O3","TEMP","DEWP"],"PRES",0.2,8)
score
pred_y = linear.predict(beijing_oh[["O3","TEMP","DEWP"]].dropna(how="any"))
beijing_oh["PRES"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[56]:


#predict DEWP - using TEMP,PRES
#linear regression 
linear,score = Preprocessing().reg_score(beijing_oh,["TEMP","PRES"],"DEWP",0.2,9)
score
pred_y = linear.predict(beijing_oh[["TEMP","PRES"]].dropna(how="any"))
beijing_oh["DEWP"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[57]:


#All columns have low correlation with RAIN
#ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["RAIN"].isnull() == True]
beijing_oh["RAIN"].fillna(method='ffill', limit=1,inplace=True)


# In[58]:


beijing_oh["RAIN"].fillna(method='bfill', limit=1,inplace=True)


# In[59]:


beijing_oh[beijing_oh["RAIN"].isnull() == True]


# In[60]:


beijing_oh["RAIN"].fillna(beijing_oh["RAIN"].median(),inplace=True)


# In[61]:


#All columns have low correlation with WSPM
#ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["WSPM"].isnull() == True]
beijing_oh["WSPM"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["WSPM"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["WSPM"].isnull() == True]


# In[62]:


beijing_oh["WSPM"].fillna(beijing_oh["WSPM"].median(),inplace=True)


# In[63]:


beijing_oh.isnull().sum() 


# In[64]:


#Remaining nan value - PM2.5, PM10,CO 
beijing_oh[beijing_oh["PM2.5"].isnull() == True]
beijing_oh[beijing_oh["PM10"].isnull() == True]
beijing_oh[beijing_oh["CO"].isnull() == True]


# In[65]:


#ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["PM2.5"].isnull() == True]
beijing_oh["PM2.5"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["PM2.5"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["PM2.5"].isnull() == True]


# In[66]:


beijing_oh["PM2.5"].fillna(beijing_oh["PM2.5"].median(),inplace=True)


# In[67]:


#ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["PM10"].isnull() == True]
beijing_oh["PM10"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["PM10"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["PM10"].isnull() == True]

beijing_oh["PM10"].fillna(beijing_oh["PM10"].median(),inplace=True)


# In[68]:


#ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["CO"].isnull() == True]
beijing_oh["CO"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["CO"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["CO"].isnull() == True]

beijing_oh["CO"].fillna(beijing_oh["CO"].median(),inplace=True)


# In[69]:


#cleaned data
beijing_oh.isnull().sum()


# In[70]:


clean_beijing = beijing_oh.copy()


# In[71]:


clean_beijing.columns


# In[72]:


#scaling dataset with standard, minmax, maxabs,robust scalier
std_beijing = pd.DataFrame(StandardScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)
mm_beijing = pd.DataFrame(MinMaxScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)
ma_beijing = pd.DataFrame(MaxAbsScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)
rb_beijing = pd.DataFrame(RobustScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)


# In[73]:


std_beijing = pd.concat([std_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
mm_beijing = pd.concat([mm_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
ma_beijing = pd.concat([ma_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
rb_beijing = pd.concat([rb_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
std_beijing.columns=clean_beijing.columns
mm_beijing.columns=clean_beijing.columns
ma_beijing.columns=clean_beijing.columns
rb_beijing.columns=clean_beijing.columns


# In[74]:


clean_beijing
std_beijing
mm_beijing
ma_beijing
rb_beijing


# In[75]:


std_weather = pd.DataFrame(StandardScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
std_weather["tm"]=weather["tm"]


# In[76]:


mm_weather = pd.DataFrame(MinMaxScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
mm_weather["tm"]=weather["tm"]


# In[77]:


ma_weather = pd.DataFrame(MaxAbsScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
ma_weather["tm"]=weather["tm"]


# In[78]:


rb_weather = pd.DataFrame(RobustScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
rb_weather["tm"]=weather["tm"]


# In[79]:


std_weather
mm_weather
ma_weather
rb_weather


# In[80]:


#seoul data onehotencoding for MSRSTE_NM
ohe = OneHotEncoder()
Nm_oh = ohe.fit_transform(seoul["MSRSTE_NM"].values.reshape(-1,1)).toarray()
seoul_oh = seoul.copy()
Nm_oh = pd.DataFrame(Nm_oh,columns=ohe.categories_[0])
Nm_oh
seoul_oh.drop(columns="MSRSTE_NM",inplace=True)
seoul_oh = pd.concat([seoul_oh,Nm_oh],axis=1)


# In[81]:


seoul_oh["영등포구"]


# In[82]:


seoul_oh


# In[83]:


std_seoul = pd.DataFrame(StandardScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)
mm_seoul = pd.DataFrame(MinMaxScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)
ma_seoul = pd.DataFrame(MaxAbsScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)
rb_seoul = pd.DataFrame(RobustScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)


# In[84]:


col = seoul_oh.iloc[:,1:7].columns.append(seoul_oh.drop(columns=std_seoul.columns).columns)


# In[85]:


std_seoul = pd.concat([std_seoul,seoul_oh.drop(columns=std_seoul.columns)],axis=1,ignore_index = True)
mm_seoul = pd.concat([mm_seoul,seoul_oh.drop(columns=mm_seoul.columns)],axis=1,ignore_index = True)
ma_seoul = pd.concat([ma_seoul,seoul_oh.drop(columns=ma_seoul.columns)],axis=1,ignore_index = True)
rb_seoul = pd.concat([rb_seoul,seoul_oh.drop(columns=rb_seoul.columns)],axis=1,ignore_index = True)
std_seoul.columns=col
mm_seoul.columns=col
ma_seoul.columns=col
rb_seoul.columns=col


# In[86]:


std_seoul
mm_seoul
ma_seoul
rb_seoul


# In[87]:


#**************data analysis*********************#
#Create decision Tree,Analysis of features that have the greatest importance in predicting fine dust.


# In[88]:


std_weather
std_seoul
std_beijing


# In[89]:


#Seoul data -> group by date, reduced to 1461 lines as shown by weather data.
#Beijing Data -> the same process.
std_seoul_group = std_seoul.groupby(std_seoul['MSRDT'].dt.date).mean()
std_seoul_group = std_seoul_group.iloc[:,:6]


# In[90]:


std_seoul_group


# In[91]:


std_beijing_group = std_beijing.iloc[:,:12].groupby(std_beijing['Date'].dt.date).mean()
#Maximum wind direction (number of times) is obtained for wind direction data analysis.
std_beijing_wd = std_beijing.iloc[:,24:].groupby(std_beijing['Date'].dt.date).sum().idxmax(axis=1)
std_beijing_wd = pd.DataFrame(std_beijing_wd,columns = ["WD"])


# In[92]:


std_beijing_group = pd.concat([std_beijing_group,std_beijing_wd,],axis=1)


# In[93]:


#wind direction onehotencoding 
ohe = OneHotEncoder()
group_wd_oh = ohe.fit_transform(std_beijing_group["WD"].values.reshape(-1,1)).toarray()
group_wd_oh = pd.DataFrame(group_wd_oh,index =std_beijing_group.index , columns=ohe.categories_[0])
std_beijing_group = pd.concat([std_beijing_group,group_wd_oh],axis = 1)


# In[94]:


std_beijing_group.drop(columns="WD",inplace = True)


# In[95]:


std_beijing_group


# In[96]:


std_weather.index = std_weather["tm"]
std_weather.drop(columns="tm",inplace= True)


# In[97]:


std_weather


# In[98]:


#Merge seoul weather dataset and seoul air pollution dataset
std_tree = pd.concat([std_weather,std_seoul_group],axis = 1)


# In[99]:


std_tree


# In[100]:


#Column name change, and merge datasets
std_beijing_group = std_beijing_group.add_prefix("beijing_")


# In[101]:


std_tree = pd.concat([std_beijing_group,std_tree],axis = 1)


# In[103]:


std_target1 = std_tree["PM10"]
std_target2 = std_tree["PM25"]
std_data = std_tree.drop(columns=["PM10","PM25"])


# In[104]:


#split train , test data for performance
std_train_X1,std_test_X1,std_train_y1,std_test_y1 = train_test_split(std_data,std_target1,test_size=0.2,random_state=1)
#split train , valid data for learning 
std_train_X1,std_valid_X1,std_train_y1,std_valid_y1 = train_test_split(std_train_X1,std_train_y1,test_size = 0.3,random_state = 11)


# In[105]:


#split train , test data for performance
std_train_X2,std_test_X2,std_train_y2,std_test_y2 = train_test_split(std_data,std_target2,test_size=0.2,random_state=2)
#split train , valid data for learning 
std_train_X2,std_valid_X2,std_train_y2,std_valid_y2 = train_test_split(std_train_X2,std_train_y2,test_size = 0.3,random_state = 21)


# In[106]:


tree_params = {
    "criterion":["mse", "friedman_mse", "mae", "poisson"],
    "splitter" : ["best","random"],
    "max_depth" : [2,6,10,14],
    "max_features" : ["auto","sqrt","log2",10,None]
}


# In[107]:


#decision tree with gridsearchcv
tree_grid = GridSearchCV(DecisionTreeRegressor(random_state = 122),
                         tree_params,
                         scoring="neg_mean_squared_error",
                         verbose = 1,
                         n_jobs = -1)


# In[108]:


tree_grid.fit(std_train_X1,std_train_y1)


# In[109]:


tree_grid.best_score_
tree_grid.best_params_


# In[110]:


#make decision tree with best params
best_tree = DecisionTreeRegressor(criterion="mae",max_depth=6,max_features="auto",splitter="best",random_state=121)


# In[111]:


#predict with test data and get score
#train dataset
best_tree.fit(std_train_X1,std_train_y1)
best_pred = best_tree.predict(std_train_X1)
mse = mean_squared_error(best_pred,std_train_y1)
print("------------std_train_mse---------------")
mse
print("------------std_train_score---------------")
best_tree.score(std_train_X1,std_train_y1)

#valid dataset
best_tree.fit(std_valid_X1,std_valid_y1)
best_valid_pred = best_tree.predict(std_valid_X1)
valid_mse = mean_squared_error(best_valid_pred,std_valid_y1)
print("------------std_valid_mse---------------")
valid_mse
print("------------std_valid_score---------------")
best_tree.score(std_valid_X1,std_valid_y1)

#test dataset
best_tree.fit(std_test_X1,std_test_y1)
best_test_pred = best_tree.predict(std_test_X1)
test_mse = mean_squared_error(best_test_pred,std_test_y1)
print("------------std_test_mse---------------")
test_mse
print("------------std_test_root_mse---------------")
rmse = np.sqrt(test_mse)
rmse
print("------------std_test_score---------------")
best_tree.score(std_test_X1,std_test_y1)


# In[112]:


#gradient boosting with std scaled dataset
gbr = GradientBoostingRegressor(random_state = 13).fit(std_train_X1,std_train_y1)
gbr_pred = gbr.predict(std_test_X1)
gbr.score(std_test_X1,std_test_y1)
RMSE = np.sqrt(mean_squared_error(gbr_pred,std_test_y1))
RMSE


# In[113]:


gbr_params = {
    "n_estimators" : [15,30,100,600],
    "learning_rate" : [0.01,0.05,0.1],
    "max_depth" : [3,9,14],
    "max_features" : [0.3,0.5,1.0],
    "min_samples_split" : [2,3,4]
  #  "loss" : ["ls","lad","huber","quantile"]
    
}


# In[114]:


gbr_grid = GridSearchCV(GradientBoostingRegressor(random_state = 33),gbr_params,cv=3,n_jobs=-1,verbose = 1)


# In[115]:


gbr_grid.fit(std_train_X1,std_train_y1)


# In[116]:


gbr_grid.best_params_
gbr_grid.best_score_


# In[117]:


#make GradientBoostingRegressor with best params
gbr_best = GradientBoostingRegressor(
                             learning_rate = 0.01,max_depth = 3,max_features = 0.5,min_samples_split = 4,n_estimators = 600,
                                    random_state=133)


# In[118]:


#predict with test data and get score
#train dataset
gbr_best.fit(std_train_X1,std_train_y1)
best_pred =gbr_best.predict(std_train_X1)
mse = mean_squared_error(best_pred,std_train_y1)
print("------------std_train_mse with GradientBoostingRegressor---------------")
mse
print("------------std_train_score with GradientBoostingRegressor---------------")
gbr_best.score(std_train_X1,std_train_y1)

#valid dataset
gbr_best.fit(std_valid_X1,std_valid_y1)
best_valid_pred = gbr_best.predict(std_valid_X1)
valid_mse = mean_squared_error(best_valid_pred,std_valid_y1)
print("------------std_valid_mse with GradientBoostingRegressor---------------")
valid_mse
print("------------std_valid_score with GradientBoostingRegressor---------------")
gbr_best.score(std_valid_X1,std_valid_y1)

#test dataset
gbr_best.fit(std_test_X1,std_test_y1)
best_test_pred = gbr_best.predict(std_test_X1)
test_mse = mean_squared_error(best_test_pred,std_test_y1)
print("------------std_test_mse with GradientBoostingRegressor---------------")
test_mse
print("------------std_test_root_mse with GradientBoostingRegressor---------------")
rmse = np.sqrt(test_mse)
rmse
print("------------std_test_score with GradientBoostingRegressor---------------")
gbr_best.score(std_test_X1,std_test_y1)


# In[119]:


#draw feature importance horizontal bar graph
feature_importance = gbr_best.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(38, 19))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(std_data.columns)[sorted_idx])
plt.title('Feature Importance (PM10)')


# In[120]:


#for target2 ( PM2.5)
tree_grid.fit(std_train_X2,std_train_y2)
tree_grid.best_score_
tree_grid.best_params_


# In[121]:


#make decision tree with best params
best_tree = DecisionTreeRegressor(criterion="mae",max_depth=6,max_features="auto",splitter="best",random_state=1212)
#predict with test data and get score
#train dataset
best_tree.fit(std_train_X2,std_train_y2)
best_pred = best_tree.predict(std_train_X2)
mse = mean_squared_error(best_pred,std_train_y2)
print("------------std_train_mse---------------")
mse
print("------------std_train_score---------------")
best_tree.score(std_train_X2,std_train_y2)

#valid dataset
best_tree.fit(std_valid_X2,std_valid_y2)
best_valid_pred = best_tree.predict(std_valid_X2)
valid_mse = mean_squared_error(best_valid_pred,std_valid_y2)
print("------------std_valid_mse---------------")
valid_mse
print("------------std_valid_score---------------")
best_tree.score(std_valid_X2,std_valid_y2)

#test dataset
best_tree.fit(std_test_X2,std_test_y2)
best_test_pred = best_tree.predict(std_test_X2)
test_mse = mean_squared_error(best_test_pred,std_test_y2)
print("------------std_test_mse---------------")
test_mse
print("------------std_test_root_mse---------------")
rmse = np.sqrt(test_mse)
rmse
print("------------std_test_score---------------")
best_tree.score(std_test_X2,std_test_y2)


# In[122]:


#gradient boosting with std scaled dataset
gbr = GradientBoostingRegressor(random_state = 131).fit(std_train_X2,std_train_y2)
gbr_pred = gbr.predict(std_test_X2)
gbr.score(std_test_X2,std_test_y2)
RMSE = np.sqrt(mean_squared_error(gbr_pred,std_test_y2))
RMSE
gbr_grid = GridSearchCV(GradientBoostingRegressor(random_state = 331),gbr_params,cv=3,n_jobs=-1,verbose = 1)
gbr_grid.fit(std_train_X2,std_train_y2)
gbr_grid.best_params_
gbr_grid.best_score_


# In[123]:


#make GradientBoostingRegressor with best params
gbr_best = GradientBoostingRegressor(
                             learning_rate = 0.01,max_depth = 3,max_features = 0.5,min_samples_split = 2,n_estimators = 600,
                                    random_state=1331)
#predict with test data and get score
#train dataset
gbr_best.fit(std_train_X2,std_train_y2)
best_pred =gbr_best.predict(std_train_X2)
mse = mean_squared_error(best_pred,std_train_y2)
print("------------std_train_mse with GradientBoostingRegressor---------------")
mse
print("------------std_train_score with GradientBoostingRegressor---------------")
gbr_best.score(std_train_X2,std_train_y2)

#valid dataset
gbr_best.fit(std_valid_X2,std_valid_y2)
best_valid_pred = gbr_best.predict(std_valid_X2)
valid_mse = mean_squared_error(best_valid_pred,std_valid_y2)
print("------------std_valid_mse with GradientBoostingRegressor---------------")
valid_mse
print("------------std_valid_score with GradientBoostingRegressor---------------")
gbr_best.score(std_valid_X2,std_valid_y2)

#test dataset
gbr_best.fit(std_test_X2,std_test_y2)
best_test_pred = gbr_best.predict(std_test_X2)
test_mse = mean_squared_error(best_test_pred,std_test_y2)
print("------------std_test_mse with GradientBoostingRegressor---------------")
test_mse
print("------------std_test_root_mse with GradientBoostingRegressor---------------")
rmse = np.sqrt(test_mse)
rmse
print("------------std_test_score with GradientBoostingRegressor---------------")

#draw feature importance horizontal bar graph
gbr_best.score(std_test_X2,std_test_y2)
feature_importance = gbr_best.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(38, 19))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(std_data.columns)[sorted_idx])
plt.title('Feature Importance (PM2.5)')


# In[ ]:




