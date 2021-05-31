#!/usr/bin/env python
# coding: utf-8

# In[302]:


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


# In[2]:


class preprocessing :
    #make linear regression model and fit, return linear regression model and score
    #df : dataframe
    #columns : columns list for make linear regression model
    #target : target feature name
    #test : train test split's test_size parameter
    #random= : train test split's random_state parameter
    #return : linear regression model and score
    def reg_score(self,df,columns,target,test = 0.2,random=0):
        t_col = columns.copy()
        t_col.append(target)
        data = df[t_col]
        clean_df = data.dropna(how="any")
        train_X, test_X, train_y, test_y = train_test_split(
            clean_df[columns],clean_df[target],test_size=test,random_state=random)
        linear = LinearRegression().fit(train_X,train_y)
        return linear,linear.score(test_X,test_y)


def makeDecisionTree(citerion, x_train, y_train,depthNum):
    if depthNum>0:
        clf = tree.DecisionTreeClassifier(criterion=citerion, max_depth=depthNum)
    else:
        clf = tree.DecisionTreeClassifier(criterion=citerion)

    clf.fit(x_train, y_train)

    return clf

def printTreeGraph(model):
    dot_data = tree.export_graphviz(model,
                                    out_file=None,
                                    feature_names=feature_name,
                                    class_names=target_name,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    print(graph)


def foldValidatation(model,train_x,train_y,foldNum):
    cv_scores=cross_val_score(model,train_x,train_y,cv=foldNum)
    print(cv_scores)
    print("cv_scores mean: {}".format(np.mean(cv_scores)))
    # print(cross_validate(model, train_x, train_y, scoring=['accuracy', 'roc_auc'], return_train_score=True))

def treeGridSearch(x_train,y_train):
    tree_params = {
        "criterion": ["gini","entropy"],
        "splitter": ["best", "random"],
        "max_depth": [2, 6, 10, 14],
        "max_features": ["auto", "sqrt", "log2", 5, None]
    }
    tree_grid = GridSearchCV(tree.DecisionTreeClassifier(random_state=12),
                             tree_params,
                             scoring="neg_mean_squared_error",
                             verbose=1,
                             n_jobs=-1)
    tree_grid.fit(x_train, y_train)
    best_param=tree_grid.best_params_
    print("Best parameter(Decision Tree) : ")
    print(best_param)
    print("Best score(Decision Tree) : ")
    print(tree_grid.best_score_)

def baggingGridSearch(best_tree):
    bagging_params = {
        "n_estimators": [50, 100, 150],
        "max_samples": [0.25, 0.5, 1.0],
        "max_features": [0.25, 0.5, 1.0],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False]
    }
    best_bagging = GridSearchCV(BaggingClassifier(best_tree),
                                   bagging_params,
                                   cv=4,
                                   verbose=1,
                                   n_jobs=-1)

    best_bagging.fit(train_x, train_y)
    print("Best parameter(Bagging) : ")
    print(best_bagging.best_params_)
    print("Best score(Bagging) : ")
    print(best_bagging.best_score_)

    print(foldValidatation(best_bagging, test_x, test_y, 3))

    # predict with train data and get score
    best_bagging.fit(train_x, train_y)
    best_pred = best_bagging.predict(train_x)

    print("------------train_score---------------")
    print(best_bagging.score(train_x, train_y))

    # predict with valid data and get score
    best_bagging.fit(valid_x, valid_y)
    best_pred = best_bagging.predict(valid_x)
    print("------------std_train_score---------------")
    print(best_bagging.score(valid_x, valid_y))

    # predict with test data and get score
    best_bagging.fit(test_x, test_y)
    best_pred = best_bagging.predict(test_x)
    print("------------std_test_score---------------")
    print(best_bagging.score(test_x, test_y))


def ordinalEncode_category(arr,str):
    enc=preprocessing.OrdinalEncoder()
    encodedData=enc.fit_transform(arr[[str]])
    column_name = str + " ordinalEncoded"
    arr[column_name] = encodedData
    return arr
    


# In[4]:


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


# In[5]:


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


# In[6]:


#change year, month, day, hour columns to date format 
tt = pd.DataFrame(pd.date_range("2013-03-01","2017-03-01",freq="h"))
tt = tt[:-1]
tt = pd.concat([tt,tt,tt,tt,tt,tt,tt,tt,tt,tt,tt,tt],ignore_index = True)
beijing["Date"] = tt
#drop year, month, day, hour columns 
beijing.drop(columns=["year","month","day","hour"],inplace=True)


# In[7]:


beijing


# In[8]:


beijing.info()
seoul.info()
weather.info()


# In[9]:


weather
seoul


# In[10]:


#change MSRDT column to date format 
date_s = seoul["MSRDT"].astype(str)
seoul["MSRDT"] = date_s.apply(lambda x: datetime.strptime(x, "%Y%m%d%H%M"))
seoul


# In[11]:


weather["tm"] = weather["tm"].apply(lambda _: datetime.strptime(_,"%Y-%m-%d"))
weather.info()


# In[12]:


#*Data value changes
#find nan values
seoul.isnull().sum()
weather.isnull().sum()
beijing.isnull().sum()


# In[13]:


#weather - sumRn이 nan value이면 비가 오지 않은 날이므로 0.0으로 fill
weather["sumRn"].fillna(0.0,inplace=True)


# In[14]:


weather.isnull().sum()


# In[15]:


#날씨 결측값을 가장 큰 양의 상관관계를 갖는 column을 파악한 뒤 regression으로 모델 학습 후 예측값으로 채우기
#0.5 정도의 값이 얻어지면 : 강력한 양(+)의 상관. 변인 x 가 증가하면 변인 y 가 증가한다.
weather.corr()


# In[16]:


#corr() heatmap
sns.heatmap(weather.corr(), annot=True, fmt='0.2f')
plt.title('weather', fontsize=20)
plt.show()


# In[17]:


#sumGsr과 avgTs는 0.5,sumGsr과 minRhm는 -0.55 sumGsr과 avgTs는 0.4
#sumGsr-avgTs --> accuray = 0.2..
#sumGsr-avgTsm,minRhm --> accuray = 0.6
#sumGsr-avgTsm,minRhm,avgTa --> accuracy = 0.72
#linear regression 
linear,score = preprocessing().reg_score(weather,["avgTs","minRhm","avgTa"],"sumGsr")
score
pred_y = linear.predict(weather.loc[:, ["avgTs","minRhm","avgTa"]])
weather['sumGsr'].fillna(pd.Series(pred_y), inplace=True)


# In[18]:


weather.isnull().sum()


# In[19]:


#// maxWd와 sumGsr은 0.2,
#0.2 정도의 값이 얻어지면 : 너무 약해서 의심스러운 양(+)의 상관
# maxWd는 다른 feature들과도 유의미한 상관관계가 나타나지 않으므로 maxWd는 전날 데이터와 다음날의 데이터로 채운다. 
weather[weather["maxWd"].isnull() == True]


# In[20]:


weather["maxWd"].fillna(method='ffill', limit=1,inplace=True)


# In[21]:


weather[weather["maxWd"].isnull() == True]


# In[22]:


weather["maxWd"].fillna(method='bfill', limit=1,inplace=True)


# In[23]:


#nan value 제거 완료
weather.isnull().sum()


# In[24]:


#Aotizhongxin의 결측값
beijing[beijing["station"]=="Aotizhongxin"].isnull().sum()


# In[25]:


#beijing 데이터의 총 결측값
beijing.isnull().sum()


# In[26]:


#weather와 같은 과정 진행
beijing.corr()


# In[27]:


sns.heatmap(beijing.corr(), annot=True, fmt='0.2f')
plt.title('Beijing', fontsize=20)
plt.show()


# In[28]:


#categorical value를 encoding, 해당 데이터의 경우 labelencoding보다 onehotencoding이 더 적합
#onehotencoding for station, wd columns
ohe = OneHotEncoder()
station_oh = ohe.fit_transform(beijing["station"].values.reshape(-1,1)).toarray()
beijing_oh = beijing.copy()
station_oh = pd.DataFrame(station_oh,columns=ohe.categories_[0])


# In[29]:


station_oh


# In[30]:


beijing_oh.drop(columns="station",inplace=True)


# In[31]:


beijing_oh = pd.concat([beijing_oh,station_oh],axis=1)


# In[32]:


beijing_oh


# In[33]:


#풍향은 categorical value, numerical value로 변환해야 corr 파악 후 예측할 수 있는데 nan값이 있기 때문에 처리가 난해..
#풍향은 unique를 통해 max 값으로 채운다.
beijing_oh["wd"].fillna(beijing_oh["wd"].value_counts().index[beijing_oh["wd"].value_counts().argmax()],inplace=True)


# In[34]:


beijing_oh["wd"].isnull().sum()


# In[35]:


#nan value를 채운 후 onehotencoding 
wd_oh = ohe.fit_transform(beijing_oh["wd"].values.reshape(-1,1)).toarray()
wd_oh = pd.DataFrame(wd_oh,columns=ohe.categories_[0])
beijing_oh = pd.concat([beijing_oh,wd_oh],axis=1)
beijing_oh.drop(columns="wd",inplace=True)


# In[36]:


beijing_oh.columns


# In[37]:


beijing_oh


# In[38]:


#encoding 후 nan value 다시 측정 
beijing_oh.isnull().sum()


# In[39]:


#weather와 같은 과정으로 결측값 채우기 진행
#beijing 데이터의 상관관계 파악
beijing.corr()


# In[40]:


sns.heatmap(beijing.corr(), annot=True, fmt='0.2f')
plt.title('Beijing', fontsize=20)
plt.show()


# In[41]:


#predict PM2.5 - using PM10, NO2, CO 
linear,score = preprocessing().reg_score(beijing_oh,["PM10","SO2","NO2","CO"],"PM2.5",0.2,1)
score
pred_y = linear.predict(beijing_oh[["PM10","SO2","NO2","CO"]].dropna(how="any"))
beijing_oh["PM2.5"].fillna(pd.Series(pred_y), inplace=True)


# In[42]:


beijing_oh.isnull().sum()#667개의 예측되지 않은 값들


# In[43]:


#predict PM10 -also using PM2.5,SO2 NO2, CO
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["PM2.5","SO2","NO2","CO"],"PM10",0.2,2)
score
pred_y = linear.predict(beijing_oh[["PM2.5","SO2","NO2","CO"]].dropna(how="any"))
beijing_oh["PM10"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum()  #478개의 예측되지 않은 값들


# In[44]:


#predict SO2 -also using PM2.5, PM10, NO2, CO
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["PM2.5","PM10","NO2","CO"],"SO2",0.2,3)
score
#낮은 score.. SO2는 regression 대신 median으로 nan 값 채우기
#std_beijing["SO2"].fillna(std_beijing["SO2"].median(), inplace=True)
#std_beijing.isnull().sum() 
beijing_oh["SO2"].fillna(beijing_oh["SO2"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[45]:


#predict NO2 -also using PM2.5,PM10,SO2,CO
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["PM2.5","PM10","SO2","CO"],"NO2",0.2,4)
score
#낮은 score.. NO2는 regression 대신 median으로 nan 값 채우기
beijing_oh["NO2"].fillna(beijing_oh["NO2"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[46]:


#predict CO -also using PM2.5,PM10,SO2,NO2
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["PM2.5","PM10","SO2","NO2"],"CO",0.2,5)
score
pred_y = linear.predict(beijing_oh[["PM2.5","PM10","SO2","NO2"]].dropna(how="any"))
beijing_oh["CO"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum()  #901개의 예측되지 않은 값들


# In[47]:


#predict O3 - using NO2,TEMP,PRES
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["NO2","TEMP","PRES"],"O3",0.2,6)
score
#낮은 score.. O3는 regression 대신 median으로 nan 값 채우기
beijing_oh["O3"].fillna(beijing_oh["O3"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[48]:


#predict TEMP - using O3,PRES,DEWP
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["O3","PRES","DEWP"],"TEMP",0.2,7)
score
pred_y = linear.predict(beijing_oh[["O3","PRES","DEWP"]].dropna(how="any"))
beijing_oh["TEMP"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[49]:


#predict PRES - using O3,TEMP,DEWP
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["O3","TEMP","DEWP"],"PRES",0.2,8)
score
pred_y = linear.predict(beijing_oh[["O3","TEMP","DEWP"]].dropna(how="any"))
beijing_oh["PRES"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[50]:


#predict DEWP - using TEMP,PRES
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["TEMP","PRES"],"DEWP",0.2,9)
score
pred_y = linear.predict(beijing_oh[["TEMP","PRES"]].dropna(how="any"))
beijing_oh["DEWP"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[51]:


#predict RAIN 상관관계가 다 낮음,  ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["RAIN"].isnull() == True]
beijing_oh["RAIN"].fillna(method='ffill', limit=1,inplace=True)


# In[52]:


beijing_oh["RAIN"].fillna(method='bfill', limit=1,inplace=True)


# In[53]:


beijing_oh[beijing_oh["RAIN"].isnull() == True]


# In[54]:


#RAIN과 WSPM - 지리적 위치가 가까운 구역을 참조하여 nan value 채우기
#https://link.springer.com/article/10.1007/s00521-018-3532-z/figures/1
#dingling changping rain 결측값 51씩 wspm 43씩
#dongsi tiantan guanyuan wanshouxigong  nongzhanguan aotizhongxin wanliu rain 결측값 20씩 wspm 14씩
#shunyi huairou rain 결측값 51,55씩 wspm 44,49
# gucheng rain 결측값 43씩 wspm 42씩

##다시 분류하면 
#dongsi tiantan guanyuan wanshouxigong  nongzhanguan aotizhongxin wanliu rain 결측값 20씩 wspm 14씩
#shunyi huairou  gucheng dingling changping
#지리적 위치가 가까운 구역을 참조하여 nan value 채우기 불가능..


# In[55]:


#연속된 nan value는 median으로 채우기, median = 0.0..
beijing_oh["RAIN"].fillna(beijing_oh["RAIN"].median(),inplace=True)


# In[56]:


#precit WSPM 상관관계가 다 낮음, ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["WSPM"].isnull() == True]
beijing_oh["WSPM"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["WSPM"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["WSPM"].isnull() == True]


# In[57]:


#연속된 nan value는 median으로 채우기, median = 1.4..
beijing_oh["WSPM"].fillna(beijing_oh["WSPM"].median(),inplace=True)


# In[58]:


beijing_oh.isnull().sum() 


# In[59]:


#남은 nan value - PM2.5 667, PM10 478,CO 901
beijing_oh[beijing_oh["PM2.5"].isnull() == True]
beijing_oh[beijing_oh["PM10"].isnull() == True]
beijing_oh[beijing_oh["CO"].isnull() == True]


# In[60]:


#ffil,limit=1 and bfil,limit=1으로 채우기
beijing_oh[beijing_oh["PM2.5"].isnull() == True]
beijing_oh["PM2.5"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["PM2.5"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["PM2.5"].isnull() == True]


# In[61]:


#연속된 nan value는 median으로 채우기, median = 1.4..
beijing_oh["PM2.5"].fillna(beijing_oh["PM2.5"].median(),inplace=True)


# In[62]:


#ffil,limit=1 and bfil,limit=1으로 채우기
beijing_oh[beijing_oh["PM10"].isnull() == True]
beijing_oh["PM10"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["PM10"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["PM10"].isnull() == True]
#연속된 nan value는 median으로 채우기, median = 82.0..
beijing_oh["PM10"].fillna(beijing_oh["PM10"].median(),inplace=True)


# In[63]:


#ffil,limit=1 and bfil,limit=1으로 채우기
beijing_oh[beijing_oh["CO"].isnull() == True]
beijing_oh["CO"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["CO"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["CO"].isnull() == True]
#연속된 nan value는 median으로 채우기, median = 900..
beijing_oh["CO"].fillna(beijing_oh["CO"].median(),inplace=True)


# In[64]:


#결측값 채우기 완료
beijing_oh.isnull().sum()


# In[65]:


clean_beijing = beijing_oh.copy()


# In[66]:


clean_beijing.columns


# In[67]:


std_beijing = pd.DataFrame(StandardScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)
mm_beijing = pd.DataFrame(MinMaxScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)
ma_beijing = pd.DataFrame(MaxAbsScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)
rb_beijing = pd.DataFrame(RobustScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)


# In[68]:


std_beijing = pd.concat([std_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
mm_beijing = pd.concat([mm_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
ma_beijing = pd.concat([ma_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
rb_beijing = pd.concat([rb_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
std_beijing.columns=clean_beijing.columns
mm_beijing.columns=clean_beijing.columns
ma_beijing.columns=clean_beijing.columns
rb_beijing.columns=clean_beijing.columns


# In[69]:


clean_beijing
std_beijing
mm_beijing
ma_beijing
rb_beijing


# In[70]:


std_weather = pd.DataFrame(StandardScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
std_weather["tm"]=weather["tm"]


# In[71]:


mm_weather = pd.DataFrame(MinMaxScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
mm_weather["tm"]=weather["tm"]


# In[72]:


ma_weather = pd.DataFrame(MaxAbsScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
ma_weather["tm"]=weather["tm"]


# In[73]:


rb_weather = pd.DataFrame(RobustScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
rb_weather["tm"]=weather["tm"]


# In[74]:


std_weather
mm_weather
ma_weather
rb_weather


# In[75]:


#서울 데이터 onehotencoding for MSRSTE_NM
ohe = OneHotEncoder()
Nm_oh = ohe.fit_transform(seoul["MSRSTE_NM"].values.reshape(-1,1)).toarray()
seoul_oh = seoul.copy()
Nm_oh = pd.DataFrame(Nm_oh,columns=ohe.categories_[0])
Nm_oh
seoul_oh.drop(columns="MSRSTE_NM",inplace=True)
seoul_oh = pd.concat([seoul_oh,Nm_oh],axis=1)


# In[76]:


seoul_oh["영등포구"]


# In[77]:


seoul_oh


# In[78]:


std_seoul = pd.DataFrame(StandardScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)
mm_seoul = pd.DataFrame(MinMaxScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)
ma_seoul = pd.DataFrame(MaxAbsScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)
rb_seoul = pd.DataFrame(RobustScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)


# In[79]:


col = seoul_oh.iloc[:,1:7].columns.append(seoul_oh.drop(columns=std_seoul.columns).columns)


# In[80]:


std_seoul = pd.concat([std_seoul,seoul_oh.drop(columns=std_seoul.columns)],axis=1,ignore_index = True)
mm_seoul = pd.concat([mm_seoul,seoul_oh.drop(columns=mm_seoul.columns)],axis=1,ignore_index = True)
ma_seoul = pd.concat([ma_seoul,seoul_oh.drop(columns=ma_seoul.columns)],axis=1,ignore_index = True)
rb_seoul = pd.concat([rb_seoul,seoul_oh.drop(columns=rb_seoul.columns)],axis=1,ignore_index = True)
std_seoul.columns=col
mm_seoul.columns=col
ma_seoul.columns=col
rb_seoul.columns=col


# In[81]:


std_seoul
mm_seoul
ma_seoul
rb_seoul


# In[275]:





# In[ ]:


#예측 tree 생성, 미세먼지 예측에 가장 큰 중요도를 가지는 feature를 분석.. 
#서울시 데이터 -> 날짜별 group by, 기상 데이터와 같이 1461행으로 줄임.
#베이징 데이터 -> 같은 작업 진행


# In[84]:


std_weather
std_seoul
std_beijing


# In[102]:


std_seoul_group = std_seoul.groupby(std_seoul['MSRDT'].dt.date).mean()
std_seoul_group = std_seoul_tree.iloc[:,:6]


# In[103]:


std_seoul_group


# In[269]:


std_beijing_group = std_beijing.iloc[:,:12].groupby(std_beijing['Date'].dt.date).mean()
#베이징 데이터의 경우 풍향 데이터 분석을 위해 최대 풍향 (횟수)를 구하여 data frame을 이어줌. 
std_beijing_wd = std_beijing.iloc[:,24:].groupby(std_beijing['Date'].dt.date).sum().idxmax(axis=1)
std_beijing_wd = pd.DataFrame(std_beijing_wd,columns = ["WD"])


# In[270]:


std_beijing_group = pd.concat([std_beijing_group,std_beijing_wd,],axis=1)


# In[271]:


#풍향 데이터 onehotencoding 
ohe = OneHotEncoder()
group_wd_oh = ohe.fit_transform(std_beijing_group["WD"].values.reshape(-1,1)).toarray()
group_wd_oh = pd.DataFrame(group_wd_oh,index =std_beijing_group.index , columns=ohe.categories_[0])
std_beijing_group = pd.concat([std_beijing_group,group_wd_oh],axis = 1)


# In[272]:


std_beijing_group.drop(columns="WD",inplace = True)


# In[273]:


std_beijing_group


# In[274]:


std_weather.index = std_weather["tm"]
std_weather.drop(columns="tm",inplace= True)


# In[275]:


std_weather


# In[276]:


#서울 기상 데이터 + 대기오염 데이터
std_tree = pd.concat([std_weather,std_seoul_group],axis = 1)


# In[277]:


std_tree


# In[278]:


#column명 변경 후 데이터 합치기 
std_beijing_group = std_beijing_group.add_prefix("beijing_")


# In[280]:


std_tree = pd.concat([std_beijing_group,std_tree],axis = 1)


# In[281]:


std_tree


# In[283]:


std_target1 = std_tree["PM10"]
std_target2 = std_tree["PM25"]
std_data = std_tree.drop(columns=["PM10","PM25"])


# In[287]:


#split train , test data for performance
std_train_X1,std_test_X1,std_train_y1,std_test_y1 = train_test_split(std_data,std_target1,test_size=0.2,random_state=1)
#split train , valid data for learning 
std_train_X1,std_valid_X1,std_train_y1,std_valid_y1 = train_test_split(std_train_X1,std_train_y1,test_size = 0.3,random_state = 11)


# In[288]:


#split train , test data for performance
std_train_X2,std_test_X2,std_train_y2,std_test_y2 = train_test_split(std_data,std_target2,test_size=0.2,random_state=2)
#split train , valid data for learning 
std_train_X2,std_valid_X2,std_train_y2,std_valid_y2 = train_test_split(std_train_X2,std_train_y2,test_size = 0.3,random_state = 21)


# In[289]:


tree_params = {
    "criterion":["mse", "friedman_mse", "mae", "poisson"],
    "splitter" : ["best","random"],
    "max_depth" : [2,6,10,14],
    "max_features" : ["auto","sqrt","log2",10,None]
}


# In[292]:


#decision tree with gridsearchcv
tree_grid = GridSearchCV(DecisionTreeRegressor(random_state = 122),
                         tree_params,
                         scoring="neg_mean_squared_error",
                         verbose = 1,
                         n_jobs = -1)


# In[293]:


tree_grid.fit(std_train_X1,std_train_y1)


# In[295]:


tree_grid.best_score_
tree_grid.best_params_


# In[296]:


#make decision tree with best params
best_tree = DecisionTreeRegressor(criterion="mae",max_depth=6,max_features="auto",splitter="best",random_state=121)


# In[299]:


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


# In[303]:


#gradient boosting with std scaled dataset
gbr = GradientBoostingRegressor(random_state = 13).fit(std_train_X1,std_train_y1)
gbr_pred = gbr.predict(std_test_X1)
gbr.score(std_test_X1,std_test_y1)
RMSE = np.sqrt(mean_squared_error(gbr_pred,std_test_y1))
RMSE


# In[304]:


gbr_params = {
    "n_estimators" : [15,30,100,600],
    "learning_rate" : [0.01,0.05,0.1],
    "max_depth" : [3,9,14],
    "max_features" : [0.3,0.5,1.0],
    "min_samples_split" : [2,3,4]
  #  "loss" : ["ls","lad","huber","quantile"]
    
}


# In[ ]:


#gradient boosting의 장점은 boosted tree가 완성된 후, 
#feature의 중요도 스코어를 내는 것이 상대적으로 쉽다는 것이다. 
#어떠한 변수가 decision tree에서 "중요한 결정" 을 내리는데 사용된다면 그 변수의 중요도는 높을 것이다. 
#중요도는 변수별로 계산되며, 각각의 변수는 중요도 랭킹에 따라 정렬될 수 있다. 
#어떠한 변수의 중요도는 하나의 decision tree에서 그 변수로 인해 performance measure를 증가하는 양으로 계산된다.
#출처: https://3months.tistory.com/169 [Deep Play]


# In[305]:


gbr_grid = GridSearchCV(GradientBoostingRegressor(random_state = 33),gbr_params,cv=3,n_jobs=-1,verbose = 1)


# In[306]:


gbr_grid.fit(std_train_X1,std_train_y1)


# In[308]:


gbr_grid.best_params_
gbr_grid.best_score_


# In[309]:


#make GradientBoostingRegressor with best params
gbr_best = GradientBoostingRegressor(
                             learning_rate = 0.01,max_depth = 3,max_features = 0.5,min_samples_split = 4,n_estimators = 600,
                                    random_state=133)


# In[310]:


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


# In[321]:


feature_importance = gbr_best.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(std_data.columns)[sorted_idx])
plt.title('Feature Importance (PM10)')

#result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                #random_state=42, n_jobs=2)
#sorted_idx = result.importances_mean.argsort()
#plt.subplot(1, 2, 2)
#plt.boxplot(result.importances[sorted_idx].T,
           # vert=False, labels=np.array(diabetes.feature_names)[sorted_idx])
#plt.title("Permutation Importance (test set)")
#fig.tight_layout()
#plt.show()


# In[ ]:


#변수 중요도는 "해당 변수가 상대적으로 얼마만큼 종속변수에 영향을 주는가?"에 대한 척도

