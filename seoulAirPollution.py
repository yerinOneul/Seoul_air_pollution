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


# In[101]:


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
    


# In[2]:


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


# In[3]:


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


# In[4]:


#change year, month, day, hour columns to date format 
tt = pd.DataFrame(pd.date_range("2013-03-01","2017-03-01",freq="h"))
tt = tt[:-1]
tt = pd.concat([tt,tt,tt,tt,tt,tt,tt,tt,tt,tt,tt,tt],ignore_index = True)
beijing["Date"] = tt
#drop year, month, day, hour columns 
beijing.drop(columns=["year","month","day","hour"],inplace=True)


# In[5]:


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


# In[12]:


weather["tm"] = weather["tm"].apply(lambda _: datetime.strptime(_,"%Y-%m-%d"))
weather.info()


# In[20]:


#*Data value changes
#find nan values
seoul.isnull().sum()
weather.isnull().sum()
beijing.isnull().sum()


# In[14]:


#weather - sumRn이 nan value이면 비가 오지 않은 날이므로 0.0으로 fill
weather["sumRn"].fillna(0.0,inplace=True)


# In[15]:


weather.isnull().sum()


# In[16]:


#날씨 결측값을 가장 큰 양의 상관관계를 갖는 column을 파악한 뒤 regression으로 모델 학습 후 예측값으로 채우기
#0.5 정도의 값이 얻어지면 : 강력한 양(+)의 상관. 변인 x 가 증가하면 변인 y 가 증가한다.
weather.corr()


# In[17]:


#corr() heatmap
sns.heatmap(weather.corr(), annot=True, fmt='0.2f')
plt.title('weather', fontsize=20)
plt.show()


# In[104]:


#sumGsr과 avgTs는 0.5,sumGsr과 minRhm는 -0.55 sumGsr과 avgTs는 0.4
#sumGsr-avgTs --> accuray = 0.2..
#sumGsr-avgTsm,minRhm --> accuray = 0.6
#sumGsr-avgTsm,minRhm,avgTa --> accuracy = 0.72
#linear regression 
linear,score = preprocessing().reg_score(weather,["avgTs","minRhm","avgTa"],"sumGsr")
score
pred_y = linear.predict(weather.loc[:, ["avgTs","minRhm","avgTa"]])
weather['sumGsr'].fillna(pd.Series(pred_y), inplace=True)


# In[105]:


weather.isnull().sum()


# In[20]:


#// maxWd와 sumGsr은 0.2,
#0.2 정도의 값이 얻어지면 : 너무 약해서 의심스러운 양(+)의 상관
# maxWd는 다른 feature들과도 유의미한 상관관계가 나타나지 않으므로 maxWd는 전날 데이터와 다음날의 데이터로 채운다. 
weather[weather["maxWd"].isnull() == True]


# In[21]:


weather["maxWd"].fillna(method='ffill', limit=1,inplace=True)


# In[22]:


weather[weather["maxWd"].isnull() == True]


# In[23]:


weather["maxWd"].fillna(method='bfill', limit=1,inplace=True)


# In[24]:


#nan value 제거 완료
weather.isnull().sum()


# In[137]:


#Aotizhongxin의 결측값
beijing[beijing["station"]=="Aotizhongxin"].isnull().sum()


# In[38]:


#beijing 데이터의 총 결측값
beijing.isnull().sum()


# In[39]:


#weather와 같은 과정 진행
beijing.corr()


# In[25]:


sns.heatmap(beijing.corr(), annot=True, fmt='0.2f')
plt.title('Beijing', fontsize=20)
plt.show()


# In[26]:


#categorical value를 encoding, 해당 데이터의 경우 labelencoding보다 onehotencoding이 더 적합
#onehotencoding for station, wd columns
ohe = OneHotEncoder()
station_oh = ohe.fit_transform(beijing["station"].values.reshape(-1,1)).toarray()
beijing_oh = beijing.copy()
station_oh = pd.DataFrame(station_oh,columns=beijing["station"].unique())


# In[27]:


station_oh


# In[28]:


beijing_oh.drop(columns="station",inplace=True)


# In[29]:


beijing_oh = pd.concat([beijing_oh,station_oh],axis=1)


# In[30]:


#풍향은 categorical value, numerical value로 변환해야 corr 파악 후 예측할 수 있는데 nan값이 있기 때문에 처리가 난해..
#풍향은 unique를 통해 max 값으로 채운다.
beijing_oh["wd"].fillna(beijing_oh["wd"].value_counts().index[beijing_oh["wd"].value_counts().argmax()],inplace=True)


# In[31]:


beijing_oh["wd"].isnull().sum()


# In[32]:


#nan value를 채운 후 onehotencoding 
wd_oh = ohe.fit_transform(beijing_oh["wd"].values.reshape(-1,1)).toarray()
wd_oh = pd.DataFrame(wd_oh,columns=beijing_oh["wd"].unique())
beijing_oh = pd.concat([beijing_oh,wd_oh],axis=1)
beijing_oh.drop(columns="wd",inplace=True)


# In[33]:


beijing_oh.columns


# In[34]:


#encoding 후 nan value 다시 측정 
beijing_oh.isnull().sum()


# In[35]:


#weather와 같은 과정으로 결측값 채우기 진행
#beijing 데이터의 상관관계 파악
beijing.corr()


# In[36]:


sns.heatmap(beijing.corr(), annot=True, fmt='0.2f')
plt.title('Beijing', fontsize=20)
plt.show()


# In[111]:


#predict PM2.5 - using PM10, NO2, CO 
linear,score = preprocessing().reg_score(beijing_oh,["PM10","SO2","NO2","CO"],"PM2.5",0.2,1)
score
pred_y = linear.predict(beijing_oh[["PM10","SO2","NO2","CO"]].dropna(how="any"))
beijing_oh["PM2.5"].fillna(pd.Series(pred_y), inplace=True)


# In[112]:


beijing_oh.isnull().sum()#667개의 예측되지 않은 값들


# In[116]:


#predict PM10 -also using PM2.5,SO2 NO2, CO
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["PM2.5","SO2","NO2","CO"],"PM10",0.2,2)
score
pred_y = linear.predict(beijing_oh[["PM2.5","SO2","NO2","CO"]].dropna(how="any"))
beijing_oh["PM10"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum()  #478개의 예측되지 않은 값들


# In[119]:


#predict SO2 -also using PM2.5, PM10, NO2, CO
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["PM2.5","PM10","NO2","CO"],"SO2",0.2,3)
score
#낮은 score.. SO2는 regression 대신 median으로 nan 값 채우기
#std_beijing["SO2"].fillna(std_beijing["SO2"].median(), inplace=True)
#std_beijing.isnull().sum() 
beijing_oh["SO2"].fillna(beijing_oh["SO2"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[121]:


#predict NO2 -also using PM2.5,PM10,SO2,CO
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["PM2.5","PM10","SO2","CO"],"NO2",0.2,4)
score
#낮은 score.. NO2는 regression 대신 median으로 nan 값 채우기
beijing_oh["NO2"].fillna(beijing_oh["NO2"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[124]:


#predict CO -also using PM2.5,PM10,SO2,NO2
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["PM2.5","PM10","SO2","NO2"],"CO",0.2,5)
score
pred_y = linear.predict(beijing_oh[["PM2.5","PM10","SO2","NO2"]].dropna(how="any"))
beijing_oh["CO"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum()  #901개의 예측되지 않은 값들


# In[127]:


#predict O3 - using NO2,TEMP,PRES
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["NO2","TEMP","PRES"],"O3",0.2,6)
score
#낮은 score.. O3는 regression 대신 median으로 nan 값 채우기
beijing_oh["O3"].fillna(beijing_oh["O3"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[129]:


#predict TEMP - using O3,PRES,DEWP
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["O3","PRES","DEWP"],"TEMP",0.2,7)
score
pred_y = linear.predict(beijing_oh[["O3","PRES","DEWP"]].dropna(how="any"))
beijing_oh["TEMP"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[131]:


#predict PRES - using O3,TEMP,DEWP
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["O3","TEMP","DEWP"],"PRES",0.2,8)
score
pred_y = linear.predict(beijing_oh[["O3","TEMP","DEWP"]].dropna(how="any"))
beijing_oh["PRES"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[133]:


#predict DEWP - using TEMP,PRES
#linear regression 
linear,score = preprocessing().reg_score(beijing_oh,["TEMP","PRES"],"DEWP",0.2,9)
score
pred_y = linear.predict(beijing_oh[["TEMP","PRES"]].dropna(how="any"))
beijing_oh["DEWP"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[134]:


#predict RAIN 상관관계가 다 낮음,  ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["RAIN"].isnull() == True]
beijing_oh["RAIN"].fillna(method='ffill', limit=1,inplace=True)


# In[135]:


beijing_oh["RAIN"].fillna(method='bfill', limit=1,inplace=True)


# In[136]:


beijing_oh[beijing_oh["RAIN"].isnull() == True]


# In[240]:


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


# In[137]:


#연속된 nan value는 median으로 채우기, median = 0.0..
beijing_oh["RAIN"].fillna(beijing_oh["RAIN"].median(),inplace=True)


# In[138]:


#precit WSPM 상관관계가 다 낮음, ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["WSPM"].isnull() == True]
beijing_oh["WSPM"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["WSPM"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["WSPM"].isnull() == True]


# In[139]:


#연속된 nan value는 median으로 채우기, median = 1.4..
beijing_oh["WSPM"].fillna(beijing_oh["WSPM"].median(),inplace=True)


# In[140]:


beijing_oh.isnull().sum() 


# In[141]:


#남은 nan value - PM2.5 667, PM10 478,CO 901
beijing_oh[beijing_oh["PM2.5"].isnull() == True]
beijing_oh[beijing_oh["PM10"].isnull() == True]
beijing_oh[beijing_oh["CO"].isnull() == True]


# In[142]:


#ffil,limit=1 and bfil,limit=1으로 채우기
beijing_oh[beijing_oh["PM2.5"].isnull() == True]
beijing_oh["PM2.5"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["PM2.5"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["PM2.5"].isnull() == True]


# In[143]:


#연속된 nan value는 median으로 채우기, median = 1.4..
beijing_oh["PM2.5"].fillna(beijing_oh["PM2.5"].median(),inplace=True)


# In[144]:


#ffil,limit=1 and bfil,limit=1으로 채우기
beijing_oh[beijing_oh["PM10"].isnull() == True]
beijing_oh["PM10"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["PM10"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["PM10"].isnull() == True]
#연속된 nan value는 median으로 채우기, median = 82.0..
beijing_oh["PM10"].fillna(beijing_oh["PM10"].median(),inplace=True)


# In[145]:


#ffil,limit=1 and bfil,limit=1으로 채우기
beijing_oh[beijing_oh["CO"].isnull() == True]
beijing_oh["CO"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["CO"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["CO"].isnull() == True]
#연속된 nan value는 median으로 채우기, median = 900..
beijing_oh["CO"].fillna(beijing_oh["CO"].median(),inplace=True)


# In[146]:


#결측값 채우기 완료
beijing_oh.isnull().sum()


# In[276]:


clean_beijing = beijing_oh.copy()


# In[280]:


clean_beijing.columns


# In[311]:


std_beijing = pd.DataFrame(StandardScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)
mm_beijing = pd.DataFrame(MinMaxScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)
ma_beijing = pd.DataFrame(MaxAbsScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)
rb_beijing = pd.DataFrame(RobustScaler().fit_transform(clean_beijing.iloc[:,:11]),columns=clean_beijing.iloc[:,:11].columns)


# In[312]:


std_beijing = pd.concat([std_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
mm_beijing = pd.concat([mm_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
ma_beijing = pd.concat([ma_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
rb_beijing = pd.concat([rb_beijing,clean_beijing.iloc[:,11:]],axis=1,ignore_index = True)
std_beijing.columns=clean_beijing.columns
mm_beijing.columns=clean_beijing.columns
ma_beijing.columns=clean_beijing.columns
rb_beijing.columns=clean_beijing.columns


# In[314]:


clean_beijing
std_beijing
mm_beijing
ma_beijing
rb_beijing


# In[154]:


std_weather = pd.DataFrame(StandardScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
std_weather["tm"]=weather["tm"]


# In[158]:


mm_weather = pd.DataFrame(MinMaxScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
mm_weather["tm"]=weather["tm"]


# In[159]:


ma_weather = pd.DataFrame(MaxAbsScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
ma_weather["tm"]=weather["tm"]


# In[160]:


rb_weather = pd.DataFrame(RobustScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
rb_weather["tm"]=weather["tm"]


# In[161]:


std_weather
mm_weather
ma_weather
rb_weather


# In[164]:


#서울 데이터 onehotencoding for MSRSTE_NM
ohe = OneHotEncoder()
Nm_oh = ohe.fit_transform(seoul["MSRSTE_NM"].values.reshape(-1,1)).toarray()
seoul_oh = seoul.copy()
Nm_oh = pd.DataFrame(Nm_oh,columns=seoul["MSRSTE_NM"].unique())
Nm_oh
seoul_oh.drop(columns="MSRSTE_NM",inplace=True)
seoul_oh = pd.concat([seoul_oh,Nm_oh],axis=1)


# In[167]:


seoul_oh


# In[209]:


std_seoul = pd.DataFrame(StandardScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)
mm_seoul = pd.DataFrame(MinMaxScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)
ma_seoul = pd.DataFrame(MaxAbsScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)
rb_seoul = pd.DataFrame(RobustScaler().fit_transform(seoul_oh.iloc[:,1:7]),columns=seoul_oh.iloc[:,1:7].columns)


# In[210]:


col = seoul_oh.iloc[:,1:7].columns.append(seoul_oh.drop(columns=std_seoul.columns).columns)


# In[211]:


std_seoul = pd.concat([std_seoul,seoul_oh.drop(columns=std_seoul.columns)],axis=1,ignore_index = True)
mm_seoul = pd.concat([mm_seoul,seoul_oh.drop(columns=mm_seoul.columns)],axis=1,ignore_index = True)
ma_seoul = pd.concat([ma_seoul,seoul_oh.drop(columns=ma_seoul.columns)],axis=1,ignore_index = True)
rb_seoul = pd.concat([rb_seoul,seoul_oh.drop(columns=rb_seoul.columns)],axis=1,ignore_index = True)
std_seoul.columns=col
mm_seoul.columns=col
ma_seoul.columns=col
rb_seoul.columns=col


# In[214]:


std_seoul
mm_seoul
ma_seoul
rb_seoul


# In[275]:


#데이터 시각화
#*Feature engineering
#outlier**** -  모든 스케일러 처리 전에는 아웃라이어 제거가 선행되어야 한다 -> 대기 오염도 및 날씨는 이상치가 중요, 아웃라이어 제거 없이 스케일러..?
#PCA
#*Data reduction

