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


# In[2]:


class preprocessing :
    #year, month, day, hour column들을 날짜형식으로 변경
    def make_date(year,month,day,hour)
    #시간별 데이터를 날짜별 데이터로 변경
    def hour_to_date()
    


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


#Drop No column and change date format like seoul dataset
beijing.drop(columns="No",inplace = True)

beijing


# In[5]:


tt = pd.DataFrame(pd.date_range("2013-03-01","2017-03-01",freq="h"))


# In[6]:


tt = tt[:-1]


# In[7]:


tt = pd.concat([tt,tt,tt,tt,tt,tt,tt,tt,tt,tt,tt,tt],ignore_index = True)


# In[8]:


beijing["Date"] = tt


# In[9]:


beijing.drop(columns=["year","month","day","hour"],inplace=True)


# In[10]:


beijing


# In[11]:


beijing.info()


# In[12]:


seoul.info()


# In[13]:


weather.info()


# In[14]:


weather


# In[15]:


seoul


# In[16]:


date_s = seoul["MSRDT"].astype(str)
seoul["MSRDT"] = date_s.apply(lambda x: datetime.strptime(x, "%Y%m%d%H%M"))


# In[17]:


seoul


# In[18]:


weather["tm"] = weather["tm"].apply(lambda _: datetime.strptime(_,"%Y-%m-%d"))


# In[19]:


weather["tm"]


# In[20]:


#*Data value changes
#find nan values
seoul.isnull().sum()
weather.isnull().sum()
beijing.isnull().sum()


# In[21]:


#서울 강수량을 0.0으로 fill
weather["sumRn"].fillna(0.0,inplace=True)


# In[22]:


weather.isnull().sum()


# In[23]:


weather  


# In[125]:


#날씨 결측값을 가장 큰 양의 상관관계를 갖는 column을 파악한 뒤 regression으로 모델 학습 후 예측값으로 채우기
#0.5 정도의 값이 얻어지면 : 강력한 양(+)의 상관. 변인 x 가 증가하면 변인 y 가 증가한다.
#std_weather = pd.DataFrame(StandardScaler().fit_transform(weather.iloc[:,1:]),columns=weather.iloc[:,1:].columns)
weather.corr()
#std_weather.corr()


# In[126]:


#sns.heatmap(std_weather.corr(), annot=True, fmt='0.2f')
sns.heatmap(weather.corr(), annot=True, fmt='0.2f')
plt.title('weather', fontsize=20)
plt.show()


# In[127]:


#sumGsr과 avgTs는 0.5,sumGsr과 minRhm는 -0.55 sumGsr과 avgTs는 0.4
#sumGsr-avgTs --> accuray = 0.2..
#sumGsr-avgTsm,minRhm --> accuray = 0.6

#// maxWd와 sumGsr은 0.2
#0.2 정도의 값이 얻어지면 : 너무 약해서 의심스러운 양(+)의 상관
#maxWd는 전날과 다음날의 평균값으로 채운다. 

#df = std_weather[["sumGsr","avgTs","minRhm","avgTa"]]
df = weather[["sumGsr","avgTs","minRhm","avgTa"]]
df = df[df["sumGsr"].notnull()]

#linear regression 
train_X, test_X, train_y, test_y = train_test_split(df[["avgTs","minRhm","avgTa"]],df["sumGsr"],test_size=0.2,random_state=0)


# In[128]:


linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)


# In[129]:


#pred_y = linear.predict(std_weather.loc[:, ["avgTs","minRhm","avgTa"]])
pred_y = linear.predict(weather.loc[:, ["avgTs","minRhm","avgTa"]])


# In[130]:


#std_weather['sumGsr'].fillna(pd.Series(pred_y), inplace=True)
weather['sumGsr'].fillna(pd.Series(pred_y), inplace=True)


# In[131]:


#std_weather.isnull().sum()
weather.isnull().sum()


# In[31]:


#std_weather["Date"] = weather["tm"]


# In[132]:


#std_weather[std_weather["maxWd"].isnull() == True]
weather[weather["maxWd"].isnull() == True]


# In[133]:


#std_weather["maxWd"].fillna(method='ffill', limit=1,inplace=True)
weather["maxWd"].fillna(method='ffill', limit=1,inplace=True)


# In[134]:


#std_weather[std_weather["maxWd"].isnull() == True]
weather[weather["maxWd"].isnull() == True]


# In[135]:


#std_weather["maxWd"].fillna(method='bfill', limit=1,inplace=True)
weather["maxWd"].fillna(method='bfill', limit=1,inplace=True)


# In[136]:


#std_weather.isnull().sum()
weather.isnull().sum()


# In[137]:


beijing[beijing["station"]=="Aotizhongxin"].isnull().sum()


# In[38]:


beijing.isnull().sum()


# In[39]:


beijing.corr()


# In[40]:


beijing


# In[41]:


sns.heatmap(beijing.corr(), annot=True, fmt='0.2f')
plt.title('Beijing', fontsize=20)
plt.show()


# In[42]:


beijing[beijing["PM10"].isnull()]


# In[170]:


#onehotencoding 
ohe = OneHotEncoder()
station_oh = ohe.fit_transform(beijing["station"].values.reshape(-1,1)).toarray()
beijing_oh = beijing.copy()
station_oh = pd.DataFrame(station_oh,columns=beijing["station"].unique())


# In[44]:


station_oh


# In[171]:


beijing_oh.drop(columns="station",inplace=True)


# In[172]:


beijing_oh = pd.concat([beijing_oh,station_oh],axis=1)


# In[173]:


#풍향은 categorical value, numerical value로 변환해야 corr 파악 후 예측할 수 있는데 nan값이 있기 때문에 
#처리가 난해..
#풍향은 unique를 통해 max 값으로 채운다. 

beijing_oh["wd"].fillna(beijing_oh["wd"].value_counts().index[beijing_oh["wd"].value_counts().argmax()],inplace=True)


# In[174]:


beijing_oh["wd"].isnull().sum()


# In[175]:


#onehotencoding 
wd_oh = ohe.fit_transform(beijing_oh["wd"].values.reshape(-1,1)).toarray()
wd_oh = pd.DataFrame(wd_oh,columns=beijing_oh["wd"].unique())


# In[176]:


beijing_oh = pd.concat([beijing_oh,wd_oh],axis=1)
beijing_oh.drop(columns="wd",inplace=True)


# In[161]:


beijing_oh.columns


# In[60]:


beijing_oh.iloc[:,:11]


# In[114]:


#std_beijing = pd.DataFrame(StandardScaler().fit_transform(beijing_oh.iloc[:,:11])
 #                          ,columns =beijing_oh.iloc[:,:11].columns)#날짜 제외 std


# In[139]:


#std_beijing.isnull().sum()
#beijing_oh.isnull().sum()


# In[142]:


#std_beijing.corr()
beijing.corr()


# In[143]:


sns.heatmap(beijing.corr(), annot=True, fmt='0.2f')
plt.title('Beijing', fontsize=20)
plt.show()


# In[177]:


#predict PM2.5 - using PM10, NO2, CO 
#df = std_beijing[["PM2.5","PM10","SO2","NO2","CO"]]
df = beijing_oh[["PM2.5","PM10","SO2","NO2","CO"]]
clean_df = df.dropna(how="any")

#linear regression 
train_X, test_X, train_y, test_y = train_test_split(clean_df[["PM10","SO2","NO2","CO"]],clean_df["PM2.5"],test_size=0.2,random_state=1)

linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)


# In[178]:


pred_y = linear.predict(clean_df[["PM10","SO2","NO2","CO"]])

#std_beijing["PM2.5"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh["PM2.5"].fillna(pd.Series(pred_y), inplace=True)


# In[179]:


#std_beijing.isnull().sum() 
beijing_oh.isnull().sum()#667개의 예측되지 않은 값들


# In[180]:


#predict PM10 -also using PM2.5,SO2 NO2, CO
#linear regression 
train_X, test_X, train_y, test_y = train_test_split(clean_df[["PM2.5","SO2","NO2","CO"]],clean_df["PM10"],test_size=0.2,random_state=2)

linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)

pred_y = linear.predict(clean_df[["PM2.5","SO2","NO2","CO"]])

beijing_oh["PM10"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum()  #478개의 예측되지 않은 값들


# In[181]:


#predict SO2 -also using PM2.5, PM10, NO2, CO
#linear regression 
train_X, test_X, train_y, test_y = train_test_split(clean_df[["PM2.5","PM10","NO2","CO"]],clean_df["SO2"],test_size=0.2,random_state=3)

linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)
#낮은 score.. SO2는 regression 대신 median으로 nan 값 채우기
#std_beijing["SO2"].fillna(std_beijing["SO2"].median(), inplace=True)
#std_beijing.isnull().sum() 
beijing_oh["SO2"].fillna(std_beijing["SO2"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[189]:


#predict NO2 -also using PM2.5,PM10,SO2,CO
#linear regression 
train_X, test_X, train_y, test_y = train_test_split(clean_df[["PM2.5","PM10","SO2","CO"]],clean_df["NO2"],test_size=0.2,random_state=4)

linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)
#낮은 score.. NO2는 regression 대신 median으로 nan 값 채우기
beijing_oh["NO2"].fillna(beijing_oh["NO2"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[185]:


#predict CO -also using PM2.5,PM10,SO2,NO2
#linear regression 
train_X, test_X, train_y, test_y = train_test_split(clean_df[["PM2.5","PM10","SO2","NO2"]],clean_df["CO"],test_size=0.2,random_state=5)

linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)
pred_y = linear.predict(clean_df[["PM2.5","PM10","SO2","NO2"]])

beijing_oh["CO"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum()  #901개의 예측되지 않은 값들


# In[190]:


#predict O3 - using NO2,TEMP,PRES
df = beijing_oh[["O3","NO2","TEMP","PRES","DEWP"]]
clean_df = df.dropna(how="any")
#linear regression 
train_X, test_X, train_y, test_y = train_test_split(clean_df[["NO2","TEMP","PRES"]],clean_df["O3"],test_size=0.2,random_state=6)

linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)
pred_y = linear.predict(clean_df[["NO2","TEMP","PRES"]])
#낮은 score.. O3는 regression 대신 median으로 nan 값 채우기
beijing_oh["O3"].fillna(beijing_oh["NO2"].median(), inplace=True)
beijing_oh.isnull().sum() 


# In[193]:


#predict TEMP - using O3,PRES,DEWP
df = beijing_oh[["O3","TEMP","PRES","DEWP"]]
clean_df = df.dropna(how="any")
#linear regression 
train_X, test_X, train_y, test_y = train_test_split(clean_df[["O3","PRES","DEWP"]],clean_df["TEMP"],test_size=0.2,random_state=7)

linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)
pred_y = linear.predict(clean_df[["O3","PRES","DEWP"]])
beijing_oh["TEMP"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[195]:


#predict PRES - using O3,TEMP,DEWP
df = beijing_oh[["O3","TEMP","PRES","DEWP"]]
clean_df = df.dropna(how="any")
#linear regression 
train_X, test_X, train_y, test_y = train_test_split(clean_df[["O3","TEMP","DEWP"]],clean_df["PRES"],test_size=0.2,random_state=8)

linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)
pred_y = linear.predict(clean_df[["O3","TEMP","DEWP"]])
beijing_oh["PRES"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[198]:


#predict DEWP - using TEMP,PRES
df = beijing_oh[["TEMP","PRES","DEWP"]]
clean_df = df.dropna(how="any")
#linear regression 
train_X, test_X, train_y, test_y = train_test_split(clean_df[["TEMP","PRES"]],clean_df["DEWP"],test_size=0.2,random_state=9)

linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)
pred_y = linear.predict(clean_df[["TEMP","PRES"]])
beijing_oh["DEWP"].fillna(pd.Series(pred_y), inplace=True)
beijing_oh.isnull().sum() 


# In[204]:


#predict RAIN 상관관계가 다 낮음,  ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["RAIN"].isnull() == True]
beijing_oh["RAIN"].fillna(method='ffill', limit=1,inplace=True)


# In[207]:


beijing_oh["RAIN"].fillna(method='bfill', limit=1,inplace=True)


# In[209]:


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


# In[257]:


#연속된 nan value는 median으로 채우기, median = 0.0..
beijing_oh["RAIN"].fillna(beijing_oh["RAIN"].median(),inplace=True)


# In[258]:


#precit WSPM 상관관계가 다 낮음, ffil,limit=1 and bfil,limit=1
beijing_oh[beijing_oh["WSPM"].isnull() == True]
beijing_oh["WSPM"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["WSPM"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["WSPM"].isnull() == True]


# In[260]:


#연속된 nan value는 median으로 채우기, median = 1.4..
beijing_oh["WSPM"].fillna(beijing_oh["WSPM"].median(),inplace=True)


# In[261]:


beijing_oh.isnull().sum() 


# In[263]:


#남은 nan value - PM2.5 667, PM10 478,CO 901
beijing_oh[beijing_oh["PM2.5"].isnull() == True]
beijing_oh[beijing_oh["PM10"].isnull() == True]
beijing_oh[beijing_oh["CO"].isnull() == True]


# In[265]:


#ffil,limit=1 and bfil,limit=1으로 채우기
beijing_oh[beijing_oh["PM2.5"].isnull() == True]
beijing_oh["PM2.5"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["PM2.5"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["PM2.5"].isnull() == True]


# In[267]:


#연속된 nan value는 median으로 채우기, median = 1.4..
beijing_oh["PM2.5"].fillna(beijing_oh["PM2.5"].median(),inplace=True)


# In[271]:


#ffil,limit=1 and bfil,limit=1으로 채우기
beijing_oh[beijing_oh["PM10"].isnull() == True]
beijing_oh["PM10"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["PM10"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["PM10"].isnull() == True]
#연속된 nan value는 median으로 채우기, median = 82.0..
beijing_oh["PM10"].fillna(beijing_oh["PM10"].median(),inplace=True)


# In[274]:


#ffil,limit=1 and bfil,limit=1으로 채우기
beijing_oh[beijing_oh["CO"].isnull() == True]
beijing_oh["CO"].fillna(method='ffill', limit=1,inplace=True)
beijing_oh["CO"].fillna(method='bfill', limit=1,inplace=True)
beijing_oh[beijing_oh["CO"].isnull() == True]
#연속된 nan value는 median으로 채우기, median = 900..
beijing_oh["CO"].fillna(beijing_oh["CO"].median(),inplace=True)


# In[275]:


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


# In[275]:


#데이터 시각화
#*Feature engineering
#outlier**** -  모든 스케일러 처리 전에는 아웃라이어 제거가 선행되어야 한다 -> 대기 오염도 및 날씨는 이상치가 중요, 아웃라이어 제거 없이 스케일러..?
#PCA
#*Data reduction

