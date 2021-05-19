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


# In[2]:


class preprocessing :
    #year, month, day, hour column들을 날짜형식으로 변경
    def make_date(year,month,day,hour)
    #시간별 데이터를 날짜별 데이터로 변경
    def hour_to_date()
    


# In[17]:


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


# In[76]:


tt = pd.DataFrame(pd.date_range("2013-03-01","2017-03-01",freq="h"))


# In[77]:


tt = tt[:-1]


# In[78]:


tt = pd.concat([tt,tt,tt,tt,tt,tt,tt,tt,tt,tt,tt,tt],ignore_index = True)


# In[80]:


beijing["Date"] = tt


# In[84]:


beijing.drop(columns=["year","month","day","hour"],inplace=True)


# In[85]:


beijing


# In[113]:


beijing.info()


# In[86]:


seoul.info()


# In[87]:


weather.info()


# In[148]:


weather


# In[106]:


seoul


# In[118]:


date_s = seoul["MSRDT"].astype(str)
seoul["MSRDT"] = date_s.apply(lambda x: datetime.strptime(x, "%Y%m%d%H%M"))


# In[119]:


seoul


# In[152]:


weather["tm"] = weather["tm"].apply(lambda _: datetime.strptime(_,"%Y-%m-%d"))


# In[153]:


weather["tm"]


# In[122]:


#*Data value changes
#find nan values
seoul.isnull().sum()
weather.isnull().sum()
beijing.isnull().sum()


# In[131]:


#서울 강수량을 0.0으로 fill
weather["sumRn"].fillna(0.0,inplace=True)


# In[132]:


weather.isnull().sum()


# In[164]:


weather  


# In[167]:


#날씨 결측값을 가장 큰 양의 상관관계를 갖는 column을 파악한 뒤 regression으로 모델 학습 후 예측값으로 채우기
#0.5 정도의 값이 얻어지면 : 강력한 양(+)의 상관. 변인 x 가 증가하면 변인 y 가 증가한다.
std_weather = pd.DataFrame(StandardScaler().fit_transform(weather.iloc[:,1:]),columns=weather.iloc[:,1:].columns)
weather.corr()
std_weather.corr()


# In[168]:


sns.heatmap(std_weather.corr(), annot=True, fmt='0.2f')
plt.title('weather', fontsize=20)
plt.show()


# In[229]:


#sumGsr과 avgTs는 0.5,sumGsr과 minRhm는 -0.55 sumGsr과 avgTs는 0.4
#sumGsr-avgTs --> accuray = 0.2..
#sumGsr-avgTsm,minRhm --> accuray = 0.6

#// maxWd와 sumGsr은 0.2
#0.2 정도의 값이 얻어지면 : 너무 약해서 의심스러운 양(+)의 상관
#maxWd는 전날과 다음날의 평균값으로 채운다. 

df = std_weather[["sumGsr","avgTs","minRhm","avgTa"]]
df = df[df["sumGsr"].notnull()]

#linear regression 
train_X, test_X, train_y, test_y = train_test_split(df[["avgTs","minRhm","avgTa"]],df["sumGsr"],test_size=0.2,random_state=0)


# In[234]:


linear = LinearRegression().fit(train_X,train_y)
linear.coef_
linear.intercept_
linear.score(test_X,test_y)


# In[249]:


pred_y = linear.predict(std_weather.loc[:, ["avgTs","minRhm","avgTa"]])


# In[255]:


std_weather['sumGsr'].fillna(pd.Series(pred_y), inplace=True)


# In[256]:


std_weather.isnull().sum()


# In[267]:


std_weather["Date"] = weather["tm"]


# In[268]:


std_weather[std_weather["maxWd"].isnull() == True]


# In[270]:


std_weather["maxWd"].fillna(method='ffill', limit=1,inplace=True)


# In[271]:


std_weather[std_weather["maxWd"].isnull() == True]


# In[272]:


std_weather["maxWd"].fillna(method='bfill', limit=1,inplace=True)


# In[274]:


std_weather.isnull().sum()


# In[288]:


beijing[beijing["station"]=="Aotizhongxin"].isnull().sum()


# In[293]:


beijing.isnull().sum()


# In[289]:


beijing.corr()


# In[285]:


beijing


# In[291]:


sns.heatmap(beijing.corr(), annot=True, fmt='0.2f')
plt.title('Beijing', fontsize=20)
plt.show()


# In[292]:


beijing[beijing["PM10"].isnull()]


# In[324]:


#onehotencoding 
ohe = OneHotEncoder()
station_oh = ohe.fit_transform(beijing["station"].values.reshape(-1,1)).toarray()
beijing_oh = beijing.copy()
station_oh = pd.DataFrame(station_oh,columns=beijing["station"].unique())


# In[325]:


station_oh


# In[326]:


beijing_oh.drop(columns="station",inplace=True)


# In[360]:


beijing_oh = pd.concat([beijing_oh,station_oh],axis=1)


# In[349]:


#풍향은 categorical value, numerical value로 변환해야 corr 파악 후 예측할 수 있는데 nan값이 있기 때문에 
#처리가 난해..
#풍향은 unique를 통해 max 값으로 채운다. 

beijing_oh["wd"].fillna(beijing_oh["wd"].value_counts().index[beijing_oh["wd"].value_counts().argmax()],inplace=True)


# In[351]:


beijing_oh["wd"].isnull().sum()


# In[352]:


#onehotencoding 
wd_oh = ohe.fit_transform(beijing_oh["wd"].values.reshape(-1,1)).toarray()
wd_oh = pd.DataFrame(wd_oh,columns=beijing_oh["wd"].unique())


# In[362]:


beijing_oh = pd.concat([beijing_oh,wd_oh],axis=1)
beijing_oh.drop(columns="wd",inplace=True)


# In[363]:


beijing_oh.columns


# In[296]:


clean_beijing = beijing.dropna(how="any")


# In[301]:


clean_beijing["station"]


# In[ ]:





# In[275]:


#데이터 시각화
#onehotencoding!
#*Feature engineering
#scaling 하기 !
#*Data reduction

