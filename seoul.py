#!/usr/bin/env python
# coding: utf-8

# In[2]:


class Preprocessing :
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
    
    #Return Scaled Dataframe
    #df : dataframe
    #method : scaling method (standard, minmax, maxabs, robust)
    #return : scaled dataframe
    def scaler(self,df,method = "std"):
        if method == "std" : 
            return (df - df.mean()) / df.std()
        elif method == "minmax" :
            return (df- df.min()) /(df.max() - df.min())
        elif method == "maxabs" :
            return df/df.abs().max()
        elif method == "robust" :
            return (df - df.median()) / (df.quantile(.75)-df.quantile(.25))
        else : 
            raise Exception("Scaling method input error.")
            
    #Return Scaled Dataframe Matrix
    #df : dataframe
    #return : scaled dataframe matrix, (0,0) = std, (0,1) = minmax, (1,0) = maxabs, (1,1) = robust           
    def scalerMatrix(self,df):
        return [[self.scaler(df,"std"),self.scaler(df,"minmax")],[self.scaler(df,"maxabs"),self.scaler(df,"robust")]]


# In[ ]:




