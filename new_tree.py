from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np, pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn import tree
import graphviz
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix


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

weather = pd.read_csv("weather.csv")

weather["tm"] = weather["tm"].apply(lambda _: datetime.strptime(_,"%Y-%m-%d"))

weather["sumRn"].fillna(0.0,inplace=True)

linear,score = preprocessing().reg_score(weather,["avgTs","minRhm","avgTa"],"sumGsr")
pred_y = linear.predict(weather.loc[:, ["avgTs","minRhm","avgTa"]])
weather['sumGsr'].fillna(pd.Series(pred_y), inplace=True)

weather[weather["maxWd"].isnull() == True]

weather["maxWd"].fillna(method='ffill', limit=1,inplace=True)

weather[weather["maxWd"].isnull() == True]

weather["maxWd"].fillna(method='bfill', limit=1,inplace=True)

std_weather = pd.DataFrame(StandardScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
std_weather["tm"]=weather["tm"]

mm_weather = pd.DataFrame(MinMaxScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
mm_weather["tm"]=weather["tm"]

ma_weather = pd.DataFrame(MaxAbsScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
ma_weather["tm"]=weather["tm"]

rb_weather = pd.DataFrame(RobustScaler().fit_transform(weather.drop(columns="tm")),columns=weather.drop(columns="tm").columns)
rb_weather["tm"]=weather["tm"]



feature_name =["avgTa","sumRn","avgWs","maxWd","avgTd","minRhm","sumGsr","avgTs"]


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


#---------------Standaradization dataset---------------
x=std_weather[feature_name]
target_name="tm"
y=std_weather[target_name]

train_x, test_x, train_y, test_y = train_test_split(x,y,train_size=0.7,random_state=0)
train_x,valid_x,train_y,valid_y=train_test_split(train_x,train_y,train_size=0.7,random_state=0)

 # 1st tree(gini)
clf=makeDecisionTree("gini",train_x,train_y,0)
printTreeGraph(clf)

 # 2nd tree(entropy)
clf2=makeDecisionTree("entropy",train_x,train_y,0)
printTreeGraph(clf2)

 # 3rd tree(pruning)
clf3=makeDecisionTree("entropy",train_x,train_y,2)
printTreeGraph(clf3)

treeGridSearch(train_x,train_y)
best_tree_sd=tree.DecisionTreeClassifier(criterion="gini",max_depth=14,max_features=None,splitter="best",random_state=12)

baggingGridSearch(best_tree_sd)


#---------------MinMax dataset---------------
x=mm_weather[feature_name]
target_name="tm"
y=mm_weather[target_name]

train_x, test_x, train_y, test_y = train_test_split(x,y,train_size=0.7,random_state=0)
train_x,valid_x,train_y,valid_y=train_test_split(train_x,train_y,train_size=0.7,random_state=0)

 # 1st tree(gini)
clf=makeDecisionTree("gini",train_x,train_y,0)
printTreeGraph(clf)

 # 2nd tree(entropy)
clf2=makeDecisionTree("entropy",train_x,train_y,0)
printTreeGraph(clf2)

 # 3rd tree(pruning)
clf3=makeDecisionTree("entropy",train_x,train_y,2)
printTreeGraph(clf3)

treeGridSearch(train_x,train_y)
best_tree_mm=tree.DecisionTreeClassifier(criterion="gini",max_depth=14,max_features=None,splitter="best",random_state=12)
baggingGridSearch(best_tree_mm)

#---------------MaxAbs dataset---------------
x=ma_weather[feature_name]
target_name="tm"
y=ma_weather[target_name]

train_x, test_x, train_y, test_y = train_test_split(x,y,train_size=0.7,random_state=0)
train_x,valid_x,train_y,valid_y=train_test_split(train_x,train_y,train_size=0.7,random_state=0)

 # 1st tree(gini)
clf=makeDecisionTree("gini",train_x,train_y,0)
printTreeGraph(clf)

 # 2nd tree(entropy)
clf2=makeDecisionTree("entropy",train_x,train_y,0)
printTreeGraph(clf2)

 # 3rd tree(pruning)
clf3=makeDecisionTree("entropy",train_x,train_y,2)
printTreeGraph(clf3)

treeGridSearch(train_x,train_y)
best_tree_ma=tree.DecisionTreeClassifier(criterion="gini",max_depth=14,max_features=None,splitter="best",random_state=12)
baggingGridSearch(best_tree_ma)

#---------------Robust dataset---------------
x=ma_weather[feature_name]
target_name="tm"
y=ma_weather[target_name]

train_x, test_x, train_y, test_y = train_test_split(x,y,train_size=0.7,random_state=0)
train_x,valid_x,train_y,valid_y=train_test_split(train_x,train_y,train_size=0.7,random_state=0)

 # 1st tree(gini)
clf=makeDecisionTree("gini",train_x,train_y,0)
printTreeGraph(clf)

 # 2nd tree(entropy)
clf2=makeDecisionTree("entropy",train_x,train_y,0)
printTreeGraph(clf2)

 # 3rd tree(pruning)
clf3=makeDecisionTree("entropy",train_x,train_y,2)
printTreeGraph(clf3)

treeGridSearch(train_x,train_y)
best_tree_rb=tree.DecisionTreeClassifier(criterion="gini",max_depth=14,max_features=None,splitter="best",random_state=12)
baggingGridSearch(best_tree_rb)
