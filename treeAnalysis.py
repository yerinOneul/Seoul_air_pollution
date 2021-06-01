
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
import graphviz
from sklearn.metrics import mean_squared_error


class treeAnalysis:
    def printTreeGraph(model, feature_name, target_name):
        dot_data = tree.export_graphviz(model,
                                        out_file=None,
                                        feature_names=feature_name,
                                        class_names=target_name,
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        print(graph)

    def foldValidatation(model, train_x, train_y, foldNum):
        cv_scores = cross_val_score(model, train_x, train_y, cv=foldNum)
        print(cv_scores)
        print("cv_scores mean: {}".format(np.mean(cv_scores)))
        print(cross_validate(model, train_x, train_y, scoring=['accuracy', 'roc_auc'], return_train_score=True))

    def findBestTree(x_train, y_train):
        tree_params = {
            "criterion": ["mse", "friedman_mse", "mae", "poisson"],
            "splitter": ["best", "random"],
            "max_depth": [2, 6, 10, 14],
            "max_features": ["auto", "sqrt", "log2", 10, None]
        }
        tree_grid = GridSearchCV(tree.DecisionTreeClassifier(random_state=12),
                                 tree_params,
                                 scoring="neg_mean_squared_error",
                                 verbose=1,
                                 n_jobs=-1)
        tree_grid.fit(x_train, y_train)
        best_param = tree_grid.best_params_
        print("Best parameter(Decision Tree) : ")
        print(best_param)
        print("Best score(Decision Tree) : ")
        print(tree_grid.best_score_)

    def findBestBoostedTree(tree_model, x_train, y_train):
        gbr = GradientBoostingRegressor(random_state=13).fit(x_train, y_train)
        gbr_pred = gbr.predict(x_train)
        gbr.score(x_train, y_train)
        RMSE = np.sqrt(mean_squared_error(gbr_pred, y_train))
        print(RMSE)
        gbr_params = {
            "n_estimators": [15, 30, 100, 600],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 9, 14],
            "max_features": [0.3, 0.5, 1.0],
            "min_samples_split": [2, 3, 4],
            "loss": ["ls", "lad", "huber", "quantile"]
        }
        gbr_grid = GridSearchCV(GradientBoostingRegressor(random_state=33), gbr_params, cv=3, n_jobs=-1, verbose=1)

        gbr_grid.fit(x_train, y_train)
        print("Best parameter(Gradient boosting) : ")
        print(gbr_grid.best_params_)
        print("Best score(Gradient boosting) : ")
        print(gbr_grid.best_score_)

    def plotFeatureImportance(gbr_model, data):
        feature_importance = gbr_model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(data.columns)[sorted_idx])
        plt.title('Feature Importance')





