from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np, pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import math
import requests
import json
from pandas.io.json import json_normalize
import os
import webbrowser
import folium
from folium import plugins


# Switch data with labeled datas in oneHotencoded data
def calculatemean(dataset):
    sum_dataset = np.zeros(25, dtype=object)
    for i in range(len(dataset)):
        index_tmp = dataset.index[dataset[col[i]] != 0]
        sum_dataset[i] = sum_dataset[i] + dataset.iloc[index_tmp, i]
    tmp_array_name = np.zeros(25, dtype=object)
    tmp_array_size = np.zeros(25, dtype=object)
    for i in range(len(sum_dataset)):
        tmp_array_name[i] = sum_dataset[i].name
        tmp_array_size[i] = sum_dataset[i].sum() / sum_dataset[i].size
    return tmp_array_name, tmp_array_size


def points_array(points):
    final_points = []

    for x in range(0, len(points)):

        if len(points[x]) == 2:
            final_points.append(points[x])
        else:
            target = points[x]
            for y in range(0, len(target)):
                final_points.append(target[y])

    return final_points


# Visualize optimized k values
def visualizeElbowMethod(model, x):
    visualizer = KElbowVisualizer(model, k=(1, 10))
    visualizer.fit(x.values)
    visualizer.show()


# bubblesort
def cal(temparray):
    for i in range(len(temparray)):
        for j in range(len(temparray) - i - 1):
            if temparray[j] > temparray[j + 1]:
                temp = temparray[j]
                temparray[j] = temparray[j + 1]
                temparray[j + 1] = temp
    return temparray


# ordering
def compare(tmp1, tmp2):
    resultarray = np.zeros(len(tmp1))
    for i in range(len(tmp1)):
        for j in range(len(tmp1)):
            if tmp1[i] == tmp2[j]:
                resultarray[i] = j
                continue
    return resultarray


# split data in 4 groups
def setting(tmp):
    sizenum = round(len(tmp) / 4)
    for i in range(len(tmp)):
        if tmp[i] < sizenum:
            tmp[i] = 0
        elif tmp[i] < sizenum * 2:
            tmp[i] = 1
        elif tmp[i] < sizenum * 3:
            tmp[i] = 2
        else:
            tmp[i] = 3


# draw seoulmap with dataset(NAME,VALUE) NAME should be the name of area of seoul
# value is the value of each dataset
def showmap(dataset, value):
    state_geo = 'TL_SCCO_SIG_WGS84.json'
    state_geo2 = 'https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json'
    json_data = open(state_geo).read()
    jsonResult = json.loads(json_data)
    center_locations = pd.DataFrame()
    codes = []
    names = []
    x_list = []
    y_list = []
    for x in range(0, len(jsonResult['features'])):
        code = jsonResult['features'][x]['properties']['SIG_CD']
        name = jsonResult['features'][x]['properties']['SIG_KOR_NM']

        points = jsonResult['features'][x]['geometry']['coordinates'][0]
        points = points_array(points)
        points_df = pd.DataFrame(points)
        points_df.columns = ['x', 'y']
        x = points_df.x
        y = points_df.y
        X = (x[1] + x[2]) / 2
        Y = (y[1] + y[2]) / 2

        codes.append(code)
        names.append(name)
        x_list.append(X)
        y_list.append(Y)

    center_locations['CODE'] = codes
    center_locations['NAME'] = names
    center_locations['X'] = x_list
    center_locations['Y'] = y_list

    tmp_array_name = np.zeros(25, dtype=object)
    tmp_array_size = np.zeros(25, dtype=object)
    temp_seoulmap = pd.DataFrame({'NAME': tmp_array_name})
    for i in range(len(temp_seoulmap)):
        tmp_array_name[i] = dataset[i].name
        tmp_array_size[i] = dataset[i].sum() / dataset[i].size
    seoul_df = pd.DataFrame({'Mean value': tmp_array_size}, index=tmp_array_name)
    temp_seoulmap['VALUE'] = value
    center_locations2 = center_locations[center_locations['Y'] >= 37.426026]
    target_df = pd.merge(temp_seoulmap, center_locations2, how='left', on='NAME')
    target_df = target_df.dropna(axis=0, subset=['X', 'Y'])

    m = folium.Map(location=[37.562225, 126.978555], tiles="OpenStreetMap", zoom_start=11)

    m.choropleth(
        geo_data=state_geo2,
        name='미세먼지 위험군',
        data=temp_seoulmap,
        columns=['NAME', 'VALUE'],
        key_on='feature.properties.name',
        fill_color='Blues',
        fill_opacity=0.7,
        line_opacity=0.3,
        color='gray',
        legend_name='income'
    )

    for i in range(0, len(target_df)):
        latitude = target_df.iloc[i]['Y']
        longitude = target_df.iloc[i]['X']
        location = (latitude, longitude)

        if target_df.iloc[i]['NAME'] in ['서초구', '강남구']:
            color = 'white'
        else:
            color = '#3186cc'

        folium.CircleMarker(location, radius=10, color=color, fill_color=color, fill_opacity=0.1, opacity=0.0,
                            popup=target_df.iloc[i]['NAME'] + "\n" + str(
                                int(round(target_df.iloc[i]['VALUE'] / 10000, 0))) + "만원").add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Save to html
    m.save('kr_incode.html')

