#!/usr/bin/env python
# coding: utf-8
# %%
from bokeh.resources import INLINE
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
import numpy as np

import geopandas as gpd
import movingpandas as mpd
import shapely as shp
import hvplot.pandas

import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
import time
from datetime import datetime, timedelta
import sys

random_state_trajs_fishing_info = 0
random_state_trajs_fishing = 0.1
min_duration_trajectory = 10  # in minutes
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======

>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465

def load_dataset():
    # Dataset processing and filter
    input_csv = "dataset/dataset_fishing_train.csv"
    n = 0
    chunksize = 1000000
    first = True
    mmsi_dict = {}
    with pd.read_csv(input_csv, chunksize=chunksize, header=0) as reader:
        for pd_chunk in reader:

            try:
                # remove without mmsi
                pd_chunk.dropna(subset=['mmsi'], inplace=True)
                pd_chunk["mmsi"] = pd_chunk["mmsi"].astype(int)

                # set column type
                pd_chunk['dh'] = pd.to_datetime(pd_chunk['dh'])

                try:
                    pd_chunk['lat'] = pd.to_numeric(
                        pd_chunk['lat'], downcast='float', errors='coerce')
                except Exception as A:
                    print("Error chunk ", n,  " ", ": ", A)
                    pd_chunk.dropna()

                try:
                    pd_chunk['lon'] = pd.to_numeric(
                        pd_chunk['lon'], downcast='float', errors='coerce')
                except Exception as A:
                    print("Error chunk ", n,  " ", ": ", A)
                    pd_chunk.dropna()

                pd_chunk['rumo'] = pd_chunk['rumo'].astype(float)
                pd_chunk['veloc'] = pd_chunk['veloc'].astype(float)

                # deleta linha com nan
                pd_chunk.dropna()

                # append results
                if first:
                    df_gfw = pd_chunk
                    first = False
                else:
                    df_gfw = df_gfw.append(pd_chunk)

                # append results
                # append data frame to CSV file
                print("chunk: " + str(n))
                n += 1
            except Exception as A:
                print("Error chunk ", n,  " ", ": ", A)
                pass

    df_gfw['dh'] = pd.to_datetime(df_gfw['dh'], utc=True)
    return df_gfw


# GDF
def load_gdf(df_gfw):
    import geopandas as gpd
    import movingpandas as mpd
    import shapely as shp
    import hvplot.pandas

    from geopandas import GeoDataFrame, read_file
    from datetime import datetime, timedelta
    from holoviews import opts

    import warnings
    warnings.filterwarnings('ignore')

    opts.defaults(opts.Overlay(
        active_tools=['wheel_zoom'], frame_width=500, frame_height=400))

    gdf = gpd.GeoDataFrame(
        df_gfw.set_index('dh'), geometry=gpd.points_from_xy(df_gfw.lon, df_gfw.lat))

    gdf.set_crs('epsg:4326')

    return gdf

# Filter GDF to equal data
<<<<<<< HEAD
<<<<<<< HEAD


=======


>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======


>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
def filter_gdf(gdf, len_gdf_only_fishing, len_gdf_no_fishing):

    gdf_only_fishing = gdf[gdf['vesselType'] ==
                           'Fishing'][:len_gdf_only_fishing]  # 263K
    gdf_no_fishing = gdf[gdf['vesselType'] !=
                         'Fishing'][:len_gdf_no_fishing]  # 16M

    # gdf_only_fishing = gdf[ gdf['vesselType'] == 'Fishing'][:2600000] #263K
    # gdf_no_fishing   = gdf[ gdf['vesselType'] != 'Fishing'][:3000000] #16M

    gdf_filtered = pd.concat([gdf_only_fishing, gdf_no_fishing])

    return gdf_only_fishing, gdf_no_fishing, gdf_filtered


# Trajectories
def create_trajectory(gdf):
    import movingpandas as mpd
    import shapely as shp
    import hvplot.pandas
    import time

    # reset index
    gdf = gdf.reset_index()
    gdf['dh'] = pd.to_datetime(gdf['dh'], utc=True)

    # limit to avoid slow
#     gdf = gdf[:10000]

    # create trajectories

    start_time = time.time()

    # Specify minimum length for a trajectory (in meters)
    minimum_length = 0
    # collection = mpd.TrajectoryCollection(gdf, "imo",  t='dh', min_length=0.001)
    collection = mpd.TrajectoryCollection(
        gdf, "mmsi",  t='dh', min_length=0.001, crs='epsg:4326')
    collection.add_direction(gdf.rumo)
    collection.add_speed(gdf.veloc)

    # configura o intervalo em minutos entre os pontos para formar uma trajetoria
    collection = mpd.ObservationGapSplitter(
        collection).split(gap=timedelta(minutes=90))

    collection.add_speed(overwrite=True)
    collection.add_direction(overwrite=True)

    end_time = time.time()
    print("Time creation trajectories: ", (end_time-start_time)/60,  " min")

    return collection


# Trajectories Filter Situation - Only use for experiments
def mean_duration_fishing(trajs):
    mean = 0.0
    for traj in trajs.trajectories:
        mean += traj.get_duration().seconds

    return mean / len(trajs.trajectories)


# filters for trajectories
def filter_trajs(trajs_fishing, trajs_no_fishing):

    new_traj_fishing = []
    new_traj_no_fishing = []
    mean_traj_duration_fishing = mean_duration_fishing(trajs_fishing)
    percent_mean = 0.1

    # fishing
    for traj in trajs_fishing.trajectories[:]:
        if traj.get_duration().seconds > 60*min_duration_trajectory and traj.df.speed.mean()*(100000*1.94384) > 1 and traj.df.speed.mean()*(100000*1.94384) < 50 and len(traj.df) > 2:
            new_traj_fishing.append(traj)

    # non fishing
    for traj in trajs_no_fishing.trajectories[:]:
        if traj.get_duration().seconds > 60*min_duration_trajectory and traj.df.speed.mean()*(100000*1.94384) > 1 and traj.df.speed.mean()*(100000*1.94384) < 50 and len(traj.df) > 2:
            new_traj_no_fishing.append(traj)

    print("fishing trajs: ", len(new_traj_fishing))
    print("non fishing trajs: ", len(new_traj_no_fishing))

    return mpd.TrajectoryCollection(new_traj_fishing),  mpd.TrajectoryCollection(new_traj_no_fishing)


# Select if load from file or build all trajectories from gdf
def load_or_build_trajectories(len_gdf_only_fishing, load_trajectories_collection_from_file=True):
    import pickle

    ####
    # select if load from file or build all trajectories from gdf
    ###
    object_data_dir = 'objects/'

    if load_trajectories_collection_from_file:
        try:
            # Load trajectories collection
            with open(object_data_dir + 'trajs_fishing.mvpandas.' + str(len_gdf_only_fishing), 'rb') as trajs_fishing_file:
                trajs_fishing = pickle.load(trajs_fishing_file)
            with open(object_data_dir + 'trajs_no_fishing.mvpandas.' + str(len_gdf_no_fishing), 'rb') as trajs_no_fishing_file:
                trajs_no_fishing = pickle.load(trajs_no_fishing_file)
        except Exception as e:
            print(e, "Trajectories Collection File not Found!")

    else:
        trajs_fishing = create_trajectory(gdf_only_fishing)
        trajs_no_fishing = create_trajectory(gdf_no_fishing)

        # store trajectories collection
        with open(object_data_dir + 'trajs_fishing.mvpandas.' + str(len_gdf_only_fishing), 'wb') as trajs_fishing_file:
            pickle.dump(trajs_fishing, trajs_fishing_file)

        with open(object_data_dir + 'trajs_no_fishing.mvpandas.' + str(len_gdf_no_fishing), 'wb') as trajs_no_fishing_file:
            pickle.dump(trajs_no_fishing, trajs_no_fishing_file)

    # trajs_fishing, trajs_no_fishing = filter_trajs(trajs_fishing, trajs_no_fishing)
    print("Loaded ", len(trajs_fishing), " trajs fishing and ",
          len(trajs_no_fishing), " trajs non fishing.")

    return trajs_fishing, trajs_no_fishing


# Traj Info
def init_trajectory_data(collection):
    import movingpandas as mpd
    import shapely as shp
    import hvplot.pandas

    collection.add_speed(overwrite=True)
    collection.add_direction(overwrite=True)

    # format trajectories to clustering
    linhas_traj_id = np.array([])
    mmsi = np.array([])
    area = np.array([])
    varRumo = np.array([])
    varVeloc = np.array([])
    duracao = np.array([])
    medrumo = np.array([])
    nome = np.array([])
    medVeloc = np.array([])
    medSpeed = np.array([])
    endTrajCoastDist = np.array([])
    vesselType = np.array([])
    traj_len = np.array([])
    n_points = np.array([])
    for traj in collection.trajectories:
        traj_id = str(traj.to_traj_gdf()["traj_id"]).split()[1]
#         mmsi        =  np.append( mmsi, traj.df['mmsi'][0].astype(int) )
        mmsi = np.append(mmsi, traj.df['mmsi'][0])
        linhas_traj_id = np.append(linhas_traj_id, traj_id)
        area = np.append(area, traj.get_mcp().area)
        varRumo = np.append(varRumo, traj.df.direction.var())
        medrumo = np.append(medrumo, traj.df.direction.mean())
        varVeloc = np.append(varVeloc, traj.df.speed.var())
        duracao = np.append(duracao, traj.get_duration().seconds)
        nome = np.append(nome, traj.df["nome_navio"][0])
        medVeloc = np.append(medVeloc, traj.df.speed.mean())
        medSpeed = np.append(medSpeed, traj.df.speed.mean()*(100000*1.94384))
        traj.df["speed_knot"] = traj.df.speed*(100000*1.94384)
#         endTrajCoastDist =    np.append( endTrajCoastDist, traj.get_end_location().distance(line_coast)*100 )
        vesselType = np.append(vesselType, traj.df["vesselType"][0])
        traj_len = np.append(traj_len, traj.get_length())
        n_points = np.append(n_points, len(traj.df))

    clus_df = pd.DataFrame()
    clus_df["traj_id"] = linhas_traj_id
    clus_df["mmsi"] = mmsi
    clus_df["area"] = area
    clus_df["varRumo"] = varRumo
    clus_df["medrumo"] = medrumo
    clus_df["varVeloc"] = varVeloc
    clus_df["duracao"] = duracao
    clus_df["nome_navio"] = nome
    clus_df["medVeloc"] = medVeloc
    clus_df["medSpeed"] = medSpeed
#     clus_df["endTrajCoastDist"]  = endTrajCoastDist
    clus_df["vesselType"] = vesselType
    clus_df["traj_len"] = traj_len
    clus_df["n_points"] = n_points

    return clus_df


# # Set Label
def set_label_trajectory_info(df_trajs_info):
    df_trajs_info['activity'] = 'normal'
    df_trajs_info.loc[df_trajs_info['vesselType']
                      == 'Fishing', ['activity']] = 'fishing'

# plot some relevant data


def plot_statistics_dataset(df_trajs_info):
    # plot mean duration by vessel type
    df_trajs_info.groupby(by=['vesselType'])[
        'duracao'].mean().plot(kind='bar', figsize=(10, 3))

    # plot mean var course by vessel type
    df_trajs_info[
        (df_trajs_info['medSpeed'] > 1) &
        (df_trajs_info['medSpeed'] < 50) &
        (df_trajs_info['duracao'] > 60*min_duration_trajectory) &
        (df_trajs_info['n_points'] > 2)
    ].groupby(by=['vesselType'])['varRumo'].mean().plot(kind='bar', figsize=(10, 3))

    # plot count by vessel type
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 3))
    plt.title('Number of Trajectories by Vessel Type')
    df_trajs_info[
        (df_trajs_info['medSpeed'] > 1) &
        (df_trajs_info['medSpeed'] < 50) &
        (df_trajs_info['duracao'] > 60*min_duration_trajectory) &
        (df_trajs_info['n_points'] > 2)
    ].groupby(by=['vesselType'])['varRumo'].count().plot(kind='bar', figsize=(10, 3))
    plt.savefig("number_of_vessels_by_type.png", bbox_inches='tight', ax=ax)
    # plt.show()


# Filter Trajectories-Data

# Filter trajectories info
def filter_trajs_info(df_trajs_info, random_state_trajs_fishing_info):

    # Filter trajectories with duration > 10 min!!!
    df_trajs_fishing_info_filtered = df_trajs_info[
        (df_trajs_info['vesselType'] == 'Fishing') &
        (df_trajs_info['medSpeed'] > 1) &
        (df_trajs_info['medSpeed'] < 50) &
        (df_trajs_info['duracao'] > 60*min_duration_trajectory) &
        (df_trajs_info['n_points'] > 2)
    ]

    df_trajs_nofishing_info_filtered = df_trajs_info[
        (df_trajs_info['vesselType'] != 'Fishing') &
        (df_trajs_info['medSpeed'] > 1) &
        (df_trajs_info['medSpeed'] < 50) &
        (df_trajs_info['duracao'] > 60*min_duration_trajectory) &
        (df_trajs_info['n_points'] > 2)
    ]

    n_traj_info_fishing = len(df_trajs_fishing_info_filtered)
    n_traj_info_nofishing = len(df_trajs_nofishing_info_filtered)

    if n_traj_info_fishing < n_traj_info_nofishing:
        n_size = n_traj_info_fishing
        # equaly data fishing and non fishing, we have a lot non fishing than fishing
        df_trajs_nofishing_info_filtered = df_trajs_nofishing_info_filtered[:n_size]

    else:
        n_size = n_traj_info_nofishing
        # equaly data fishing and non fishing, we have a lot non fishing than fishing
        df_trajs_fishing_info_filtered = df_trajs_fishing_info_filtered[:n_size]

    # # equaly data fishing and non fishing, we have a lot non fishing than fishing
    # df_trajs_nofishing_info_filtered = df_trajs_nofishing_info_filtered[:n_size]

    # random dataframes
#     random_state_trajs_fishing_info += 1
#     df_trajs_fishing_info_filtered   = df_trajs_fishing_info_filtered.sample(frac=1, random_state=random_state_trajs_fishing_info, replace=True, ignore_index=True)
#     df_trajs_nofishing_info_filtered = df_trajs_nofishing_info_filtered.sample(frac=1, random_state=random_state_trajs_fishing_info, replace=True, ignore_index=True)

    # get sample to use in validation, (20%)
    x_test = pd.concat([
        df_trajs_fishing_info_filtered[[
            'duracao', 'varRumo', 'varVeloc', 'traj_len', 'n_points']][int(0.8*n_size):],
        df_trajs_nofishing_info_filtered[[
            'duracao', 'varRumo', 'varVeloc', 'traj_len', 'n_points']][int(0.8*n_size):]
    ])

    # get y concat 20% fishing info trajectories and 20% non fishing info trajectories
    y_test = pd.concat([
        df_trajs_fishing_info_filtered[['activity']][int(0.8*n_size):n_size],
        df_trajs_nofishing_info_filtered[['activity']][int(0.8*n_size):n_size]
    ])

    # concat fishing trajectories whith no fishing trajectories (normal) to use in train
    df_trajs_info_filtered = pd.concat([
        df_trajs_fishing_info_filtered[:int(0.8*n_size)],
        df_trajs_nofishing_info_filtered[:int(0.8*n_size)]
    ])

    df_trajs_info_filtered

    return df_trajs_info_filtered, x_test, y_test


# Building Model for Fishing Activity Detection

# Logistic Regression in Trajectory-base data
def logistic_regression(x, y, x_test, y_test):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score

    print("*** Logistic Regression")

    # Definindo valores que serão testados em Logistic Regression:
    valores_C = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100])
    regularizacao = ['l1', 'l2']
    valores_grid = {'C': valores_C, 'penalty': regularizacao}

    # Criando o modelo
    modelLR = LogisticRegression(solver='liblinear', max_iter=1000)

    # Criando os GRIDS
    model = GridSearchCV(estimator=modelLR, param_grid=valores_grid, cv=5)
    model.fit(x, y)

    # Imprimindo a melhor acurácia e os melhores parametros
    print("LR Best accuracy: ", model.best_score_)
    print("C parameter: ",     model.best_estimator_.C)
    print("Regularization: ",   model.best_estimator_.penalty)

    # results
    start_time = time.time()
    y_pred = model.predict(x_test)
    end_time = time.time()
    print("LR Execution time: ", (end_time-start_time))

    from sklearn.metrics import confusion_matrix
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred)
    print("LR Confusion Matrix: \n", cm)

    accuracy = accuracy_score(y_true, y_pred)
    print("LR TEST accuracy: ", accuracy)

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    group_names = ['True Fishing', 'False Fishing',
                   'False Sailing', 'True Sailing']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = sn.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
    fig = plot.get_figure()
    fig.savefig("cf_rl.png")

    # recall, precision, f1
    from sklearn.metrics import precision_recall_fscore_support
    print("LR: ", precision_recall_fscore_support(
        y_true, y_pred, average='macro'))
    print("LR: ", precision_recall_fscore_support(y_true, y_pred, average=None))

    precision_recall_fscore = np.array(
        precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro')
<<<<<<< HEAD
<<<<<<< HEAD
=======

    return model.best_score_, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======

    return model.best_score_, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465

    return model.best_score_, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
# Decision Tree in Trajectory-base data
def decision_tree(x, y, x_test, y_test):
    from sklearn import decomposition, datasets
    from sklearn import tree
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("*** Decision Tree")

    std_slc = StandardScaler()
    pca = decomposition.PCA()
    dec_tree = tree.DecisionTreeClassifier()

    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])

    n_components = list(range(1, x.shape[1]+1, 1))
    criterion = ['gini', 'entropy']
    max_depth = [2, 4, 6, 8, 10, 12]

    parameters = dict(pca__n_components=n_components,
                      dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)

    model = GridSearchCV(pipe, parameters, cv=5)
    r = model.fit(x, y)

    print('Best Criterion:', model.best_estimator_.get_params()
          ['dec_tree__criterion'])
    print('Best max_depth:', model.best_estimator_.get_params()
          ['dec_tree__max_depth'])
    print('Best Number Of Components:',
          model.best_estimator_.get_params()['pca__n_components'])
    print()
    print(model.best_estimator_.get_params()['dec_tree'])
    print("DT Best accuracy: ", model.best_score_)

    # results
    start_time = time.time()
    y_pred = model.predict(x_test)
    end_time = time.time()
    print("DT Execution time: ", (end_time-start_time))

    from sklearn.metrics import confusion_matrix
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred)
    print("DT Confusion Matrix: \n", cm)

    accuracy = accuracy_score(y_true, y_pred)
    print("DT TEST accuracy: ", accuracy)

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    group_names = ['True Fishing', 'False Fishing',
                   'False Sailing', 'True Sailing']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = sn.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
    fig = plot.get_figure()
    fig.savefig("cf_decision_tree_val.png")

    # recall, precision, f1
    from sklearn.metrics import precision_recall_fscore_support
    print("DT: ", precision_recall_fscore_support(
        y_true, y_pred, average='macro'))
    print("DT: ", precision_recall_fscore_support(y_true, y_pred, average=None))
<<<<<<< HEAD
<<<<<<< HEAD

    precision_recall_fscore = np.array(
        precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro')

    return model.best_score_, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)

=======

    precision_recall_fscore = np.array(
        precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro')

    return model.best_score_, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)

>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======

    precision_recall_fscore = np.array(
        precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro')

    return model.best_score_, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)

>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465

# SVM in Trajectory-base data
def svm(x, y, x_test, y_test):
    from sklearn import decomposition, datasets
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("*** SVM")

    std_slc = StandardScaler()
    pca = decomposition.PCA()
    svm = SVC()

#     param_grid = { 'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
    # A GridSearchCV round with search parameters was made, and this parameter is optimal.
    # SVM is too slow, then I will use the optimal parameters, if you wish, use the commented line above.
    param_grid = {'C': [1], 'kernel': ['linear'], 'gamma': [1]}

    model = GridSearchCV(svm, param_grid, cv=5, n_jobs=14)
    r = model.fit(x, y.values.ravel())

    print("SVM Best accuracy: ", model.best_score_)
    print("SVM Best parameters: ", model.best_params_)

    # results
    start_time = time.time()
    y_pred = model.predict(x_test)
    end_time = time.time()
    print("SVM Execution time: ", (end_time-start_time))

    from sklearn.metrics import confusion_matrix
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred)
    print("SVM Confusion Matrix: \n", cm)

    accuracy = accuracy_score(y_true, y_pred)
    print("SVM TEST accuracy: ", accuracy)

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    group_names = ['True Fishing', 'False Fishing',
                   'False Sailing', 'True Sailing']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = sn.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
    fig = plot.get_figure()
    fig.savefig("svm_decision_tree_val.png")

    # recall, precision, f1
    from sklearn.metrics import precision_recall_fscore_support
    print("SVM: ", precision_recall_fscore_support(
        y_true, y_pred, average='macro'))
    print("SVM: ", precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore = np.array(
        precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro')

    return model.best_score_, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)

# Random Forest in Trajectory-base data


def random_forest(x, y, x_test, y_test):
    from sklearn import decomposition, datasets
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
<<<<<<< HEAD

    print("*** Random Forest")

    rf = RandomForestClassifier(max_depth=2, random_state=0)

=======

    print("*** Random Forest")

    rf = RandomForestClassifier(max_depth=2, random_state=0)

>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]
    }
<<<<<<< HEAD

    model = GridSearchCV(rf, param_grid, cv=5, n_jobs=14)
    r = model.fit(x, y.values.ravel())

    print("RF Best accuracy: ", model.best_score_)
    print("RF Best parameters: ", model.best_params_)

    # results
    start_time = time.time()
    y_pred = model.predict(x_test)
    end_time = time.time()
    print("RF Execution time: ", (end_time-start_time))

    from sklearn.metrics import confusion_matrix
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred)
    print("RF Confusion Matrix: \n", cm)

    accuracy = accuracy_score(y_true, y_pred)
    print("RF TEST accuracy: ", accuracy)

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    group_names = ['True Fishing', 'False Fishing',
                   'False Sailing', 'True Sailing']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = sn.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
    fig = plot.get_figure()
    fig.savefig("rf_decision_tree_val.png")

    # recall, precision, f1
    from sklearn.metrics import precision_recall_fscore_support
    print("RF: ", precision_recall_fscore_support(
        y_true, y_pred, average='macro'))
    print("RF: ", precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore = np.array(
        precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro')

    return model.best_score_, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)
=======

    model = GridSearchCV(rf, param_grid, cv=5, n_jobs=14)
    r = model.fit(x, y.values.ravel())
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465

    print("RF Best accuracy: ", model.best_score_)
    print("RF Best parameters: ", model.best_params_)

<<<<<<< HEAD
=======
    # results
    start_time = time.time()
    y_pred = model.predict(x_test)
    end_time = time.time()
    print("RF Execution time: ", (end_time-start_time))

    from sklearn.metrics import confusion_matrix
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred)
    print("RF Confusion Matrix: \n", cm)

    accuracy = accuracy_score(y_true, y_pred)
    print("RF TEST accuracy: ", accuracy)

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    group_names = ['True Fishing', 'False Fishing',
                   'False Sailing', 'True Sailing']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = sn.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
    fig = plot.get_figure()
    fig.savefig("rf_decision_tree_val.png")

    # recall, precision, f1
    from sklearn.metrics import precision_recall_fscore_support
    print("RF: ", precision_recall_fscore_support(
        y_true, y_pred, average='macro'))
    print("RF: ", precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore = np.array(
        precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro')

    return model.best_score_, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)


>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
# NN in data Trajectory-base data
def nn(x, y, x_test, y_test, epochs):
    import gc
    from keras import backend as K
    # checkpoint
    checkpoint_nn_path = 'nn_val_checkpoint'

    print("*** Neural Network")

    # Dependencies
    import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import to_categorical
    from tensorflow.keras.layers import Dropout
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.preprocessing import LabelEncoder
    import seaborn as sn
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold, StratifiedKFold

    sc = StandardScaler()
    X = sc.fit_transform(x)

    lb = LabelEncoder()
    lb_trainy = lb.fit_transform(y)
    Y = to_categorical(lb_trainy)

    input_x = len(x.iloc[0])

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_nn_path,
                                  monitor='acc',
                                  mode='max',
                                  save_best_only=True,
                                  save_weights_only=True,
                                  verbose=1
                                  )

    # normalize x_test and y_test
    X_test = sc.fit_transform(x_test)
    Y_test = lb.fit_transform(y_test)
    Y_test = to_categorical(Y_test)

    best_accuracy = 0.0
    mean_time = np.array([])
    mean_accuracy = np.array([])
    n_rounds = 0.0

    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

    for train_index, val_index in skf.split(X, y):
        train_x = X[train_index]
        train_y = Y[train_index]
        val_x = X[val_index]
        val_y = Y[val_index]

        model = Sequential()
        model.add(Dense(32, input_dim=input_x, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['acc'])
#         print(model.summary())

        r = model.fit(train_x, train_y, epochs=epochs, batch_size=64, validation_data=(
            val_x, val_y), callbacks=[cp_callback], verbose=2)

        # Loads the weights
        model.load_weights(checkpoint_nn_path)

        # prediction y_test
        start_time = time.time()
        p = model.predict(val_x)
        end_time = time.time()
        print(p)

        # for nn, for RL comment
        y_pred = np.array([np.argmax(i) for i in p])
        y_true = np.array([np.argmax(i) for i in val_y])

        accuracy = accuracy_score(y_true, y_pred)
        print("NN prediction accuracy: ", accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_weights('nn_best_model.vgg.h5')

        mean_accuracy = np.append(mean_accuracy, accuracy)
        mean_time = np.append(mean_time, end_time-start_time)
        n_rounds += 1

    print("NN BEST prediction accuracy: ", best_accuracy)
    print("NN MEAN prediction accuracy: ", mean_accuracy.mean())
    print("NN STD prediction accuracy: ", mean_accuracy.std())
    print("NN VAR prediction accuracy: ", mean_accuracy.var())
    print("NN MEAN execution time     : ", mean_time.mean())
    print("NN STD execution time     : ", mean_time.std())
    print("NN VAR execution time     : ", mean_time.var())
    print("NN prediction accuracies: ", mean_accuracy)

    # Loads the best weights
    model.load_weights('nn_best_model.vgg.h5')
    start_time = time.time()
    p = model.predict(X_test)
    end_time = time.time()
    print("NN TEST execution time: ", (end_time-start_time))
    # for nn, for RL comment
    y_pred = np.array([np.argmax(i) for i in p])
    y_true = np.array([np.argmax(i) for i in Y_test])

    cm = confusion_matrix(y_true, y_pred)
    print("NN Confusion Matrix: \n", cm)

    accuracy = accuracy_score(y_true, y_pred)
    print("NN TEST accuracy: ", accuracy)

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    group_names = ['True Fishing', 'False Fishing',
                   'False Sailing', 'True Sailing']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = sn.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
    fig = plot.get_figure()
    fig.savefig("cf_nn_val.png")

    # recall, precision, f1
    from sklearn.metrics import precision_recall_fscore_support
    print("NN: ", precision_recall_fscore_support(
        y_true, y_pred, average='macro'))
    print("NN: ", precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore = np.array(
        precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro')

    # history epochs
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.plot(r.history['acc'])
    plt.plot(r.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("nn_val_epochs_history.png", bbox_inches='tight', ax=ax)
    # plt.show()

    K.clear_session()
    gc.collect()
    del model

    return best_accuracy, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)


# CNN model in raw data


def rand_func():
    return random_state_trajs_fishing


N_pixels = 32


def pixel_normalize(lon_p, lat_p, lon_max, lon_min, lat_max, lat_min, N_pixels):
    distance_x = float(abs(lon_max - lon_min))
    distance_y = float(abs(lat_max - lat_min))
    distance_px = float(abs(lon_p - lon_min))
    distance_py = float(abs(lat_p - lat_min))
#     print( "[", distance_x, distance_y, "]", "[", distance_px, distance_py, "]")
    if distance_x > 0:
        norm_px = distance_px / distance_x
#         print("norm_px = ", norm_px)
    else:
        norm_px = 0.0
    if distance_y > 0:
        norm_py = distance_py / distance_y
#         print("norm_py = ", norm_py)
    else:
        norm_py = 0.0

    pixel_x = round(norm_px * N_pixels)
    pixel_y = round(norm_py * N_pixels)

    if pixel_x > N_pixels:
        pixel_x = N_pixels
    if pixel_x < 1:
        pixel_x = 1
    if pixel_y > N_pixels:
        pixel_y = N_pixels
    if pixel_y < 1:
        pixel_y = 1

    return pixel_x, pixel_y


def save_traj_img(traj, directory):
    from sklearn.preprocessing import MinMaxScaler

    lon_max = traj.df.lon.max()
    lon_min = traj.df.lon.min()
    lat_max = traj.df.lat.max()
    lat_min = traj.df.lat.min()

    direction_variance = traj.df.direction.var()
    norm_dir_var = (direction_variance / 100000.0) * 255.0

    pixels_x, pixels_y, pixels_speed = [], [], []
    for index, row in traj.df.iterrows():
        pixel_x, pixel_y = pixel_normalize(
            row['lon'], row['lat'], lon_max, lon_min, lat_max, lat_min, N_pixels)
        pixels_x.append(pixel_x)
        pixels_y.append(pixel_y)
        # *(100000*1.94384) convert to knots
        pixels_speed.append(row.speed * (100000*1.94384))

    from bresenham import bresenham

    # complete the trajectory between points using bresenham algorithm
    for i in range(len(pixels_x) - 1):
        points = list(
            bresenham(pixels_x[i], pixels_y[i], pixels_x[i+1], pixels_y[i+1]))
        for p in points:
            pixels_x.append(p[0])
            pixels_y.append(p[1])

    import numpy
    import matplotlib.pyplot as plt
    from PIL import Image as im

    # image = numpy.asarray([[ (1., 1., 1.) for i in range(N_pixels) ] for j in range(N_pixels) ])
    # x, y, (r,g,b)
    image = np.zeros([N_pixels, N_pixels, 3], dtype=np.uint8)
#     image = np.zeros([N_pixels, N_pixels, 3], dtype=np.uint8)
#     image = np.zeros([N_pixels, N_pixels], dtype=np.uint8)
#     image[:,:] = [255, 255, 255]
    image[:, :] = (255.0, 255.0, 255.0)
#     image[:,:] = 0.0

    size_pixels = len(pixels_x)
    for i in range(size_pixels):
        # paint pixel point normalized
        if i < len(pixels_speed):
            #         image[ pixels_x[i]-1 ][ pixels_y[i]-1 ] = ( ((pixels_speed[i]/50)*255.0), 0., 0.)
            #             color = speed_to_color( pixels_speed[i] )
            color = (pixels_speed[i], pixels_speed[i], pixels_speed[i])
#             color = (0., 0., 255.)
            image[pixels_x[i]-1][pixels_y[i]-1] = color

#             if pixels_x[i]-2 > 0:
#                 image[ pixels_x[i]-2 ][ pixels_y[i]-1 ] = color
#             if pixels_y[i]-2 > 0:
#                 image[ pixels_x[i]-1 ][ pixels_y[i]-2 ] = color
#             if pixels_x[i] < N_pixels:
#                 image[ pixels_x[i]   ][ pixels_y[i]-1 ] = color
#             if pixels_y[i] < N_pixels:
#                 image[ pixels_x[i]-1 ][ pixels_y[i]   ] = color

        else:
            # paint pixel points interpolated
            image[pixels_x[i]-1][pixels_y[i]-1] = (255.0, 0, 0)

    traj_id = str(traj.to_traj_gdf()["traj_id"]).split()[1]
    data = im.fromarray(image)
#     data.save(output_dir_images + directory + '/' + traj_id + '.jpg')
    return np.asarray(data)


def draw_images(trajs_fishing, trajs_no_fishing):
    output_dir_images = "images/"

    n_fishing_trajectories = len(trajs_fishing.trajectories)
    print("n fishing trajectories = ", n_fishing_trajectories)
    n_nonfishing_trajectories = len(trajs_no_fishing.trajectories)
    print("n non fishing trajectories = ", n_nonfishing_trajectories)

    n_count_train_fishing_traj = 0

    img_train = []
    img_train_label = []
    img_test = []
    img_test_label = []

    filtered_trajs_fishing = []
    filtered_trajs_fishing_label = []
    filtered_trajs_nofishing = []
    filtered_trajs_nofishing_label = []

    # This is a threashold for trajectory duration
    # If trajectory was longer than max_duration will be cutted.
    # max_traj_duration=7000
    max_traj_duration = 6000
    # filter trajs fishing
    for traj in trajs_fishing.trajectories:
        # This multiplication is a convertion for knots unit
        if traj.df["vesselType"][0] == "Fishing" and traj.df.speed.mean()*(100000*1.94384) > 1 and traj.df.speed.mean()*(100000*1.94384) < 50:
            t1 = traj.get_start_time()
            t2 = t1 + timedelta(seconds=max_traj_duration)
            # cut trajetory
            traj_tmp = traj.get_segment_between(t1, t2)

            filtered_trajs_fishing.append(save_traj_img(traj_tmp, 'all'))
            filtered_trajs_fishing_label.append("fishing")

    # filter trajs no fishing
    for traj in trajs_no_fishing.trajectories:
        # This multiplication is a convertion for knots unit
        if traj.df["vesselType"][0] != "Fishing" and traj.df.speed.mean()*(100000*1.94384) > 1 and traj.df.speed.mean()*(100000*1.94384) < 50:
            t1 = traj.get_start_time()
            t2 = t1 + timedelta(seconds=max_traj_duration)
            # cut trajetory
            traj_tmp = traj.get_segment_between(t1, t2)

            filtered_trajs_nofishing.append(save_traj_img(traj_tmp, 'all'))
            filtered_trajs_nofishing_label.append("normal")

    n_count_train_fishing_traj = len(filtered_trajs_fishing)
    n_count_train_nofishing_traj = len(filtered_trajs_nofishing)

    # equalize dataset size
    if n_count_train_fishing_traj < n_count_train_nofishing_traj:
        n_size = n_count_train_fishing_traj
    else:
        n_size = n_count_train_nofishing_traj

    # randomize trajectories
    random.shuffle(filtered_trajs_fishing,         rand_func)
    random.shuffle(filtered_trajs_fishing_label,   rand_func)
    random.shuffle(filtered_trajs_nofishing,       rand_func)
    random.shuffle(filtered_trajs_nofishing_label, rand_func)

    ####
    # train
    ####
    # fishing
    img_train = filtered_trajs_fishing[: int(0.8*n_size)]
    img_train_label = filtered_trajs_fishing_label[: int(0.8*n_size)]
    # no fishing
    img_train += filtered_trajs_nofishing[: int(0.8*n_size)]
    img_train_label += filtered_trajs_nofishing_label[: int(0.8*n_size)]

    ####
    # test
    ####
    # fishing
    img_test = filtered_trajs_fishing[int(0.8*n_size): n_size]
    img_test_label = filtered_trajs_fishing_label[int(
        0.8*n_count_train_fishing_traj): n_size]
    # no fishing
    img_test += filtered_trajs_nofishing[int(0.8*n_size): n_size]
    img_test_label += filtered_trajs_nofishing_label[int(0.8*n_size): n_size]
<<<<<<< HEAD
<<<<<<< HEAD
=======

    return np.array(img_train), np.array(img_train_label), np.array(img_test), np.array(img_test_label)
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======

    return np.array(img_train), np.array(img_train_label), np.array(img_test), np.array(img_test_label)
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465

    return np.array(img_train), np.array(img_train_label), np.array(img_test), np.array(img_test_label)

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
def cnn(trajs_fishing, trajs_no_fishing, epochs, load_images_dataset=False):
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import to_categorical
    from keras.layers import Input, Lambda, Dense, Flatten
    from keras.models import Model
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing import image
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from glob import glob
    import tensorflow as tf
    from sklearn.model_selection import KFold, StratifiedKFold
    import gc
    from keras import backend as K

    print("*** Convolutional Neural Network")

    # folders path
    train_data_dir = 'images/train'
    validation_data_dir = 'images/test'
    checkpoint_cnn_path = "cnn_checkpoint"

    # path to images numpy array in files
    img_train_np_file = train_data_dir + "/" + \
        "train_images_variance.npy" + str(N_pixels)
    img_train_label_np_file = train_data_dir + "/" + \
        "train_images_label_variance.npy" + str(N_pixels)
    img_test_np_file = validation_data_dir + "/" + \
        "val_images_variance.npy" + str(N_pixels)
    img_test_label_np_file = validation_data_dir + "/" + \
        "val_images_label_variance.npy" + str(N_pixels)

    # load images from binaries files or use the prior builded trajectories
#     load_images_dataset=False

    if load_images_dataset:
        # load images from numpy array in files
        # Use this only when you don't wanna run all code again, eg. to increase epochs.
        print("loading images...")
        img_train_np = np.load(img_train_np_file)
        img_train_label_np = np.load(img_train_label_np_file)
        img_test_np = np.load(img_test_np_file)
        img_test_label_np = np.load(img_test_label_np_file)
        print("finished loading images.")
    else:
        print("building images...")
        img_train, img_train_label, img_test, img_test_label = draw_images(
            trajs_fishing, trajs_no_fishing)
        # Build Images labels
        let = LabelEncoder()
        img_train_label_np = let.fit_transform(img_train_label)
        img_train_label_np = to_categorical(img_train_label_np)

        les = LabelEncoder()
        img_test_label_np = les.fit_transform(img_test_label)
        img_test_label_np = to_categorical(img_test_label_np)

        # convert array to numpy
        img_train_np = np.array(img_train)
        img_test_np = np.array(img_test)

        # save images numpy array in files
        with open(img_train_np_file, 'wb') as f:
            np.save(f, img_train_np)
        with open(img_train_label_np_file, 'wb') as f:
            np.save(f, img_train_label_np)

        with open(img_test_np_file, 'wb') as f:
            np.save(f, img_test_np)
        with open(img_test_label_np_file, 'wb') as f:
            np.save(f, img_test_label_np)
        print("finished building images.")

    print("executing CNN Model...")
    best_accuracy = 0.0
    mean_time = np.array([])
    mean_accuracy = np.array([])
    n_rounds = 0.0
    # run 5 times and calc mean
    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
    for train_index, val_index in skf.split(img_train_np, img_train_label):
        train_x = img_train_np[train_index]
        train_y = img_train_label_np[train_index]
        val_x = img_train_np[val_index]
        val_y = img_train_label_np[val_index]

        # CNN Architecture
        vgg = VGG16(include_top=False, weights='imagenet',
                    input_shape=(N_pixels, N_pixels, 3))
        print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))

        # Now we will be training only the classifiers (FC layers)
        for layer in vgg.layers:
            layer.trainable = False

        x = Flatten()(vgg.output)

        prediction = Dense(2, activation='softmax')(x)
        model = Model(inputs=vgg.input, outputs=prediction)

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['acc'])

        batch_size = 10
        gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_input
        )

        train_generator = gen.flow(
            train_x, train_y, shuffle=True, batch_size=batch_size)
        test_generator = gen.flow(
            val_x, val_y, shuffle=False, batch_size=batch_size)

        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_cnn_path,
                                      monitor='val_acc',
                                      mode='max',
                                      save_best_only=True,
                                      save_weights_only=True,
                                      verbose=1)

        r = model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=epochs,
            steps_per_epoch=len(train_x) // batch_size,
            validation_steps=len(val_x) // batch_size,
            workers=14,
            callbacks=[cp_callback],
            verbose=2
        )

        # Loads the weights
        model.load_weights(checkpoint_cnn_path)

        start_time = time.time()
        p = model.predict(test_generator)
        end_time = time.time()
        print("CNN execution time: ", (end_time-start_time))

        # for nn, for RL comment
        y_pred = np.array([np.argmax(i) for i in p])
        y_true = np.array([np.argmax(i) for i in val_y])

#         print("p: ", len(p) )
#         print("img_test_label_np: ", len(img_test_label_np))

        accuracy = accuracy_score(y_true, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_weights('cnn_best_model.vgg.h5')

        mean_accuracy = np.append(mean_accuracy, accuracy)
        mean_time = np.append(mean_time, end_time-start_time)
        n_rounds += 1

    print("CNN BEST prediction accuracy: ", best_accuracy)
    print("CNN MEAN prediction accuracy: ", mean_accuracy.mean())
    print("CNN STD prediction accuracy: ", mean_accuracy.std())
    print("CNN VAR prediction accuracy: ", mean_accuracy.var())
    print("CNN MEAN execution time     : ", mean_time.mean())
    print("CNN STD execution time     : ", mean_time.std())
    print("CNN VAR execution time     : ", mean_time.var())
    print("CNN prediction accuracies: ", mean_accuracy)

    # Loads the best model
    model.load_weights('cnn_best_model.vgg.h5')

    valid_generator = gen.flow(
        img_test_np, img_test_label_np, shuffle=False, batch_size=batch_size)
    start_time = time.time()
    p = model.predict(valid_generator)
    end_time = time.time()
    print("CNN execution time: ", (end_time-start_time))

    # for nn, for RL comment
    y_pred = np.array([np.argmax(i) for i in p])
    y_true = np.array([np.argmax(i) for i in img_test_label_np])

    cm = confusion_matrix(y_true, y_pred)
    print("CNN Confusion Matrix: \n", cm)

    accuracy = accuracy_score(y_true, y_pred)
    print("CNN TEST accuracy: ", accuracy)

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    group_names = ['True Fishing', 'False Fishing',
                   'False Sailing', 'True Sailing']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = sn.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
    fig = plot.get_figure()
    fig.savefig("cf_cnn_val.png")

    # recall, precision, f1
    from sklearn.metrics import precision_recall_fscore_support
    print("CNN: ", precision_recall_fscore_support(
        y_true, y_pred, average='macro'))
    print("CNN: ", precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore = np.array(
        precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro')

    # history epochs
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.plot(r.history['acc'])
    plt.plot(r.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("cnn_val_epochs_history.png", bbox_inches='tight', ax=ax)
    # plt.show()

    K.clear_session()
    gc.collect()
    del model
<<<<<<< HEAD
<<<<<<< HEAD
=======

    return best_accuracy, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======

    return best_accuracy, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465

    return best_accuracy, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
def speed_to_color(speed):
    ret = float(speed)
    return ret


# RNN Model in raw data
def prepare_data_rnn(trajs_fishing, trajs_no_fishing):

    n_fishing_trajectories = len(trajs_fishing.trajectories)
    print("n fishing trajectories = ", n_fishing_trajectories)

    n_nonfishing_trajectories = len(trajs_no_fishing.trajectories)
    print("n non fishing trajectories = ", n_nonfishing_trajectories)

    n_count_train_fishing_traj = 0

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    filtered_trajs_fishing = []
    filtered_trajs_fishing_label = []
    filtered_trajs_nofishing = []
    filtered_trajs_nofishing_label = []

    # filter trajs fishing
    for traj in trajs_fishing.trajectories:
        if traj.df["vesselType"][0] == "Fishing" and traj.df.speed.mean()*(100000*1.94384) > 1 and traj.df.speed.mean()*(100000*1.94384) < 50:
            filtered_trajs_fishing.append(
                traj.df[['lat', 'lon', 'veloc', 'rumo']].to_numpy())
            filtered_trajs_fishing_label.append("fishing")

    # filter trajs no fishing
    for traj in trajs_no_fishing.trajectories:
        if traj.df["vesselType"][0] != "Fishing" and traj.df.speed.mean()*(100000*1.94384) > 1 and traj.df.speed.mean()*(100000*1.94384) < 50:
            filtered_trajs_nofishing.append(
                traj.df[['lat', 'lon', 'veloc', 'rumo']].to_numpy())
            filtered_trajs_nofishing_label.append("normal")

    # verify the size of dataset and set smaller size
    n_count_train_fishing_traj = len(filtered_trajs_fishing)
    n_count_train_nonfishing_traj = len(filtered_trajs_nofishing)

    if n_count_train_fishing_traj < n_count_train_nonfishing_traj:
        n_size = n_count_train_fishing_traj
    else:
        n_size = n_count_train_nonfishing_traj

    # random trajectories
    random.shuffle(filtered_trajs_fishing,         rand_func)
    random.shuffle(filtered_trajs_fishing_label,   rand_func)
    random.shuffle(filtered_trajs_nofishing,       rand_func)
    random.shuffle(filtered_trajs_nofishing_label, rand_func)

    # train
    # fishing
    train_x = filtered_trajs_fishing[: int(0.8*n_size)]
    train_y = filtered_trajs_fishing_label[: int(0.8*n_size)]

    # no fishing
    train_x += filtered_trajs_nofishing[: int(0.8*n_size)]
    train_y += filtered_trajs_nofishing_label[: int(0.8*n_size)]

    # test
    # fishing
    test_x = filtered_trajs_fishing[int(0.8*n_size): n_size]
    test_y = filtered_trajs_fishing_label[int(0.8*n_size): n_size]
    # no fishing
    test_x += filtered_trajs_nofishing[int(0.8*n_size): n_size]
    test_y += filtered_trajs_nofishing_label[int(0.8*n_size): n_size]
<<<<<<< HEAD
<<<<<<< HEAD

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
=======
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

<<<<<<< HEAD
=======

>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
# RNN in raw data
def rnn(trajs_fishing, trajs_no_fishing, epochs):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.preprocessing import sequence
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from keras.utils import to_categorical
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.model_selection import KFold, StratifiedKFold
    import time
    import gc
    from keras import backend as K

    print("*** Recurent Neural Network")
    # truncate and pad input sequences
    max_trajectory_length = 500

    x, y, test_x, test_y = prepare_data_rnn(trajs_fishing, trajs_no_fishing)

    test_X = sequence.pad_sequences(test_x, maxlen=max_trajectory_length)
    lb = LabelEncoder()
    lb_valy = lb.fit_transform(test_y)
    test_Y = to_categorical(lb_valy)

    best_accuracy = 0.0
    mean_time = np.array([])
    mean_accuracy = np.array([])
    n_rounds = 0.0
    # run 5 times and calc mean
    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

    for train_index, val_index in skf.split(x, y):
        train_x = x[train_index]
        train_y = y[train_index]
        val_x = x[val_index]
        val_y = y[val_index]

        X_train = sequence.pad_sequences(train_x, maxlen=max_trajectory_length)
        X_val = sequence.pad_sequences(val_x, maxlen=max_trajectory_length)

        # Set label in columns format
        lb = LabelEncoder()
        lb_trainy = lb.fit_transform(train_y)
        Y_train = to_categorical(lb_trainy)
        lb = LabelEncoder()
        lb_testy = lb.fit_transform(val_y)
        Y_val = to_categorical(lb_testy)

        ##
        # RNN Architecture
        ##
        model = Sequential()
        # model.add(Embedding(5000, embedding_vecor_length, input_length=(max_trajectory_length*4) ))
        # model.add(Embedding(5000, embedding_vecor_length, input_shape=(max_trajectory_length, 4) ))
        # model.add(Dropout(0.2))
        model.add(LSTM(100, input_shape=(max_trajectory_length, 4)))
        # model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        # model.compile(loss='macro_crossentropy', optimizer='adam', metrics=['acc'])
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['acc'])
        print(model.summary())

        ##
        # Fit the model
        ##
        checkpoint_path = '.'
        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      monitor='val_acc',
                                      mode='max',
                                      save_best_only=True,
                                      save_weights_only=True,
                                      verbose=1
                                      )

        r = model.fit(X_train, Y_train, epochs=epochs, batch_size=64,
                      validation_data=(X_val, Y_val),  callbacks=[cp_callback])

        # Loads the best accuracy weights
        model.load_weights(checkpoint_path)

        start_time = time.time()
        p = model.predict(X_val)
        end_time = time.time()
        print("RNN execution time: ", (end_time-start_time))

        y_true = np.array([np.argmax(i) for i in Y_val])
        y_pred = np.array([np.argmax(i) for i in p])

        accuracy = accuracy_score(y_true, y_pred)
        print("RNN prediction accuracy: ", accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_weights('rnn_best_model.vgg.h5')

        mean_accuracy = np.append(mean_accuracy, accuracy)
        mean_time = np.append(mean_time, end_time-start_time)
        n_rounds += 1

    print("RNN BEST prediction accuracy: ", best_accuracy)
    print("RNN MEAN prediction accuracy: ", mean_accuracy.mean())
    print("RNN STD prediction accuracy: ", mean_accuracy.std())
    print("RNN VAR prediction accuracy: ", mean_accuracy.var())
    print("RNN MEAN execution time     : ", mean_time.mean())
    print("RNN STD execution time     : ", mean_time.std())
    print("RNN VAR execution time     : ", mean_time.var())
    print("RNN accuracies: ", mean_accuracy)

    # Loads the best accuracy weights
    model.load_weights('rnn_best_model.vgg.h5')

    ##
    # Results
    ##

    start_time = time.time()
    p = model.predict(test_X)
    end_time = time.time()
    print("RNN execution time: ", (end_time-start_time))

    y_true = np.array([np.argmax(i) for i in test_Y])
    y_pred = np.array([np.argmax(i) for i in p])

    cm = confusion_matrix(y_true, y_pred)
    print("RNN Confusion Matrix: \n", cm)

    accuracy = accuracy_score(y_true, y_pred)
    print("RNN TEST accuracy: ", accuracy)

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    group_names = ['True Fishing', 'False Fishing',
                   'False Sailing', 'True Sailing']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = sn.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
    fig = plot.get_figure()
    fig.savefig("cf_rnn_val.png")

    # recall, precision, f1
    from sklearn.metrics import precision_recall_fscore_support
    print("RNN: ", precision_recall_fscore_support(
        y_true, y_pred, average='macro'))
    print("RNN: ", precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore = np.array(
        precision_recall_fscore_support(y_true, y_pred, average=None))
    precision_recall_fscore_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro')

    # history epochs
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.plot(r.history['acc'])
    plt.plot(r.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("rnn_val_epochs_history.png", bbox_inches='tight', ax=ax)
    # plt.show()

    K.clear_session()
    gc.collect()
    del model

    return best_accuracy, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy, (end_time-start_time)


####################
# Call Models
####################

# def disable_gpu( ):
#     try:
#         # Disable all GPUS
#         tf.config.set_visible_devices([], 'GPU')
#         visible_devices = tf.config.get_visible_devices()
#         for device in visible_devices:
#             assert device.device_type != 'GPU'
#     except:
#         # Invalid device or cannot modify virtual devices once initialized.
#         print("Invalid device or cannot modify virtual devices once initialized.")
#         pass

#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#######################################
# MAIN
# Dataset
# | train | validation | test (20%) |
#######################################

#disable_gpu( )

# load dataset from file
df_gfw = load_dataset()
# transform data frame in geo data frame
gdf = load_gdf(df_gfw)

# limit the dataset
# we have 2.6M fishing AIS messages and 17M non fishing AIS messages
# We need equalize both classes
len_gdf_only_fishing = 2600000
len_gdf_no_fishing = 3000000
gdf_only_fishing, gdf_no_fishing, gdf_filtered = filter_gdf(
    gdf, len_gdf_only_fishing, len_gdf_no_fishing)

# transform gdf in moving pandas trajectories. Or load from file the prior built.
<<<<<<< HEAD
<<<<<<< HEAD
trajs_fishing, trajs_no_fishing = load_or_build_trajectories( len_gdf_only_fishing, load_trajectories_collection_from_file=True )
=======
trajs_fishing, trajs_no_fishing = load_or_build_trajectories(
    len_gdf_only_fishing, load_trajectories_collection_from_file=False)
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======
trajs_fishing, trajs_no_fishing = load_or_build_trajectories(
    len_gdf_only_fishing, load_trajectories_collection_from_file=False)
>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465

# we have 12K fishing trajectories and 108K non fishing trajectories
# limit trajs non fishing to avoid waste unnecessary processing; 
trajs_no_fishing = mpd.TrajectoryCollection(
    trajs_no_fishing.trajectories[:20000])

# trajs info fishing (trajectory-based data)
traj_info_fishing = init_trajectory_data(trajs_fishing)
n_traj_info_fishing = len(traj_info_fishing)
print("n_traj_info_fishing: ", n_traj_info_fishing)
# trajs info no fishing (trajectory-based data)
traj_info_no_fishing = init_trajectory_data(trajs_no_fishing)
n_traj_info_nofishing = len(traj_info_no_fishing)
print("n_traj_info_nofishing", n_traj_info_nofishing)
# create traj info for all trajectories collections
df_trajs_info = pd.concat([traj_info_fishing, traj_info_no_fishing])

# set labels fishing or non fishing
set_label_trajectory_info(df_trajs_info)

# save plots about dataset numbers
plot_statistics_dataset(df_trajs_info)

random_state_trajs_fishing += 0.01
# filter, random rows and get xval, yval from traj info
df_trajs_info_filtered, x_test, y_test = filter_trajs_info(
    df_trajs_info, random_state_trajs_fishing_info)
data_model = df_trajs_info_filtered[[
    'duracao', 'varRumo', 'varVeloc', 'activity', 'traj_len', 'n_points']]

# set x and y to models using trajectory-based
x = data_model[['duracao', 'varRumo', 'varVeloc', 'traj_len', 'n_points']]
y = data_model[['activity']]

trajs_fishing, trajs_no_fishing = filter_trajs(trajs_fishing, trajs_no_fishing)

# %%

# Results Arrays
models = {'LR': 0, 'DT': 1, 'SVM': 2, 'RF': 3, 'NN': 4, 'CNN': 5, 'RNN': 6}
# models = {'LR': 0, 'DT': 1, 'RF': 2, 'NN': 3, 'CNN': 4, 'RNN': 5}
countModel = int(0)
scoreTrain = []
classFish = []
classNonFish = []
scoreMacro = []
scoreTest = []
timePrediction = []
timeTrain = []

# number of epochs to train NN, CNN and RNN
epochs = 50

###############################
# Trajectory-based data models
###############################
start_time = time.time()
#model.best_score_, precision_recall_fscore[:, 0], precision_recall_fscore[:, 1], precision_recall_fscore_macro, accuracy
sTrain, prfClassFish, prfClassNonFish, prfMacro, sTest, timep = logistic_regression(
    x, y, x_test, y_test)
end_time = time.time()
scoreTrain.append(sTrain)
classFish.append(prfClassFish)
classNonFish.append(prfClassNonFish)
scoreMacro.append(prfMacro)
scoreTest.append(sTest)
timePrediction.append(timep)
timeTrain.append((end_time-start_time))
print("LR train TIME: ", (end_time-start_time), " seconds")
<<<<<<< HEAD
print()

start_time = time.time()
sTrain, prfClassFish, prfClassNonFish, prfMacro, sTest, timep = decision_tree(
    x, y, x_test, y_test)
end_time = time.time()
scoreTrain.append(sTrain)
classFish.append(prfClassFish)
classNonFish.append(prfClassNonFish)
scoreMacro.append(prfMacro)
scoreTest.append(sTest)
timePrediction.append(timep)
timeTrain.append((end_time-start_time))
print("DT train TIME: ", (end_time-start_time), " seconds")
<<<<<<< HEAD
print()

start_time = time.time()
sTrain, prfClassFish, prfClassNonFish, prfMacro, sTest, timep = svm(x, y, x_test, y_test)
end_time = time.time()
scoreTrain.append  ( sTrain          )
classFish.append   ( prfClassFish    )
classNonFish.append( prfClassNonFish )
scoreMacro.append  ( prfMacro  )
scoreTest.append   ( sTest     )
timePrediction.append( timep   )
timeTrain.append((end_time-start_time))
print("SVM train TIME: ", (end_time-start_time), " seconds" )
print()

=======
print()

start_time = time.time()
sTrain, prfClassFish, prfClassNonFish, prfMacro, sTest, timep = svm(x, y, x_test, y_test)
end_time = time.time()
scoreTrain.append  ( sTrain          )
classFish.append   ( prfClassFish    )
classNonFish.append( prfClassNonFish )
scoreMacro.append  ( prfMacro  )
scoreTest.append   ( sTest     )
timePrediction.append( timep   )
timeTrain.append((end_time-start_time))
print("SVM train TIME: ", (end_time-start_time), " seconds" )
print()

>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======
print()

start_time = time.time()
sTrain, prfClassFish, prfClassNonFish, prfMacro, sTest, timep = decision_tree(
    x, y, x_test, y_test)
end_time = time.time()
scoreTrain.append(sTrain)
classFish.append(prfClassFish)
classNonFish.append(prfClassNonFish)
scoreMacro.append(prfMacro)
scoreTest.append(sTest)
timePrediction.append(timep)
timeTrain.append((end_time-start_time))
print("DT train TIME: ", (end_time-start_time), " seconds")
print()

start_time = time.time()
sTrain, prfClassFish, prfClassNonFish, prfMacro, sTest, timep = svm(x, y, x_test, y_test)
end_time = time.time()
scoreTrain.append  ( sTrain          )
classFish.append   ( prfClassFish    )
classNonFish.append( prfClassNonFish )
scoreMacro.append  ( prfMacro  )
scoreTest.append   ( sTest     )
timePrediction.append( timep   )
timeTrain.append((end_time-start_time))
print("SVM train TIME: ", (end_time-start_time), " seconds" )
print()

>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465

start_time = time.time()
sTrain, prfClassFish, prfClassNonFish, prfMacro, sTest, timep = random_forest(
    x, y, x_test, y_test)
end_time = time.time()
scoreTrain.append(sTrain)
classFish.append(prfClassFish)
classNonFish.append(prfClassNonFish)
scoreMacro.append(prfMacro)
scoreTest.append(sTest)
timePrediction.append(timep)
timeTrain.append((end_time-start_time))
print("RF train TIME: ", (end_time-start_time), " seconds")
print()


start_time = time.time()
sTrain, prfClassFish, prfClassNonFish, prfMacro, sTest, timep = nn(
    x, y, x_test, y_test, epochs)
end_time = time.time()
scoreTrain.append(sTrain)
classFish.append(prfClassFish)
classNonFish.append(prfClassNonFish)
scoreMacro.append(prfMacro)
scoreTest.append(sTest)
timePrediction.append(timep)
timeTrain.append((end_time-start_time))
print("NN train TIME: ", (end_time-start_time), " seconds")
print()

########################
# Raw-based data models
########################
start_time = time.time()
sTrain, prfClassFish, prfClassNonFish, prfMacro, sTest, timep = cnn(
    trajs_fishing, trajs_no_fishing, epochs, load_images_dataset=False)
end_time = time.time()
scoreTrain.append(sTrain)
classFish.append(prfClassFish)
classNonFish.append(prfClassNonFish)
scoreMacro.append(prfMacro)
scoreTest.append(sTest)
timePrediction.append(timep)
timeTrain.append((end_time-start_time))
print("CNN train TIME: ", (end_time-start_time), " seconds")
print()

start_time = time.time()
sTrain, prfClassFish, prfClassNonFish, prfMacro, sTest, timep = rnn(
    trajs_fishing, trajs_no_fishing, epochs)
end_time = time.time()
scoreTrain.append(sTrain)
classFish.append(prfClassFish)
classNonFish.append(prfClassNonFish)
scoreMacro.append(prfMacro)
scoreTest.append(sTest)
timePrediction.append(timep)
timeTrain.append((end_time-start_time))
print("RNN train TIME: ", (end_time-start_time), " seconds")
print()

#################
# Print Results
#################
train_count = df_trajs_info_filtered['activity'].value_counts()
val_count = y_test['activity'].value_counts()

print("Fishing Trajectories Train: ", train_count[0])
print("Non Fishing Trajectories Train: ", train_count[1])
print("Fishing Trajectories Test: ", val_count[0])
print("Non Fishing Trajectories Test: ", val_count[1])

print("\n** Using Trajectory-based Data")
print("Logistic Regression Accuracy :",
      scoreTrain[0],  "Recall, Precision, F1: ", scoreMacro[0])
print("Decision Tree Accuracy       :",
      scoreTrain[1],  "Recall, Precision, F1: ", scoreMacro[1])
print("SVM Accuracy                 :", scoreTrain[2], "Recall, Precision, F1: ", prfMacro[2] )
print("RF Accuracy                  :",
      scoreTrain[3],  "Recall, Precision, F1: ", scoreMacro[3])
print("Neural Network Accuracy      :",
      scoreTrain[4],  "Recall, Precision, F1: ", scoreMacro[4])
print("\n** Using Raw Data")
print("Convolutional Neural Network :",
      scoreTrain[5], "Recall, Precision, F1: ", scoreMacro[5])
print("Recurrent Neural Network     :",
      scoreTrain[6], "Recall, Precision, F1: ", scoreMacro[6])
print("\n*******************\n\n")

# TODO
# Graphic with resuts and table!!!!
d = models.copy()
for k, i in models.items():
    d[k] = np.array(classFish[i])

df_graf_fish = pd.DataFrame(
    d, index=['Recall', 'Precision', 'F1', 'Support'])
df_graf_fish.loc["Accuracy"] = scoreTest
df_graf_fish.loc["Time Prediction"] = timePrediction
hv = df_graf_fish[
    (df_graf_fish.index != 'Support') &
    (df_graf_fish.index != 'Time Prediction')
].hvplot.bar(rot=90, title='Fishing Class', height=800, width=1400 )
hvplot.save(hv, 'class_fishing.png')

print("Fishing Class")
print(df_graf_fish)

d = models.copy()
for k, i in models.items():
    d[k] = np.array(classNonFish[i])

df_graf_nonfish = pd.DataFrame(
    d, index=['Recall', 'Precision', 'F1', 'Support'])
df_graf_nonfish.loc["Accuracy"] = scoreTest
df_graf_nonfish.loc["Time Prediction"] = timePrediction
hv = df_graf_nonfish[
    (df_graf_nonfish.index != 'Support') &
    (df_graf_nonfish.index != 'Time Prediction')
].hvplot.bar(rot=90, title='Sailing Class', height=800, width=1400)
hvplot.save(hv, 'class_nonfishing.png')

print("\nSailing")
print(df_graf_nonfish)

d = models.copy()
for k, i in models.items():
    d[k] = np.array(scoreTrain[i])
df_score_train = pd.DataFrame(
    d, index=['Accuracy'])
df_score_train.loc["Time Prediction"] = timePrediction
df_score_train.loc["Time Trainning"] = timeTrain

print("\nScore Train")
print(df_score_train)

d = models.copy()
for k, i in models.items():
    d[k] = np.array(scoreMacro[i])
df_score_macro = pd.DataFrame(
    d, index=['Recall', 'Precision', 'F1', 'Support'])

print("\nScore Macro")
print(df_score_macro)

## save table images
import dataframe_image as dfi

df_styled = df_graf_fish[
    (df_graf_fish.index != 'Support') 
    ].T[['Precision', 'Recall', 'F1', 'Accuracy', 'Time Prediction']].style.background_gradient().set_table_styles([dict(selector='th', props=[('text-align', 'center')])]) #adding a gradient based on values in cell
df_styled.set_properties(**{'text-align': 'center'})
dfi.export(df_styled,"table_fishing_class.png", dpi=500)

df_styled = df_graf_nonfish[
    (df_graf_nonfish.index != 'Support') 
    ].T[['Precision', 'Recall', 'F1', 'Accuracy', 'Time Prediction']].style.background_gradient().set_table_styles([dict(selector='th', props=[('text-align', 'center')])]) #adding a gradient based on values in cell
df_styled.set_properties(**{'text-align': 'center'})
dfi.export(df_styled,"table_nonfishing_class.png", dpi=500)

df_styled = df_score_train.T.style.background_gradient().set_table_styles([dict(selector='th', props=[('text-align', 'center')])]) #adding a gradient based on values in cell
df_styled.set_properties(**{'text-align': 'center'})
dfi.export(df_styled,"table_score_train.png", dpi=500)

df_styled = df_score_macro.T[['Precision', 'Recall', 'F1', 'Support']].style.background_gradient().set_table_styles([dict(selector='th', props=[('text-align', 'center')])]) #adding a gradient based on values in cell
df_styled.set_properties(**{'text-align': 'center'})
dfi.export(df_styled,"table_score_macro.png", dpi=500)
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465
=======

>>>>>>> 35cd80e3252a5336b9140bf890775eebd8022465

# # # Sample Images

# # trajs_no_fishing.trajectories[250].hvplot(geo=True, tiles='OSM', line_width=2, c='mmsi', hover_cols=['mmsi', 'dh', 'veloc'], cmap='rainbow')
# trajs_no_fishing.get_trajectory('3352713000_16').hvplot(geo=True, tiles='OSM', line_width=6, c='mmsi', hover_cols=['mmsi', 'dh', 'veloc'], cmap='fire')

# df_trajs_info_filtered[df_trajs_info_filtered['activity'] == 'normal'].sort_values(by="n_points",ascending=True)

# # -42.9294406, -22.9845024
# # -42.9294406, -23.2763364
# # -42.4047420, -23.2826461
# # -42.4143569, -22.9554107
# # -42.9294406, -22.9845024

# pol = Polygon([[-42.9294406, -22.9845024], [-42.9294406, -23.2763364], [-42.4047420, -23.2826461], [-42.4143569, -22.9554107], [-42.9294406, -22.9845024]])
# trajs_pol = trajs_no_fishing.clip( pol )
# gdf_pol = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[pol])

# plot = mpd.TrajectoryCollection( trajs_pol.trajectories[10:50] ).hvplot(geo=True, tiles='OSM', line_width=6, c='mmsi', hover_cols=['mmsi', 'dh', 'veloc'], cmap='fire') * gdf_pol.hvplot(geo=True, alpha=0.1, cmap='rainbow')
# hvplot.show(plot)

# %%
