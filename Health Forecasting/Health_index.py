from pandas import Series
from pandas import read_csv
from pandas import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
import os
import numpy as np

save_info = 1
# Root path to save the generated figure
root_path = '/home/liming/Dropbox/US_Steel/USS-RF-Fan-Data-Analytics/_13_Preliminary-results/LSTM-preciction/multi-step-prediction/compare-v3/health_index'
sensor_names = {
    'Accelerator Pedal','Steering Wheel Angle', 'Brake Pressure'
}
operating_range = {
    'Accelerator Pedal':(65,40,90),'Steering Wheel Angle':(110,90,130),'Brake Pressure':(63,40,90)
}
weights = {
    'Accelerator Pedal':1,'Steering Wheel Angle':1,'Brake Pressure':0
}
line_colors = {
    'Accelerator Pedal':'b','Steering Wheel Angle':'g','Brake Pressure':'r'
}
sensor_acronym = {
    'AccP': 'P1', 'SWA': 'T1', 'BP': 'P2'
}

def load_dataset(paths):
    series = {}
    sensor_names = []
    for path in paths:
        serie = read_csv(path, sep=',')

        serie.time = [datetime.datetime.strptime(
            i, "%Y-%m-%d") for i in serie.time]
        sensor_name = os.path.basename(path).split('.')[0]
        series[sensor_name] = serie
    return series

def get_paths():
    root_health_index_all = os.path.join(os.curdir, 'health_index', 'health_index_all')
    files = os.listdir(root_health_index_all)
    paths_health_index_all = [os.path.join(root_health_index_all, s) for s in files]

    root_health_index_pred = os.path.join(os.curdir, 'health_index', 'health_index_pred')
    files = os.listdir(root_health_index_pred)
    paths_health_index_pred = [os.path.join(root_health_index_pred, s) for s in files]

    return paths_health_index_all, paths_health_index_pred

def plot_health_index_combined(series, overall_health_index, path):
    """
    Combine all health index in one figure
    """
    label_fontsize = 35
    legend_fontsize = 18
    axis_fontsize = 30
    linewidth = 3

    fig = plt.figure()
    axis = fig.add_subplot(1,1,1)
    # axis.xaxis_date()
    # axis.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    for key in series.keys():
        if key in ['Accelerator Pedal','Steering Wheel Angle','Brake Pressure']:
            continue
        # plt.plot(series[key].time,series[key].health_index, label=sensor_acronym[key], linewidth=linewidth,alpha=0.3, color=line_colors[key])
        plt.plot(np.arange(len(series[key].health_index)), series[key].health_index, label=sensor_acronym[key], linewidth=linewidth,alpha=0.3, color=line_colors[key])
    plt.plot(np.arange(len(overall_health_index.values)),overall_health_index.values, label = 'overall', linewidth=linewidth+3, color = 'black')
    plt.xlabel('Days', fontsize=label_fontsize)
    plt.ylabel('Health Index', fontsize=label_fontsize)
    plt.title('Health Index', fontsize=label_fontsize)
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.legend(fontsize=legend_fontsize,bbox_to_anchor=(1.13,0.5), loc="center right")
    # plt.show()
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    fig.subplots_adjust(right=0.88)
    if save_info:
        fig.savefig(os.path.join(path, 'health_index_combined.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

def plot_health_index_separated(series, path):
    """
    For each health index, generate a figure
    """
    label_fontsize = 35
    legend_fontsize = 20
    axis_fontsize = 30
    linewidth = 5

    for key in series.keys():
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.xaxis_date()
        axis.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        plt.plot(series[key].time,series[key].health_index, label=sensor_acronym[key], linewidth=linewidth)
        # plt.plot(overall_health_index, label='overall_health_index', linewidth=linewidth)
        plt.xlabel('Health Index', fontsize=label_fontsize)
        plt.ylabel('Health Index', fontsize=label_fontsize)
        axis.set_ylim([0, 1])
        plt.title('Health Index: ' + sensor_acronym[key], fontsize=label_fontsize)
        plt.xticks(fontsize=axis_fontsize)
        plt.yticks(fontsize=axis_fontsize)
        plt.legend(fontsize=legend_fontsize)
        # plt.show()
        if save_info:
            fig.set_size_inches(18.5, 10.5)
            fig.savefig(os.path.join(path, 'health_index_' + key + '.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)
    # plot overall health index
    # fig = plt.figure()
    # axis = fig.add_subplot(1, 1, 1)
    # axis.xaxis_date()
    # axis.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    # plt.plot(overall_health_index, label='overall_health_index', linewidth=linewidth)
    # plt.xlabel('Days', fontsize=label_fontsize)
    # plt.ylabel('Health Index', fontsize=label_fontsize)
    # plt.title('Overall Health Index', fontsize=label_fontsize)
    # plt.xticks(fontsize=axis_fontsize)
    # plt.yticks(fontsize=axis_fontsize)
    # plt.show()
    # fig.set_size_inches(18.5, 10.5)
    # fig.savefig(os.path.join(root_path, 'health_index_overall.png'), bbox_inches='tight', dpi=150)
    # plt.close(fig)

def plot_health_index_overlay(series_all,series_pred, path):
    label_fontsize = 35
    legend_fontsize = 20
    axis_fontsize = 30
    linewidth = 5

    for key in series_all.keys():
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.xaxis_date()
        axis.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        plt.plot(series_all[key].time,series_all[key].health_index, label='ground truth health index', linewidth=linewidth)
        plt.plot(series_pred[key].time,series_pred[key].health_index, label='predicted health index', linewidth=linewidth)
        # plt.plot(overall_health_index, label='overall_health_index', linewidth=linewidth)
        plt.xlabel('Health Index', fontsize=label_fontsize)
        plt.ylabel('Health Index', fontsize=label_fontsize)
        axis.set_ylim([0,1])
        plt.title('Health Index: ' + key, fontsize=label_fontsize)
        plt.xticks(fontsize=axis_fontsize)
        plt.yticks(fontsize=axis_fontsize)
        plt.legend(fontsize=legend_fontsize)
        # plt.show()
        if save_info:
            fig.set_size_inches(18.5, 10.5)
            fig.savefig(os.path.join(path, 'health_index_' + key + '.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)


def get_combined_health_index(series):
    """
    plot all health index in one figure
    """
    # health_indices = [series[key].health_index_pred.values for key in series.keys()]
    #
    # for i in zip(health_indices[0], health_indices[1],health_indices[2],
    #              health_indices[3],health_indices[4],health_indices[5],
    #              health_indices[6],health_indices[7],health_indices[8],
    #              health_indices[9],health_indices[10],health_indices[11]):
    #     print(i)
    df = pd.DataFrame()
    index = None
    for key in series.keys():
        df = pd.concat([df,pd.DataFrame({key:series[key].health_index})], axis=1)
    # df = pd.concat([df,pd.DataFrame({'aaa':[2,3,4,5]})],axis=1)

    overall_health_index = []
    for i, row in df.iterrows():
        data = Series.to_dict(row)
        s = 0
        weights2 = {}
        for key in data.keys():
            weights2[key] = weights[key]*abs(0.5-data[key])
            s = s + data[key]*weights2[key]
        s = s/sum(weights2.values())
        # s = s/sum(data*)
        overall_health_index.append(s)
    overall_health_index = pd.DataFrame({'overall_health_index':overall_health_index},index = series['PT-204'].time)

    return overall_health_index

if __name__ == '__main__':
    plot = 'pred'
    paths_health_index_all,paths_health_index_pred = get_paths()
    series_all = load_dataset(paths_health_index_all)
    series_pred = load_dataset(paths_health_index_pred)
    if plot == 'pred':
        path = os.path.join(root_path,'health_index_pred')
        overall_health_index = get_combined_health_index(series_pred)
        plot_health_index_combined(series_pred, overall_health_index, path)
        plot_health_index_separated(series_pred, path)
    elif plot=='all':
        path = os.path.join(root_path,'health_index_all')
        overall_health_index = get_combined_health_index(series_all)
        plot_health_index_combined(series_all, overall_health_index, path)
        plot_health_index_separated(series_all, path)
    elif plot == 'overlay':
        path = os.path.join(root_path,'health_index_overlay')
        plot_health_index_overlay(series_all,series_pred, path)