import pdb, sys, pytz, os, argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from scipy import stats

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'util'))
import load_data_basic


threshold_list = [0.4, 0.5, 0.6]
agg_list = [6]

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']

y_range_dict = {'arousal': [-0.25, 0.25], 
                'pos_threshold': [0, 0.5], 'neg_threshold': [0, 0.8],
                'ratio_mean': [0.1, 0.4],
                'inter_75_ratio': [0, 50],
                'inter_25_ratio': [0, 50],
                'inter_90': [0, 0.75], 'inter_10': [-0.75, 0],
                'num': [0, 30],
                'inter_session_time': [0, 20],
                'inter_session_time_ns': [0, 40],
                'inter_session_time_night': [0, 20],
                'session_time_above_1min': [10, 60],
                'session_time_above_1min_day': [10, 70],
                'session_time_above_1min_night': [0, 60],
                'session_time_day': [1, 2.5],
                'session_time_night': [1, 2.5]}


y_tick_dict = {'arousal': [-0.3, -0.2, -0.1, 0, 0.1, 0.2],
                'pos_threshold': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                'neg_threshold': [0, 0.2, 0.4, 0.6, 0.8],
                'inter_75_ratio': [0, 10, 20, 30, 40, 50],
                'inter_25_ratio': [0, 10, 20, 30, 40, 50],
                'ratio_mean': [0.1, 0.2, 0.3, 0.4],
                'num': [0, 10, 20, 30],
                'inter_session_time': [0, 5, 10, 15, 20],
                'session_time_day': [1, 1.5, 2, 2.5],
                'session_time_night': [1, 1.5, 2, 2.5],
                'inter_session_time_day': [0, 5, 10, 15],
                'inter_session_time_night': [0, 5, 10, 15, 20],
                'session_time_above_1min': [10, 20, 30, 40, 50, 60],
                'session_time_above_1min_day': [10, 30, 50, 70],
                'session_time_above_1min_night': [0, 20, 40, 60],
                'inter_session_time_ns': [0, 10, 20, 30, 40]}


data_title_dict = {'arousal': 'Average Arousal',
                    'pos_threshold': 'Positive Arousal Ratio',
                    'neg_threshold': 'Negative Arousal Ratio',
                    'inter_90': 'Arousal (Percentile 90th)',
                    'inter_10': 'Arousal (Percentile 10th)',
                    'inter_75_ratio': 'Arousal (Percentile 90th)',
                    'inter_25_ratio': 'Arousal (Percentile 10th)',
                    'ratio_mean': 'Speaking Ratio',
                    'num': 'Number of Recordings',
                    'session_time': 'Session Time',
                    'inter_session_time': 'Inter-session Time',
                    'session_time_above_1min': '>1min Session Ratio'}

    
def plot_arousal(data_df, threshold, data_type, save_root_path, loc, demo_type='shift'):
    
    fig, axes = plt.subplots(figsize=(6.5, 4), nrows=1, ncols=1)
    option_dict = {'shift': ['Day shift', 'Night shift'], 'icu': ['ICU', 'Non-ICU'], 'gender': ['Male', 'Female']}
    title_dict = {'shift': 'Day shift/Night shift', 'icu': 'ICU/Non-ICU'}

    data_df = data_df.dropna()
    data_df = data_df.loc[data_df['type'] == data_type]
    data_df = data_df.reset_index()

    first_df = data_df.loc[data_df[demo_type] == option_dict[demo_type][0]]
    second_df = data_df.loc[data_df[demo_type] == option_dict[demo_type][1]]
    
    # pdb.set_trace()
    sns.lineplot(x="time", y='score', dashes=False, marker="o", hue=demo_type, data=data_df, palette="seismic", ax=axes)

    # set ticks
    x_tick_list = ['1-2', '3-4', '5-6', '7-8', '9-10', '11-12']

    # Calculate p value
    for time in range(len(set(data_df['time']))):
        # pdb.set_trace()
        tmp_first_df = first_df.loc[first_df['time'] == time]
        tmp_second_df = second_df.loc[second_df['time'] == time]

        stats_value, p = stats.mannwhitneyu(np.array(tmp_first_df['score']), np.array(tmp_second_df['score']))
        x_tick_list[time] = x_tick_list[time] + '\n(p<0.01)' if p < 0.01 else x_tick_list[time] + '\n(p=' + str(p)[:4] + ')'
    axes.set_xticks(range(len(set(data_df['time']))))
    axes.set_xticklabels(x_tick_list, fontdict={'fontweight': 'bold', 'fontsize': 12})
    axes.yaxis.set_tick_params(size=1)

    axes.set_yticks(y_tick_dict[data_type])
    axes.set_yticklabels(y_tick_dict[data_type], fontdict={'fontweight': 'bold', 'fontsize': 12})

    # set limits
    axes.set_xlim([-0.5, len(set(data_df['time'])) - 0.5])
    axes.set_ylim(y_range_dict[data_type])

    # set grids
    axes.grid(linestyle='--')
    axes.grid(False, axis='y')

    # set labels
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams['axes.labelweight'] = 'bold'

    axes.set_xlabel('\nTime in a shift (Hour)', fontsize=13.5, fontweight='bold')
    if data_type == 'inter_session_time': axes.set_ylabel('Minutes', fontsize=12.5, fontweight='bold')
    else: axes.set_ylabel('Ratio (%)', fontsize=12.5, fontweight='bold')
    axes.set_title(data_title_dict[data_type]+'\n', fontdict={'fontweight': 'bold', 'fontsize': 13.5})

    # Set Legend
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles=handles[0:], labels=labels[0:], prop={'size': 12, 'weight': 'bold'}, loc='upper right')
    for tick in axes.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
        tick.label1.set_fontweight('bold')

    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    # plt.figtext(0.5, 0.95, title_dict[data_type], ha='center', va='center', fontsize=13.5, fontweight='bold')
    Path.mkdir(save_root_path.joinpath(data_type), parents=True, exist_ok=True)
    plt.savefig(save_root_path.joinpath(data_type, loc+'.png'), dpi=300)
    plt.close()


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rssi_threshold", default=150, type=int)
    parser.add_argument("--fg_threshold", default=0.5, type=float)
    parser.add_argument("--data_dir", default="/media/data/tiles-opendataset/")
    parser.add_argument("--output_dir", default="/media/data/projects/speech-privacy/tiles/")
    args = parser.parse_args()
    
    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'

    # Download the participant information data
    save_setting_str = 'fg'+str(args.fg_threshold).replace(".", "")+'_rssi'+str(args.rssi_threshold)
    save_root_path = Path(os.path.realpath(__file__)).parents[1].joinpath('plot', save_setting_str)
    Path.mkdir(save_root_path, parents=True, exist_ok=True)
    
    # Read all igtb
    igtb_df = load_data_basic.read_participant_info(Path(args.data_dir).joinpath(bucket_str))
    nurse_df = igtb_df.loc[(igtb_df['currentposition'] == 'A') | (igtb_df['currentposition'] == 'B')]
    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()
    
    # read data dict
    plot_dict = pickle.load(open(Path(os.path.realpath(__file__)).parents[0].joinpath(save_setting_str+'.pkl'), 'rb'))
    
    for agg in agg_list:
        for loc in ['all', 'ns', 'pat', 'outside']:
            # plot_arousal(plot_dict[agg][loc], args.fg_threshold, data_type='pos_threshold', save_root_path=save_root_path, loc=loc)
            day_df = plot_dict[agg][loc].loc[plot_dict[agg][loc]['shift'] == 'Day shift']
            night_df = plot_dict[agg][loc].loc[plot_dict[agg][loc]['shift'] == 'Night shift']
            icu_df = plot_dict[agg][loc].loc[plot_dict[agg][loc]['icu'] == 'Day shift']
            night_df = plot_dict[agg][loc].loc[plot_dict[agg][loc]['shift'] == 'Night shift']
            # plot_arousal(day_df, args.fg_threshold, data_type='inter_session_time', save_root_path=save_root_path, loc=loc, demo_type='gender')
            # plot_arousal(day_df, args.fg_threshold, data_type='session_time_above_1min', save_root_path=save_root_path, loc=loc, demo_type='gender')
            # plot_arousal(day_df, args.fg_threshold, data_type='neg_threshold', save_root_path=save_root_path, loc=loc, demo_type='gender')
            # plot_arousal(plot_dict[agg][loc], args.fg_threshold, data_type='inter_session_time', save_root_path=save_root_path, loc=loc, demo_type='shift')
            # plot_arousal(plot_dict[agg][loc], args.fg_threshold, data_type='session_time_above_1min', save_root_path=save_root_path, loc=loc)
            # plot_arousal(plot_dict[agg][loc], args.fg_threshold, data_type='pos_threshold', save_root_path=save_root_path, loc=loc, demo_type='shift')
            # plot_arousal(plot_dict[agg][loc], args.fg_threshold, data_type='inter_75_ratio', save_root_path=save_root_path, loc=loc, demo_type='shift')
            # plot_arousal(plot_dict[agg][loc], threshold, data_type='neg_threshold', save_root_path=save_root_path, loc=loc)
            # plot_arousal(plot_dict[agg][loc], threshold, data_type='num', save_root_path=save_root_path, loc=loc)
            
            plot_arousal(night_df, args.fg_threshold, data_type='inter_session_time', save_root_path=save_root_path, loc=loc, demo_type='gender')




