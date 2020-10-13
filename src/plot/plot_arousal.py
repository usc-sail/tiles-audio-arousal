from util.load_data_basic import *
import pickle
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns


threshold_list = [0.4, 0.5, 0.6]
agg_list = [4, 6]


def create_folder(save_path):
    if Path.exists(save_path) is False: Path.mkdir(save_path)


def return_row_df(time, type, shift, supervision, nurse_year, data):
    row_df = pd.DataFrame(index=[time])
    row_df['time'] = time
    row_df['type'] = type
    row_df['shift'] = shift
    row_df['supervision'] = supervision
    row_df['nurse_year'] = nurse_year
    row_df['score'] = np.nanmean(data)

    return row_df


def plot_arousal(data_df, threshold, data_type, save_root_path, loc):
    fig, axes = plt.subplots(figsize=(12.5, 4), nrows=1, ncols=2)

    # option_list = [['Day shift', 'Night shift'], ['Manager', 'Non-Manager']]
    # type_list = ['shift', 'supervision']
    # title_list = ['Day shift/Night shift', 'Manager/Non-Manager']

    option_list = [['Day shift', 'Night shift'], ['<= 10 Years', '> 10 Years']]
    type_list = ['shift', 'nurse_year']
    title_list = ['Day shift/Night shift', 'Work Experience']
    # option_list = [['Day shift', 'Night shift'], ['Higher OCB', 'Lower OCB']]
    # title_list = ['Day shift/Night shift', 'OCB (Higher/Lower)']

    y_range_dict = {'arousal': [-0.25, 0.25], 'pos_threshold': [0, 0.5], 'neg_threshold': [0, 0.8],
                    'ratio_mean': [0.1, 0.4], 'inter_90': [0, 0.75], 'inter_10': [-0.75, 0],
                    'num': [10, 70]}
    y_tick_dict = {'arousal': [-0.3, -0.2, -0.1, 0, 0.1, 0.2],
                   'pos_threshold': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                   'neg_threshold': [0, 0.2, 0.4, 0.6, 0.8],
                   'inter_90': [0, 0.25, 0.5, 0.75],
                   'inter_10': [-0.75, -0.5, -0.25, -0.0],
                   'ratio_mean': [0.1, 0.2, 0.3, 0.4],
                   'num': [10, 30, 50, 70],}
    title_dict = {'arousal': 'Average Arousal',
                  'pos_threshold': 'Positive Arousal Ratio',
                  'neg_threshold': 'Negative Arousal Ratio',
                  'inter_90': 'Arousal (Percentile 90th)',
                  'inter_10': 'Arousal (Percentile 10th)',
                  'ratio_mean': 'Speaking Ratio',
                  'num': 'Number of Recordings'}

    for i in range(2):
        data_df = data_df.dropna()
        data_df = data_df.loc[data_df['type'] == data_type]

        first_df = data_df.loc[data_df[type_list[i]] == option_list[i][0]]
        second_df = data_df.loc[data_df[type_list[i]] == option_list[i][1]]

        sns.lineplot(x="time", y='score', dashes=False, marker="o", hue=type_list[i], data=data_df, palette="seismic", ax=axes[i])

        # set ticks
        x_tick_list = ['0-3', '3-6', '6-9', '9-12'] if len(set(data_df['time'])) == 4 else ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12']

        # Calculate p value
        for time in range(len(set(data_df['time']))):
            tmp_first_df = first_df.loc[first_df['time'] == time]
            tmp_second_df = second_df.loc[second_df['time'] == time]

            stats_value, p = stats.kruskal(np.array(tmp_first_df['score']), np.array(tmp_second_df['score']))
            x_tick_list[time] = x_tick_list[time] + '\n(p<0.01)' if p < 0.01 else x_tick_list[time] + '\n(p=' + str(p)[:4] + ')'
        axes[i].set_xticks(range(len(set(data_df['time']))))
        axes[i].set_xticklabels(x_tick_list, fontdict={'fontweight': 'bold', 'fontsize': 12})
        axes[i].yaxis.set_tick_params(size=1)

        axes[i].set_yticks(y_tick_dict[data_type])
        axes[i].set_yticklabels(y_tick_dict[data_type], fontdict={'fontweight': 'bold', 'fontsize': 12})

        # set limits
        axes[i].set_xlim([-0.5, len(set(data_df['time'])) - 0.5])
        axes[i].set_ylim(y_range_dict[data_type])

        # set grids
        axes[i].grid(linestyle='--')
        axes[i].grid(False, axis='y')

        # set labels
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams['axes.labelweight'] = 'bold'

        axes[i].set_xlabel('Time in a shift (Hour)', fontsize=12.5, fontweight='bold')
        axes[i].set_ylabel('Arousal', fontsize=12.5, fontweight='bold')
        axes[i].set_title(title_list[i], fontdict={'fontweight': 'bold', 'fontsize': 12.5})

        # Set Legend
        handles, labels = axes[i].get_legend_handles_labels()
        axes[i].legend(handles=handles[0:], labels=labels[0:], prop={'size': 12, 'weight': 'bold'}, loc='upper right')
        for tick in axes[i].yaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
            tick.label1.set_fontweight('bold')

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.figtext(0.5, 0.95, title_dict[data_type], ha='center', va='center', fontsize=13.5, fontweight='bold')

    create_folder(save_root_path.joinpath('plot'))
    create_folder(save_root_path.joinpath('plot', str(threshold)))
    create_folder(save_root_path.joinpath('plot', str(threshold), str(agg)))
    create_folder(save_root_path.joinpath('plot', str(threshold), str(agg), loc))

    print('Plot %s, %s, %s, %s' % (str(threshold), str(agg), loc, data_type))
    plt.savefig(save_root_path.joinpath('plot', str(threshold), str(agg), loc, data_type + '.png'), dpi=300)
    plt.close()


if __name__ == '__main__':

    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'

    # Download the participant information data
    save_root_path = Path(__file__).parent.absolute().parents[1].joinpath('data')

    # Read all igtb
    igtb_df = read_AllBasic(save_root_path.joinpath(bucket_str))
    nurse_df = igtb_df.loc[igtb_df['currentposition'] == 'A']
    days_at_work_df = read_days_at_work(save_root_path.joinpath(bucket_str))

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()

    if Path.exists(Path.cwd().joinpath('plot.pkl')) is True:
        plot_dict = pickle.load(open(Path.cwd().joinpath('plot.pkl'), 'rb'))
    else:
        plot_dict = {}
        for threshold in threshold_list:
            plot_dict[threshold] = {}
            for agg in agg_list:
                plot_dict[threshold][agg] = {}
                for loc in ['all', 'ns', 'pat', 'other']:
                    plot_dict[threshold][agg][loc] = pd.DataFrame()

        # read data and rate
        for id in nurse_id_list[:]:
            shift = 'Day shift' if nurse_df.loc[nurse_df['participant_id'] == id]['Shift'].values[0] == 'Day shift' else 'Night shift'
            supervision = 'Manager' if nurse_df.loc[nurse_df['participant_id'] == id]['supervise'].values[0] == 1 else 'Non-Manager'
            nurse_year = '> 10 Years' if nurse_df.loc[nurse_df['participant_id'] == id]['nurseyears'].values[0] > 10 else '<= 10 Years'
            # nurse_year = 'Higher OCB' if nurse_df.loc[nurse_df['participant_id'] == id]['ocb'].values[0] > np.nanmedian(nurse_df['ocb']) else 'Lower OCB'
            # nurse_year = '> 10 Years' if nurse_df.loc[nurse_df['participant_id'] == id]['lang'].values[0] == 1 else '<= 10 Years'
            if Path.exists(save_root_path.joinpath('analysis', 'arousal', 'time_in_shift', id + '.pkl')) is False:
                continue

            data_dict = pickle.load(open(save_root_path.joinpath('analysis', 'arousal', 'time_in_shift', id + '.pkl'), 'rb'))
            for threshold in threshold_list:
                for agg in agg_list:
                    tmp_dict = data_dict[threshold][agg]
                    for i in range(agg):
                        for loc in ['all', 'ns', 'pat', 'other']:
                            plot_dict[threshold][agg][loc] = plot_dict[threshold][agg][loc].append(return_row_df(i, 'arousal', shift, supervision, nurse_year, data=tmp_dict[i][loc]['mean']))
                            plot_dict[threshold][agg][loc] = plot_dict[threshold][agg][loc].append(return_row_df(i, 'pos', shift, supervision, nurse_year, data=tmp_dict[i][loc]['pos']))
                            plot_dict[threshold][agg][loc] = plot_dict[threshold][agg][loc].append(return_row_df(i, 'pos_threshold', shift, supervision, nurse_year, data=tmp_dict[i][loc]['pos_threshold']))
                            plot_dict[threshold][agg][loc] = plot_dict[threshold][agg][loc].append(return_row_df(i, 'neg', shift, supervision, nurse_year, data=tmp_dict[i][loc]['neg']))
                            plot_dict[threshold][agg][loc] = plot_dict[threshold][agg][loc].append(return_row_df(i, 'neg_threshold', shift, supervision, nurse_year, data=tmp_dict[i][loc]['neg_threshold']))
                            plot_dict[threshold][agg][loc] = plot_dict[threshold][agg][loc].append(return_row_df(i, 'ratio_mean', shift, supervision, nurse_year, data=tmp_dict[i][loc]['ratio_mean']))
                            plot_dict[threshold][agg][loc] = plot_dict[threshold][agg][loc].append(return_row_df(i, 'ratio_median', shift, supervision, nurse_year, data=tmp_dict[i][loc]['ratio_median']))
                            plot_dict[threshold][agg][loc] = plot_dict[threshold][agg][loc].append(return_row_df(i, 'inter_90', shift, supervision, nurse_year, data=tmp_dict[i][loc]['inter_90']))
                            plot_dict[threshold][agg][loc] = plot_dict[threshold][agg][loc].append(return_row_df(i, 'inter_10', shift, supervision, nurse_year, data=tmp_dict[i][loc]['inter_10']))
                            plot_dict[threshold][agg][loc] = plot_dict[threshold][agg][loc].append(return_row_df(i, 'num', shift, supervision, nurse_year, data=tmp_dict[i][loc]['num']))

        pickle.dump(plot_dict, open(Path.cwd().joinpath('plot.pkl'), "wb"))

    for threshold in threshold_list:
        for agg in agg_list:
            for loc in ['all', 'ns', 'pat', 'other']:
                plot_arousal(plot_dict[threshold][agg][loc], threshold, data_type='num', save_root_path=save_root_path, loc=loc)
                plot_arousal(plot_dict[threshold][agg][loc], threshold, data_type='arousal', save_root_path=save_root_path, loc=loc)
                plot_arousal(plot_dict[threshold][agg][loc], threshold, data_type='pos_threshold', save_root_path=save_root_path, loc=loc)
                plot_arousal(plot_dict[threshold][agg][loc], threshold, data_type='neg_threshold', save_root_path=save_root_path, loc=loc)
                plot_arousal(plot_dict[threshold][agg][loc], threshold, data_type='ratio_mean', save_root_path=save_root_path, loc=loc)
                plot_arousal(plot_dict[threshold][agg][loc], threshold, data_type='inter_90', save_root_path=save_root_path, loc=loc)
                plot_arousal(plot_dict[threshold][agg][loc], threshold, data_type='inter_10', save_root_path=save_root_path, loc=loc)



