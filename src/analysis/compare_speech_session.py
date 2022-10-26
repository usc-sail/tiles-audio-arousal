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


agg_list = [1]

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


def print_latex_stats(data_df, demo, loc, data_type):
    loc_dict = {'all': 'All Location', 'ns': 'Nursing Station', 'pat': 'Patient Room', 'other': 'Other Location'}
    option_dict = {'shift': ['Day shift', 'Night shift'],
                   'icu': ['ICU', 'Non-ICU']}

    if data_type == 'frequency' or 'session' in data_type: mulp, unit = 1, ''
    else: mulp, unit = 100, '\\%'
        
    # data_df = data_df.dropna()
    compare_df = data_df.loc[data_df['type'] == data_type]

    first_df = compare_df.loc[compare_df[demo] == option_dict[demo][0]]
    second_df = compare_df.loc[compare_df[demo] == option_dict[demo][1]]

    # pdb.set_trace()

    # Calculate p value, this is shift level, only consider time at 0
    tmp_first_df = first_df.loc[first_df['time'] == 0]
    tmp_second_df = second_df.loc[second_df['time'] == 0]
    stats_value, p = stats.mannwhitneyu(np.array(tmp_first_df['score']), np.array(tmp_second_df['score']))

    # pdb.set_trace()
    print('\multicolumn{1}{l}{\\hspace{0.25cm}{%s}} &' % loc_dict[loc])
    print('\multicolumn{1}{c}{$%.2f%s$ ($%.2f%s$)} &' % (np.nanmedian(tmp_first_df['score']) * mulp, unit, np.nanmean(tmp_first_df['score']) * mulp, unit))
    print('\multicolumn{1}{c}{$%.2f%s$ ($%.2f%s$)} &' % (np.nanmedian(tmp_second_df['score']) * mulp, unit, np.nanmean(tmp_second_df['score']) * mulp, unit))

    if p < 0.001: print('\multicolumn{1}{c}{$\mathbf{<0.001^{**}}$} \\rule{0pt}{3ex} \\\\' % (p))
    elif p < 0.05: print('\multicolumn{1}{c}{$\mathbf{%.3f^*}$} \\rule{0pt}{3ex} \\\\' % (p))
    else: print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))


def compare_stats(data_df, demo, loc, data_type):
    option_dict = {'shift': ['Day shift', 'Night shift'],
                   'gender': ['Male', 'Female'],
                   'nurse_year': ['<= 10 Years', '> 10 Years'],
                   'icu': ['ICU', 'Non-ICU'],
                   'ocb': ['Higher OCB', 'Lower OCB'],
                   'stai': ['Higher Anxiety', 'Lower Anxiety'],
                   'itp': ['Higher ITP', 'Lower ITP'],
                   'irb': ['Higher IRB', 'Lower IRB'],
                   'pos': ['Higher Pos. Affect', 'Lower Pos. Affect'],
                   'neg': ['Higher Neg. Affect', 'Lower Neg. Affect']}

    # data_type_list = ['frequency', 'pos_', 'neg_', 'arousal']
    data_df = data_df.dropna()
    # for i in range(len(data_type_list)):
    pdb.set_trace()

    compare_df = data_df.loc[data_df['type'] == data_type]

    first_df = compare_df.loc[compare_df[demo] == option_dict[demo][0]]
    second_df = compare_df.loc[compare_df[demo] == option_dict[demo][1]]

    # Calculate p value
    for time in range(len(set(compare_df['time']))):
        tmp_first_df = first_df.loc[first_df['time'] == time]
        tmp_second_df = second_df.loc[second_df['time'] == time]
        stats_value, p = stats.mannwhitneyu(np.array(tmp_first_df['score']), np.array(tmp_second_df['score']))

        print('Location = %s' % loc)
        print('Number of valid participant: %s: %i; %s: %i' % (option_dict[demo][0], len(tmp_first_df['score'].dropna()), option_dict[demo][1], len(tmp_second_df['score'].dropna())))
        print('%s: median = %.2f, mean = %.2f' % (option_dict[demo][0], np.nanmedian(tmp_first_df['score']), np.nanmean(tmp_first_df['score'])))
        print('%s: median = %.2f, mean = %.2f' % (option_dict[demo][1], np.nanmedian(tmp_second_df['score']), np.nanmean(tmp_second_df['score'])))
        print('mannwhitneyu test for %s' % data_type)
        print('Statistics = %.3f, p = %.3f\n\n' % (stats_value, p))


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_dir", default="/media/data/tiles-opendataset/")
    parser.add_argument("--output_dir", default="/media/data/projects/speech-privacy/tiles/")
    args = parser.parse_args()
    
    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'

    # Download the participant information data
    save_root_path = Path(args.output_dir)
    Path.mkdir(save_root_path.joinpath('process', 'arousal', 'time_in_shift'), parents=True, exist_ok=True)       
    
    # Read all igtb
    igtb_df = load_data_basic.read_participant_info(Path(args.data_dir).joinpath(bucket_str))
    nurse_df = igtb_df.loc[igtb_df['currentposition'] == 'A']

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()
    # read data dict
    data_dict = pickle.load(open(Path.cwd().joinpath('data.pkl'), 'rb'))
    
    # iterate over aggregation list
    for agg in agg_list:
        for loc in ['all', 'ns', 'pat', 'other']:
            for data_type in ['inter_session_time', 'session_time_above_1min']:
                # demographic variable
                demo = 'shift'
                if 'ratio' in data_type and loc == 'all': continue

                # pdb.set_trace()
                print_latex_stats(data_dict[agg][loc], demo=demo, loc=loc, data_type=data_type)
                pdb.set_trace()
                # day_df = data_dict[agg][loc].loc[data_dict[agg][loc]['shift'] == 'Day shift']
                # night_df = data_dict[agg][loc].loc[data_dict[agg][loc]['shift'] == 'Night shift']
                