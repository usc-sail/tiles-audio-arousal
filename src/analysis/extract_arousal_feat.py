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

agg_list = [1, 6]

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']



def return_row_stats_df(time, type, demographic_dict, data, all_data=None, id=''):

    row_df = pd.DataFrame(index=[time])
    row_df['time'] = time
    row_df['id'] = id
    row_df['type'] = type
    row_df['shift'] = demographic_dict['shift']
    row_df['gender'] = demographic_dict['gender']
    row_df['supervision'] = demographic_dict['supervision']
    row_df['icu'] = demographic_dict['icu']
    for ground_truth in ["ocb", "stai", "pan_PosAffect", "pan_NegAffect", "itp", "irb"]:
        row_df[ground_truth] = demographic_dict[ground_truth]
    
    if len(data) == 0:
        row_df['score'] = np.nan
    else:
        if type == 'arousal':
            row_df['score'] = np.nanmean(data['fusion'])
        elif type == 'occurance_rate':
            row_df['score'] = np.nanmean(data)
        elif type == 'speech_prob':
            row_df['score'] = np.nanmean(data)
        elif type == 'median':
            row_df['score'] = np.nanmedian(data['fusion'])
        elif type == 'inter_75_ratio':
            row_df['score'] = np.nanmean(np.array(data['fusion']) >= demographic_dict['inter_75']) * 100
        elif type == 'inter_25_ratio':
            row_df['score'] = np.nanmean(np.array(data['fusion']) <= demographic_dict['inter_25']) * 100
        elif type == 'neutral':
            row_df['score'] = np.nanmean((demographic_dict['inter_25'] < np.array(data['fusion'])) & (np.array(data['fusion']) < demographic_dict['inter_75'])) * 100
        elif type == 'pos_threshold':
            row_df['score'] = np.nanmean(np.array(data['fusion']) > 0.3)
        elif type == 'neg_threshold':
            row_df['score'] = np.nanmean(np.array(data['fusion']) < -0.3)
        elif type == 'inter_90':
            row_df['score'] = np.nanpercentile(np.array(data['fusion']), 90)
        elif type == 'inter_75':
            row_df['score'] = np.nanpercentile(np.array(data['fusion']), 70)
        elif type == 'inter_25':
            row_df['score'] = np.nanpercentile(np.array(data['fusion']), 25)
        elif type == 'inter_10':
            row_df['score'] = np.nanpercentile(np.array(data['fusion']), 10)
        elif type == 'inter_10':
            row_df['score'] = np.nanmean(np.array(data['fusion']), 10)
        elif type == 'frequency':
            # filter case that are like errors
            if len(data) > 0:
                data = np.array(data)[np.array(data) < 120]
            row_df['score'] = np.nanmean(data)
        elif type == 'inter_session_time':
            # filter case that are like errors
            # pdb.set_trace()
            # if len(data) > 0: data = np.array(data)[np.array(data) < 120]
            row_df['score'] = np.nanmean(data)
        elif type == 'session_time':
            row_df['score'] = np.nanmean(data)
        elif type == 'session_time_75':
            row_df['score'] = np.nanpercentile(data, 75)
        elif type == 'session_time_above_1min':
            row_df['score'] = np.nanmean(np.array(data) > 1) * 100
            # pdb.set_trace()
        elif type == 'ratio':
            row_df['score'] = len(data) / len(all_data)

    return row_df


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rssi_threshold", default=150, type=int)
    parser.add_argument("--fg_threshold", default=0.7, type=float)
    parser.add_argument("--data_dir", default="/media/data/tiles-opendataset/")
    parser.add_argument("--output_dir", default="/media/data/projects/speech-privacy/tiles/")
    args = parser.parse_args()
    
    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'

    # Download the participant information data
    save_root_path = Path(args.output_dir)
    save_setting_str = 'fg'+str(args.fg_threshold).replace(".", "")+'_rssi'+str(args.rssi_threshold)
    Path.mkdir(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str), parents=True, exist_ok=True)       
    
    # Read all igtb
    igtb_df = load_data_basic.read_participant_info(Path(args.data_dir).joinpath(bucket_str))
    nurse_df = igtb_df.loc[(igtb_df['currentposition'] == 'A') | (igtb_df['currentposition'] == 'B')]
    participant_info_df = pd.read_csv(Path(os.path.realpath(__file__)).parents[0].joinpath('participant_info.csv'))

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()

    sel_nurse_df = pd.DataFrame()
    for nurse_id in nurse_id_list:
        if Path.exists(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str, nurse_id + '.pkl')) == False: continue
        data_dict = pickle.load(open(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str, nurse_id + '.pkl'), 'rb'))
        # Ensure data quality, have at least 5 days of data
        if 'inter_session_time' in list(data_dict[1][0]['all']):
            if len(data_dict[1][0]['all']['inter_session_time']) >= 5:
                sel_nurse_df = pd.concat([sel_nurse_df, nurse_df.loc[nurse_df['participant_id'] == nurse_id]])
    
    # pdb.set_trace()
    save_dict = dict()
    for agg in agg_list:
        save_dict[agg] = dict()
        for loc in ['all', 'ns', 'pat', 'other', 'outside']: save_dict[agg][loc] = pd.DataFrame()

    # read data and rate
    nurse_id_list = list(sel_nurse_df.participant_id)
    nurse_id_list.sort()
    # pdb.set_trace()
    for nurse_id in nurse_id_list:

        print(f'Process {nurse_id}')
        shift = 'Day shift' if nurse_df.loc[nurse_df['participant_id'] == nurse_id]['Shift'].values == 'Day shift' else 'Night shift'
        supervision = 'Manager' if nurse_df.loc[nurse_df['participant_id'] == nurse_id]['supervise'].values == 1 else 'Non-Manager'
        gender = 'Male' if nurse_df.loc[nurse_df['participant_id'] == nurse_id]['gender'].values == 1 else 'Female'

        demographic_dict = dict()
        for ground_truth in ["ocb", "stai", "pan_PosAffect", "pan_NegAffect", "itp", "irb"]:
            demographic_dict[ground_truth] = sel_nurse_df.loc[sel_nurse_df['participant_id'] == nurse_id][ground_truth].values

        primary_unit = participant_info_df.loc[participant_info_df['ParticipantID'] == nurse_id]['PrimaryUnit'].values
        icu_str = 'Non-ICU'
        for unit in icu_list:
            if unit in primary_unit: icu_str = 'ICU'

        demographic_dict['shift'] = shift
        demographic_dict['supervision'] = supervision
        demographic_dict['icu'] = icu_str
        demographic_dict['gender'] = gender

        if Path.exists(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str, nurse_id+'.pkl')) == False: continue
        data_dict = pickle.load(open(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str, nurse_id+'.pkl'), 'rb'))
    
        inter_75 = np.nanpercentile(data_dict[1][0]['all']['data']['fusion'], 75)
        inter_25 = np.nanpercentile(data_dict[1][0]['all']['data']['fusion'], 25)
        demographic_dict['inter_75'] = inter_75
        demographic_dict['inter_25'] = inter_25
        # pdb.set_trace()

        for agg in agg_list:
            agg_dict = data_dict[agg]
            for agg_idx in range(agg):
                for loc in ['all', 'ns', 'pat', 'other', 'outside']:
                    # pdb.set_trace()
                    # if agg_idx == 2: pdb.set_trace()
                    if len(agg_dict[agg_idx][loc]['inter_75']) < 5: continue
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'session_time', demographic_dict, data=agg_dict[agg_idx][loc]['session_time'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'session_time_above_1min', demographic_dict, data=agg_dict[agg_idx][loc]['session_time'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'inter_session_time', demographic_dict, data=agg_dict[agg_idx][loc]['inter_session_time'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'arousal', demographic_dict, data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'pos_threshold', demographic_dict,data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'neg_threshold', demographic_dict, data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'inter_90', demographic_dict, data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'inter_10', demographic_dict, data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'inter_75', demographic_dict, data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'inter_25', demographic_dict, data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'inter_75_ratio', demographic_dict, data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'inter_25_ratio', demographic_dict, data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'occurance_rate', demographic_dict, data=agg_dict[agg_idx][loc]['occurance_rate'], id=nurse_id))
                    save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'speech_prob', demographic_dict, data=agg_dict[agg_idx][loc]['speech_prob'], id=nurse_id))
                    # save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'median', demographic_dict, data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
                    # save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'session_time_75', demographic_dict, data=agg_dict[agg_idx][loc]['session_time'], id=nurse_id))
                    # save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'inter_session_time', demographic_dict, data=agg_dict[agg_idx][loc]['frequency_list'], id=nurse_id))
                    # save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'ratio', demographic_dict, data=agg_dict[agg_idx][loc]['data'], all_data=agg_dict[agg_idx]['all']['data'], id=nurse_id))
                    # save_dict[agg][loc] = save_dict[agg][loc].append(return_row_stats_df(agg_idx, 'neutral', demographic_dict, data=agg_dict[agg_idx][loc]['data'], id=nurse_id))
    pickle.dump(save_dict, open(Path(os.path.realpath(__file__)).parents[0].joinpath(save_setting_str+'.pkl'), "wb"))



