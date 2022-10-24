import warnings
warnings.simplefilter(action='ignore')

import pdb, sys, pytz, os, argparse
import pickle
import numpy as np
import pandas as pd
import more_itertools as mit
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'util'))
import load_data_basic

feat_list = ['inter_90', 'inter_75', 'inter_25', 'inter_10',
             'median', 'mean', 'max', 'min', 'pos', 'neg',
             'pos_threshold', 'neg_threshold', 'ratio_mean', 'ratio_median',
             'frequency', 'frequency_list', 'session_time', 'inter_session_time']

agg_list = [1, 6]



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

    # read data and rate
    for nurse_id in nurse_id_list:
        shift = 'day' if nurse_df.loc[nurse_df['participant_id'] == nurse_id]['Shift'].values[0] == 'Day shift' else 'night'
        if Path.exists(save_root_path.joinpath('process', 'arousal', 'rating', nurse_id + '.csv')) == False: continue
        if Path.exists(save_root_path.joinpath('process', 'owl-in-one', nurse_id + '.csv')) == False: continue

        data_dict = dict()
        print(f'process {nurse_id}')
        
        owl_in_one_df = pd.read_csv(save_root_path.joinpath('process', 'owl-in-one', nurse_id + '.csv'), index_col=0)
        rating_df = pd.read_csv(save_root_path.joinpath('process', 'arousal', 'rating', nurse_id + '.csv'), index_col=0)
        
        owl_in_one_df = owl_in_one_df.sort_index()
        rating_df = rating_df.sort_index()

        # number of days available
        num_of_days = (pd.to_datetime(rating_df.index[-1]) - pd.to_datetime(rating_df.index[0])).days + 1
        if shift == 'day': data_start_time_str = (pd.to_datetime(rating_df.index[0])).replace(hour=7, minute=0, second=0)
        else: data_start_time_str = (pd.to_datetime(rating_df.index[0]) - timedelta(days=1)).replace(hour=19, minute=0, second=0)
        data_start_time_str = data_start_time_str.strftime(load_data_basic.date_time_format)[:-3]

        for agg in agg_list:
            data_dict[agg] = dict()
            for agg_idx in range(agg):
                data_dict[agg][agg_idx] = dict()
                for loc in ['all', 'ns', 'pat', 'other']:
                    data_dict[agg][agg_idx][loc] = dict()
                    data_dict[agg][agg_idx][loc]['data'] = pd.DataFrame()
                    data_dict[agg][agg_idx][loc]['num'] = list()
                    for feat in feat_list: data_dict[agg][agg_idx][loc][feat] = list()

        # iterate over days
        for day_idx in range(num_of_days):
            # start_time_str = work_timeline_df['start'][i]
            # end_time_str = work_timeline_df['end'][i]
            start_time_str = (pd.to_datetime(data_start_time_str) + timedelta(days=day_idx)).strftime(load_data_basic.date_time_format)[:-3]
            end_time_str = (pd.to_datetime(start_time_str) + timedelta(hours=12)).strftime(load_data_basic.date_time_format)[:-3]
            
            # iterate over aggregations
            for agg in agg_list:
                agg_window = int(12 / agg)
                for agg_idx in range(agg):
                    start_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=agg_idx*agg_window)).strftime(load_data_basic.date_time_format)[:-3]
                    end_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=agg_idx*agg_window+agg_window, minutes=-1)).strftime(load_data_basic.date_time_format)[:-3]
                    seg_df = rating_df.loc[start_agg_str:end_agg_str]
                    
                    # too few samples
                    if len(seg_df) < 10: continue
                    
                    seg_owl_in_one_df = owl_in_one_df.loc[start_agg_str:end_agg_str]
                    seg_df.loc[seg_df.index, 'room'] = "other"
                    for time_str in list(seg_df.index):
                        # lets see if last minute or next minute have the loc data, since ble is not really reliable
                        last_minute_str = ((pd.to_datetime(time_str) - timedelta(minutes=1))).strftime(load_data_basic.date_time_format)[:-3]
                        next_minute_str = ((pd.to_datetime(time_str) + timedelta(minutes=1))).strftime(load_data_basic.date_time_format)[:-3]
                        if time_str in list(seg_owl_in_one_df.index):
                            seg_df.loc[time_str, 'room'] = seg_owl_in_one_df.loc[time_str, 'room']
                        elif last_minute_str in list(seg_owl_in_one_df.index):
                            seg_df.loc[time_str, 'room'] = seg_owl_in_one_df.loc[last_minute_str, 'room']
                        elif next_minute_str in list(seg_owl_in_one_df.index):
                            seg_df.loc[time_str, 'room'] = seg_owl_in_one_df.loc[next_minute_str, 'room']
                    
                    for loc in ['all', 'ns', 'pat', 'other']:
                        if loc == 'all': analysis_df = seg_df
                        else: analysis_df = seg_df.loc[seg_df['room'] == loc]
                        if len(analysis_df) < 5 and agg == 6: continue
                        if len(analysis_df) < 30 and agg == 1: continue

                        # inter session feature
                        interval = ((pd.to_datetime(analysis_df.index[1:]) - pd.to_datetime(analysis_df.index[:-1])).seconds / 60)
                        interval_list = np.array(interval)[np.array(interval) < 120]
                        interval_mean = np.nanmean(interval_list)
                        # find continuous speech activity, first subtract the start
                        offset_time = list((pd.to_datetime(analysis_df.index) - pd.to_datetime(analysis_df.index[0])).seconds / 60)
                        session_list = [list(group) for group in mit.consecutive_groups(offset_time)]

                        if len(session_list) != 0:
                            for session_idx, session_data in enumerate(session_list):
                                data_dict[agg][agg_idx][loc]['session_time'].append(len(session_data))
                                if session_idx + 1 != len(session_list):
                                    next_start, curr_end = session_list[session_idx+1][0], session_data[-1]
                                    data_dict[agg][agg_idx][loc]['inter_session_time'].append(next_start-curr_end-1)
                        for data in interval_list:
                            data_dict[agg][agg_idx][loc]['frequency_list'].append(data)

                        data_dict[agg][agg_idx][loc]['data'] = data_dict[agg][agg_idx][loc]['data'].append(analysis_df)
                        data_dict[agg][agg_idx][loc]['frequency'].append(interval_mean)
                        data_dict[agg][agg_idx][loc]['inter_90'].append(np.nanpercentile(analysis_df['fusion'], 90))
                        data_dict[agg][agg_idx][loc]['inter_75'].append(np.nanpercentile(analysis_df['fusion'], 75))
                        data_dict[agg][agg_idx][loc]['inter_25'].append(np.nanpercentile(analysis_df['fusion'], 25))
                        data_dict[agg][agg_idx][loc]['inter_10'].append(np.nanpercentile(analysis_df['fusion'], 10))
                        data_dict[agg][agg_idx][loc]['median'].append(np.nanmedian(analysis_df['fusion']))
                        data_dict[agg][agg_idx][loc]['mean'].append(np.nanmean(analysis_df['fusion']))
                        data_dict[agg][agg_idx][loc]['max'].append(np.nanmax(analysis_df['fusion']))
                        data_dict[agg][agg_idx][loc]['min'].append(np.nanmin(analysis_df['fusion']))
                        data_dict[agg][agg_idx][loc]['pos'].append(np.nanmean((analysis_df['fusion']) >= 0))
                        data_dict[agg][agg_idx][loc]['neg'].append(np.nanmean((analysis_df['fusion']) < 0))
                        data_dict[agg][agg_idx][loc]['pos_threshold'].append(np.nanmean((analysis_df['fusion']) >= 0.25))
                        data_dict[agg][agg_idx][loc]['neg_threshold'].append(np.nanmean((analysis_df['fusion']) <= -0.25))
                        data_dict[agg][agg_idx][loc]['ratio_mean'].append(np.nanmean((analysis_df['ratio'])))
                        data_dict[agg][agg_idx][loc]['ratio_median'].append(np.nanmedian((analysis_df['ratio'])))
                        data_dict[agg][agg_idx][loc]['num'].append(len(analysis_df))

        pickle.dump(data_dict, open(save_root_path.joinpath('process', 'arousal', 'time_in_shift', nurse_id + '.pkl'), "wb"))

