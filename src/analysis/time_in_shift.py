from util.load_data_basic import *
import pickle
from datetime import timedelta


arousal_feat_list = ['F0_sma', 'pcm_intensity_sma', 'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma']
threshold_list = [0.4, 0.5, 0.6]

agg_list = [4, 6]


def create_folder(save_path):
    if Path.exists(save_path) is False: Path.mkdir(save_path)


if __name__ == '__main__':

    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'

    # Download the participant information data
    save_root_path = Path(__file__).parent.absolute().parents[1].joinpath('data')

    # Read all igtb
    igtb_df = read_AllBasic(save_root_path.joinpath(bucket_str))
    nurse_df = igtb_df.loc[igtb_df['currentposition'] == 'A']

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()

    # read data and rate
    for id in nurse_id_list[:]:
        shift = 'day' if nurse_df.loc[nurse_df['participant_id'] == id]['Shift'].values[0] == 'Day shift' else 'night'

        if Path.exists(save_root_path.joinpath('process', 'arousal', 'rating', str(threshold_list[0]), id + '.csv.gz')) is True:
            data_dict = {}
            for threshold in threshold_list:
                print('process %s, threshold %s' % (id, str(threshold)))

                rating_df = pd.read_csv(save_root_path.joinpath('process', 'arousal', 'rating', str(threshold), id + '.csv.gz'), index_col=0)
                owl_in_one_df = pd.read_csv(save_root_path.joinpath('process', 'owl-in-one', id + '.csv.gz'), index_col=0)
                rating_df = rating_df.sort_index()

                # number of days available
                num_of_days = (pd.to_datetime(rating_df.index[-1]) - pd.to_datetime(rating_df.index[0])).days + 1
                if shift == 'day':
                    data_start_time_str = (pd.to_datetime(rating_df.index[0])).replace(hour=7, minute=0, second=0)
                else:
                    data_start_time_str = (pd.to_datetime(rating_df.index[0]) - timedelta(days=1)).replace(hour=19, minute=0, second=0)
                data_start_time_str = data_start_time_str.strftime(date_time_format)[:-3]

                work_timeline_df = pd.read_csv(save_root_path.joinpath('process', 'work_timeline', id + '.csv.gz'), index_col=0)
                work_timeline_df = work_timeline_df.sort_index()

                data_dict[threshold] = {}
                for agg in agg_list:
                    data_dict[threshold][agg] = {}
                    for j in range(agg):
                        data_dict[threshold][agg][j] = {}
                        for loc in ['all', 'ns', 'pat', 'other']:
                            data_dict[threshold][agg][j][loc] = {}
                            data_dict[threshold][agg][j][loc]['data'] = pd.DataFrame()
                            data_dict[threshold][agg][j][loc]['num'] = []
                            data_dict[threshold][agg][j][loc]['inter_90'] = []
                            data_dict[threshold][agg][j][loc]['inter_75'] = []
                            data_dict[threshold][agg][j][loc]['inter_25'] = []
                            data_dict[threshold][agg][j][loc]['inter_10'] = []
                            data_dict[threshold][agg][j][loc]['median'] = []
                            data_dict[threshold][agg][j][loc]['mean'] = []
                            data_dict[threshold][agg][j][loc]['max'] = []
                            data_dict[threshold][agg][j][loc]['min'] = []
                            data_dict[threshold][agg][j][loc]['pos'] = []
                            data_dict[threshold][agg][j][loc]['neg'] = []
                            data_dict[threshold][agg][j][loc]['pos_threshold'] = []
                            data_dict[threshold][agg][j][loc]['neg_threshold'] = []
                            data_dict[threshold][agg][j][loc]['ratio_mean'] = []
                            data_dict[threshold][agg][j][loc]['ratio_median'] = []

                # for i in range(len(work_timeline_df)):
                for i in range(num_of_days):
                    # start_time_str = work_timeline_df['start'][i]
                    # end_time_str = work_timeline_df['end'][i]
                    start_time_str = (pd.to_datetime(data_start_time_str) + timedelta(days=i)).strftime(date_time_format)[:-3]
                    end_time_str = (pd.to_datetime(start_time_str) + timedelta(hours=12)).strftime(date_time_format)[:-3]

                    for agg in agg_list:
                        agg_window = int(12 / agg)

                        for j in range(agg):
                            start_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=j*agg_window)).strftime(date_time_format)[:-3]
                            end_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=j*agg_window+agg_window, minutes=-1)).strftime(date_time_format)[:-3]

                            seg_df = rating_df[start_agg_str:end_agg_str]

                            # too few samples
                            if len(seg_df) < 10:
                                continue

                            seg_owl_in_one_df = owl_in_one_df[start_agg_str:end_agg_str]
                            seg_df.loc[:, 'room'] = 'other'
                            for time_str in list(seg_df.index):
                                # lets see if last minute or next minute have the loc data, since ble is not really reliable
                                last_minute_str = ((pd.to_datetime(time_str) - timedelta(minutes=1))).strftime(date_time_format)[:-3]
                                next_minute_str = ((pd.to_datetime(time_str) + timedelta(minutes=1))).strftime(date_time_format)[:-3]

                                if time_str in list(seg_owl_in_one_df.index):
                                    seg_df.loc[time_str, 'room'] = seg_owl_in_one_df.loc[time_str, 'room']
                                elif last_minute_str in list(seg_owl_in_one_df.index):
                                    seg_df.loc[time_str, 'room'] = seg_owl_in_one_df.loc[last_minute_str, 'room']
                                elif next_minute_str in list(seg_owl_in_one_df.index):
                                    seg_df.loc[time_str, 'room'] = seg_owl_in_one_df.loc[next_minute_str, 'room']

                            for loc in ['all', 'ns', 'pat', 'other']:
                                if loc == 'all':
                                    analysis_df = seg_df
                                else:
                                    analysis_df = seg_df.loc[seg_df['room'] == loc]

                                if len(analysis_df) < 5:
                                    continue

                                data_dict[threshold][agg][j][loc]['data'] = data_dict[threshold][agg][j][loc]['data'].append(analysis_df)
                                data_dict[threshold][agg][j][loc]['inter_90'].append(np.nanpercentile(analysis_df['fusion'], 90))
                                data_dict[threshold][agg][j][loc]['inter_75'].append(np.nanpercentile(analysis_df['fusion'], 75))
                                data_dict[threshold][agg][j][loc]['inter_25'].append(np.nanpercentile(analysis_df['fusion'], 25))
                                data_dict[threshold][agg][j][loc]['inter_10'].append(np.nanpercentile(analysis_df['fusion'], 10))
                                data_dict[threshold][agg][j][loc]['median'].append(np.nanmedian(analysis_df['fusion']))
                                data_dict[threshold][agg][j][loc]['mean'].append(np.nanmean(analysis_df['fusion']))
                                data_dict[threshold][agg][j][loc]['max'].append(np.nanmax(analysis_df['fusion']))
                                data_dict[threshold][agg][j][loc]['min'].append(np.nanmin(analysis_df['fusion']))
                                data_dict[threshold][agg][j][loc]['pos'].append(np.nanmean((analysis_df['fusion']) >= 0))
                                data_dict[threshold][agg][j][loc]['neg'].append(np.nanmean((analysis_df['fusion']) < 0))
                                data_dict[threshold][agg][j][loc]['pos_threshold'].append(np.nanmean((analysis_df['fusion']) >= 0.25))
                                data_dict[threshold][agg][j][loc]['neg_threshold'].append(np.nanmean((analysis_df['fusion']) <= -0.25))
                                data_dict[threshold][agg][j][loc]['ratio_mean'].append(np.nanmean((analysis_df['ratio'])))
                                data_dict[threshold][agg][j][loc]['ratio_median'].append(np.nanmedian((analysis_df['ratio'])))

                                data_dict[threshold][agg][j][loc]['num'].append(len(analysis_df))

            create_folder(save_root_path.joinpath('analysis'))
            create_folder(save_root_path.joinpath('analysis', 'arousal'))
            create_folder(save_root_path.joinpath('analysis', 'arousal', 'time_in_shift'))

            pickle.dump(data_dict, open(save_root_path.joinpath('analysis', 'arousal', 'time_in_shift', id + '.pkl'), "wb"))

