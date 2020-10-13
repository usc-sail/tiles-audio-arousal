from util.load_data_basic import *
import pickle
from numpy import *


arousal_feat_list = ['F0_sma', 'pcm_intensity_sma', 'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma']
threshold_list = [0.4, 0.5, 0.6]


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
    for id in nurse_id_list[50:]:
        for threshold in threshold_list:
            if Path.exists(save_root_path.joinpath('process', 'arousal', 'baseline', str(threshold), id + '.csv.gz')) is False:
                continue

            baseline_df = pd.read_csv(save_root_path.joinpath('process', 'arousal', 'baseline', str(threshold), id + '.csv.gz'), index_col=0)
            data_dict = pickle.load(open(save_root_path.joinpath('process', 'fg-audio', str(threshold), id + '.pkl'), 'rb'))

            # if we have less than 10 days of data, skip
            if len(data_dict.keys()) < 10:
                continue

            save_df = pd.DataFrame()

            for date_str in list(data_dict.keys()):
                for time_str in list(data_dict[date_str].keys()):
                    save_time_str = pd.to_datetime(time_str).replace(second=0, microsecond=0).strftime(date_time_format)[:-3]
                    print('Process data for %s, %s' % (id, save_time_str))

                    tmp_df = data_dict[date_str][time_str][arousal_feat_list]
                    tmp_df = tmp_df.loc[(40 < tmp_df['F0_sma']) & (tmp_df['F0_sma'] < 500)]

                    # no valid data
                    if len(tmp_df) < 100:
                        continue

                    median_pitch = np.log10(np.nanmedian(tmp_df['F0_sma']))
                    median_intensity = np.nanmedian(tmp_df['pcm_intensity_sma'])
                    median_hf_lf_ratio = np.nanmedian(np.array(tmp_df['pcm_fftMag_fband1000-4000_sma']) / np.array(tmp_df['pcm_fftMag_fband250-650_sma']))

                    pitch_arousal = np.nanmean(median_pitch > np.array(baseline_df['log_pitch'])) * 2 - 1
                    intensity_arousal = np.nanmean(median_intensity > np.array(baseline_df['pcm_intensity_sma'])) * 2 - 1
                    hf_lf_ratio_arousal = np.nanmean(median_hf_lf_ratio > np.array(baseline_df['hf_lf_ratio'])) * 2 - 1

                    save_row_df = pd.DataFrame(index=[save_time_str])

                    save_row_df['pitch'] = pitch_arousal
                    save_row_df['intensity'] = intensity_arousal
                    save_row_df['hf_lf_ratio'] = hf_lf_ratio_arousal
                    save_row_df['mean_score'] = (pitch_arousal + intensity_arousal + hf_lf_ratio_arousal) / 3
                    save_row_df['ratio'] = len(tmp_df) / 2000

                    save_df = save_df.append(save_row_df)

            p_pitch = save_df.corr(method='spearman').loc['mean_score', 'pitch']
            p_intensity= save_df.corr(method='spearman').loc['mean_score', 'intensity']
            p_hf_lf_ratio = save_df.corr(method='spearman').loc['mean_score', 'hf_lf_ratio']

            norms_array = np.array([p_pitch, p_intensity, p_hf_lf_ratio]) / np.linalg.norm(np.array([p_pitch, p_intensity, p_hf_lf_ratio]))
            save_df['fusion'] = norms_array[0] * np.array(save_df['pitch']) + norms_array[1] * np.array(save_df['intensity']) + norms_array[2] * np.array(save_df['hf_lf_ratio'])

            create_folder(save_root_path.joinpath('process'))
            create_folder(save_root_path.joinpath('process', 'arousal'))
            create_folder(save_root_path.joinpath('process', 'arousal', 'rating'))
            create_folder(save_root_path.joinpath('process', 'arousal', 'rating', str(threshold)))

            save_df.to_csv(save_root_path.joinpath('process', 'arousal', 'rating', str(threshold), id + '.csv.gz'), compression='gzip')
