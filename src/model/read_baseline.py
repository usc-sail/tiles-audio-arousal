from util.load_data_basic import *
import pickle


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

    for id in nurse_id_list[40:60]:
        for threshold in threshold_list:
            save_df = pd.DataFrame()
            if Path.exists(save_root_path.joinpath('process', 'fg-audio', str(threshold), id + '.pkl')) is False:
                continue

            data_dict = pickle.load(open(save_root_path.joinpath('process', 'fg-audio', str(threshold), id + '.pkl'), 'rb'))

            for date_str in list(data_dict.keys()):
                for time_str in list(data_dict[date_str].keys()):
                    print('Appending data for %s, %s' % (id, time_str))
                    tmp_df = data_dict[date_str][time_str][arousal_feat_list]
                    save_df = save_df.append(tmp_df)

            save_df = save_df.loc[(40 < save_df['F0_sma']) & (save_df['F0_sma'] < 500)]
            save_df.loc[:, 'log_pitch'] = np.log10(np.array(save_df['F0_sma']))
            save_df.loc[:, 'hf_lf_ratio'] = np.array(save_df['pcm_fftMag_fband1000-4000_sma']) / np.array(save_df['pcm_fftMag_fband250-650_sma'])

            create_folder(save_root_path.joinpath('process'))
            create_folder(save_root_path.joinpath('process', 'arousal'))
            create_folder(save_root_path.joinpath('process', 'arousal', 'baseline'))
            create_folder(save_root_path.joinpath('process', 'arousal', 'baseline', str(threshold)))

            save_df.to_csv(save_root_path.joinpath('process', 'arousal', 'baseline', str(threshold), id + '.csv.gz'), compression='gzip')

