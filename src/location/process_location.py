from datetime import timedelta
from util.load_data_basic import *
import pytz

pt = pytz.timezone('US/Pacific')
utc = pytz.timezone('UTC')

audio_feat_list = ['frameIndex', 'F0_sma', 'F0env_sma',
                   'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                   'pcm_intensity_sma', 'pcm_loudness_sma']

threshold_list = [0.4, 0.5, 0.6]


def create_folder(save_path):
    if Path.exists(save_path) is False: Path.mkdir(save_path)


if __name__ == '__main__':

    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'

    # Download the participant information data
    save_root_path = Path(__file__).parent.absolute().parents[1].joinpath('data')
    create_folder(save_root_path)

    # Read all igtb
    igtb_df = read_AllBasic(save_root_path.joinpath(bucket_str))
    nurse_df = igtb_df.loc[igtb_df['currentposition'] == 'A']

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()

    days_at_work_df = read_days_at_work(save_root_path.joinpath(bucket_str))

    for id in nurse_id_list[:]:
        shift = 'day' if nurse_df.loc[nurse_df['participant_id'] == id]['Shift'].values[0] == 'Day shift' else 'night'
        if Path.exists(save_root_path.joinpath(bucket_str, 'owlinone', 'jelly', id + '.csv.gz')) is False:
            continue

        print('process %s' % (id))

        owl_in_one_df = pd.read_csv(save_root_path.joinpath(bucket_str, 'owlinone', 'jelly', id + '.csv.gz'), index_col=0)
        owl_in_one_df = owl_in_one_df.loc[owl_in_one_df['rssi'] >= 140]
        owl_in_one_df = owl_in_one_df.sort_index()

        start_date = pd.to_datetime(owl_in_one_df.index[0]).strftime(date_only_date_time_format)[:-3]
        end_date = pd.to_datetime(owl_in_one_df.index[-1]).strftime(date_only_date_time_format)[:-3]

        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        save_df = pd.DataFrame()
        for i in range(days):
            if shift == 'night':
                start_time_str = (pd.to_datetime(start_date) + timedelta(days=i, hours=19)).strftime(date_time_format)[:-3]
            else:
                start_time_str = (pd.to_datetime(start_date) + timedelta(days=i, hours=7)).strftime(date_time_format)[:-3]
            end_time_str = (pd.to_datetime(start_time_str) + timedelta(hours=12)).strftime(date_time_format)[:-3]

            work_owl_in_one_df = owl_in_one_df[start_time_str:end_time_str]
            if len(work_owl_in_one_df) == 0:
                continue

            for j in range(720):
                minute_start_str = (pd.to_datetime(start_time_str) + timedelta(minutes=j)).strftime(date_time_format)[:-3]
                minute_end_str = (pd.to_datetime(start_time_str) + timedelta(minutes=j, seconds=59)).strftime(date_time_format)[:-3]

                minute_df = work_owl_in_one_df[minute_start_str:minute_end_str]
                if len(minute_df) == 0:
                    continue

                row_df = pd.DataFrame(index=[minute_start_str])
                room_type = minute_df.max()['receiverDirectory'].split(':')[1]
                if room_type != 'ns' and room_type != 'pat':
                    room_type = 'other'

                row_df['room'] = room_type
                save_df = save_df.append(row_df)

        create_folder(save_root_path.joinpath('process'))
        create_folder(save_root_path.joinpath('process', 'owl-in-one'))
        save_df.to_csv(save_root_path.joinpath('process', 'owl-in-one', id + '.csv.gz'), compression='gzip')

