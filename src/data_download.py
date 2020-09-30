import boto3
import botocore
from pathlib import Path
from datetime import datetime
from util.load_data_basic import *
import pytz
import pickle


pt = pytz.timezone('US/Pacific')
utc = pytz.timezone('UTC')


audio_feat_list = ['frameIndex', 'F0_sma', 'F0env_sma',
                   'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                   'pcm_intensity_sma', 'pcm_loudness_sma']

threshold_list = [0.4, 0.5, 0.6]


def create_folder(save_path):
	if Path.exists(save_path) is False: Path.mkdir(save_path)


def download_data(save_path, download_bucket, prefix=''):
	# Create the local saved folder
	create_folder(save_path)

	# Download data from bucket
	for object_summary in download_bucket.objects.filter(Prefix=prefix):

		if '.csv.gz' not in object_summary.key:
			continue

		if len(object_summary.key.split('/')) != 0:
			save_sub_path = save_path
			for i in range(len(object_summary.key.split('/')) - 1):
				save_sub_path = save_sub_path.joinpath(object_summary.key.split('/')[i])
				create_folder(save_sub_path)

		if Path.exists(save_path.joinpath(object_summary.key)) is True:
			print('Data already downloaded: ' + object_summary.key)
			continue

		print('Download data: ' + object_summary.key)

		try:
			download_bucket.download_file(object_summary.key, str(save_path.joinpath(object_summary.key)))
		except botocore.exceptions.ClientError as e:
			if e.response['Error']['Code'] == '404':
				print('The object does not exist.')
			else:
				raise


def read_audio(save_path, id, threshold=0.5):
	foreground_list = [str(file_str).split('/')[-1] for file_str in Path.iterdir(Path.joinpath(save_path, 'fg-predictions-csv', id))]
	foreground_list.sort()

	data_dict = {}
	for file_str in foreground_list:
		# Read data
		foreground_df = pd.read_csv(Path.joinpath(save_path, 'fg-predictions-csv', id, file_str))
		raw_feaf_df = pd.read_csv(Path.joinpath(save_path, 'raw-features', id, file_str))

		# Foreground data
		fg_array = np.argwhere(np.array(list(foreground_df['fg_prediction'])) > threshold)
		fg_feat_df = raw_feaf_df.iloc[fg_array.reshape(len(fg_array)), :][audio_feat_list]

		utc_time_str = str(file_str).split('/')[-1].split('.csv.gz')[0]
		time_str = datetime.fromtimestamp(float(float(utc_time_str) / 1000.0)).strftime(date_time_format)[:-3]
		date_str = datetime.fromtimestamp(float(float(utc_time_str) / 1000.0)).strftime(date_only_date_time_format)

		print('read data for %s, %s' % (id, time_str))
		if date_str not in list(data_dict.keys()):
			data_dict[date_str] = {}

		data_dict[date_str][time_str] = fg_feat_df

	return data_dict


if __name__ == '__main__':

	# Bucket information
	s3 = boto3.resource('s3')
	bucket_str = 'tiles-phase1-opendataset'
	audio_bucket_str = 'tiles-phase1-opendataset-audio'

	# Download the participant information data
	save_root_path = Path(__file__).parent.absolute().parents[0].joinpath('data')
	create_folder(save_root_path)

	download_data(save_root_path.joinpath(bucket_str), s3.Bucket(bucket_str), prefix='survey')
	download_data(save_root_path.joinpath(bucket_str), s3.Bucket(bucket_str), prefix='metadata')

	# Read all igtb
	igtb_df = read_AllBasic(save_root_path.joinpath(bucket_str))
	nurse_df = igtb_df.loc[igtb_df['currentposition'] == 'A']

	nurse_id_list = list(nurse_df.participant_id)
	nurse_id_list.sort()

	for id in nurse_id_list:
		download_data(save_root_path.joinpath(audio_bucket_str), s3.Bucket(audio_bucket_str), prefix='fg-predictions-csv/'+id)
		download_data(save_root_path.joinpath(audio_bucket_str), s3.Bucket(audio_bucket_str), prefix='raw-features/'+id)

		for threshold in threshold_list:
			data_dict = read_audio(save_root_path.joinpath(audio_bucket_str), id, threshold=threshold)

			create_folder(save_root_path.joinpath('process'))
			create_folder(save_root_path.joinpath('process', 'fg-audio'))
			create_folder(save_root_path.joinpath('process', 'fg-audio', str(threshold)))

			pickle.dump(data_dict, open(save_root_path.joinpath('process', 'fg-audio', str(threshold), id + '.pkl'), "wb"))

