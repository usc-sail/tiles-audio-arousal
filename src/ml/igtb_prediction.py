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

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

baseline_dict = {'stai': 'Anxiety',
                 'pan_PosAffect': 'Positive Affect', 
                 'pan_NegAffect': 'Negative Affect', 
                 'swls': 'Life Satisfaction'}

modality_dict = {'fitbit': 'Physiological Features',
                 'speech': 'Speech Activity Features',
                 'fusion': 'Multi-modal Features'}


feature_dict = {'all_inter_session_time': 'Inter-Session Time',
                'pat_session_time_above_1min': 'Above 1min session ratio (Pat)', 
                'ns_session_time_above_1min': 'Above 1min session ratio (NS)', 
                'pat_occurance_rate': 'Speech Activity occurrence ratio (Pat)', 
                'ns_occurance_rate': 'Speech Activity occurrence ratio (NS)', 
                'mid_pat_neg_threshold': 'Negative Arousal Speech ratio (Pat, 4-9h)', 
                'start_pat_neg_threshold': 'Negative Arousal Speech ratio (Pat, 1-3h)', 
                'end_pat_neg_threshold': 'Negative Arousal Speech ratio (Pat, 10-12h)', 
                'mid_ns_neg_threshold': 'Negative Arousal Speech ratio (NS, 4-9h)', 
                'start_ns_neg_threshold': 'Negative Arousal Speech ratio (NS, 1-3h)', 
                'end_ns_neg_threshold': 'Negative Arousal Speech ratio (NS, 10-12h)', 
                'mid_pat_pos_threshold': 'Positive Arousal Speech ratio (Pat, 4-9h)', 
                'start_pat_pos_threshold': 'Positive Arousal Speech ratio (Pat, 1-3h)', 
                'end_pat_pos_threshold': 'Positive Arousal Speech ratio (Pat, 10-12h)', 
                'mid_ns_pos_threshold': 'Positive Arousal Speech ratio (NS, 4-9h)', 
                'start_ns_pos_threshold': 'Positive Arousal Speech ratio (NS, 1-3h)', 
                'end_ns_pos_threshold': 'Positive Arousal Speech ratio (NS, 10-12h)', 
                'ns_pos_threshold': 'Positive Arousal Speech ratio (NS)', 
                'ns_neg_threshold': 'Negative Arousal Speech ratio (NS)', 
                'pat_pos_threshold': 'Positive Arousal Speech ratio (Pat)', 
                'pat_neg_threshold': 'Negative Arousal Speech ratio (Pat)', 
                'step_ratio': 'Walk Activity ratio', 
                'duration': 'Sleep Duration', 
                }

def plot_feature_importances(top_features, feature_importances, baseline='pan_PosAffect', pred_opt='fitbit'):
    
    fig, axs = plt.subplots(ncols=1, figsize=(8, 4))
    axs.barh(range(len(top_features)), feature_importances, color="blue", align="center")
    # axs[0].set_yticks(range(len(day_model_name)), day_model_name[::-1])
    axs.set_yticks(range(len(top_features)))
    axs.set_yticklabels(top_features, fontsize=11)
    axs.set_xlim([0, 0.1])
    axs.set_xlabel('Feature importance', fontweight="bold", fontsize=13)
    axs.set_title('Top 10 important features\n' + 'Baseline=' + baseline_dict[baseline], fontweight="bold", fontsize=13)
    axs.xaxis.set_tick_params(labelsize=11)
    
    for idx in range(len(top_features)): axs.get_yticklabels()[idx].set_weight("bold")
    for idx in range(len(axs.get_xticklabels())): axs.get_xticklabels()[idx].set_weight("bold")
        
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_root_path.joinpath(baseline+'_'+pred_opt+'.png'), 
                dpi=400, bbox_inches='tight', pad_inches=0)
    # print()
    

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
    save_root_path = Path(os.path.realpath(__file__)).parents[1].joinpath('plot', save_setting_str, 'igtb')
    Path.mkdir(save_root_path, parents=True, exist_ok=True)
    
    ml_df = pd.read_csv(str(Path(os.path.realpath(__file__)).parents[0].joinpath(save_setting_str+'.csv')), index_col=0)
    
    # ML experiments
    demo_cols = ['shift', 'icu']
    # demo_cols = ['shift']
    # demo_cols = []
    speech_cols = [ # 'shift', 'icu',
                    # 'all_inter_session_time', 'all_session_time_above_1min', 'all_pos_threshold', 'all_neg_threshold',
                    # 'ns_occurance_rate', 'ns_session_time_above_1min', 'ns_pos_threshold', 'ns_neg_threshold',
                    # 'all_inter_session_time', 'all_session_time_above_1min', 
                    # 'all_neg_threshold',
                    # 'start_all_inter_session_time', 'mid_all_inter_session_time', 'end_all_inter_session_time',
                    'all_inter_session_time',
                    # 'mid_all_inter_session_time',
                    'all_session_time_above_1min',
                    # 'ns_session_time_above_1min',
                    # 'pat_session_time_above_1min',
                    # 'mid_all_session_time_above_1min',
                    
                    'ns_occurance_rate', 
                    'pat_occurance_rate',
                    # 'mid_ns_occurance_rate', 
                    # 'mid_pat_occurance_rate',
                    
                    # 'outside_occurance_rate',
                    # 'start_ns_neg_threshold', 
                    # 'start_pat_neg_threshold',
                    # 'start_ns_pos_threshold',
                    # 'start_pat_pos_threshold',
                    # 'mid_all_neg_threshold',
                    # 'start_all_pos_threshold',
                    # 'mid_all_pos_threshold',
                    'all_pos_threshold',
                    # 'all_neg_threshold',
                    # 'end_all_pos_threshold',
                    
                    # 'end_all_neg_threshold',
                    # 'mid_ns_neg_threshold',
                    # 'mid_pat_neg_threshold',
                    # 'mid_ns_pos_threshold',
                    'mid_pat_pos_threshold',
                    
                    # 'end_ns_neg_threshold', 
                    # 'end_pat_neg_threshold',
                    # 'end_ns_pos_threshold',
                    # 'end_pat_pos_threshold',
                    # 'mid_outside_neg_threshold',
                    # 'end_ns_neg_threshold',
                    # 'end_pat_neg_threshold'
                    ]
    # 'ns_neg_threshold', 'pat_neg_threshold',
    # 'mid_all_inter_session_time', 'mid_all_session_time_above_1min',
    # 'mid_pat_neg_threshold', 'mid_ns_neg_threshold',
    # 'mid_pat_pos_threshold', 'mid_ns_pos_threshold']
    # 'mid_pat_occurance_rate', 'mid_ns_occurance_rate']
    # 'ns_session_time_above_1min', 'pat_session_time_above_1min']
    # 'other_occurance_rate', 'other_session_time_above_1min', 'other_pos_threshold', 'other_neg_threshold']
    # fitbit_cols = ['mean_step_ratio', 'std_step_ratio', 'mean_run_ratio', 'std_run_ratio', 'mean_sleep_awake_ratio', 'std_sleep_awake_ratio', 'mean_duration', 'std_duration']
    fitbit_cols = list()
    speech_cols = list()
    for stats_col in ['mean', 'std']:
    # for stats_col in ['mean']:
        # for col in ['step_ratio', 'sleep_awake_ratio', 'duration']:
        # for col in ['step_ratio', 'run_ratio', 'duration']:
        for col in ['step_ratio', 'duration']:
            fitbit_cols.append(stats_col+'_'+col)
    # for stats_col in ['mean', 'std']:
    for stats_col in ['mean', 'std']:
        # for col in ['all_inter_session_time', 'ns_occurance_rate', 'pat_occurance_rate', 'all_pos_threshold']:
        # for col in ['all_inter_session_time', 'ns_occurance_rate', 'pat_occurance_rate', 'pat_pos_threshold', 'ns_pos_threshold']:
        # for col in ['all_inter_session_time', 'ns_occurance_rate', 'pat_occurance_rate', 'mid_ns_pos_threshold', 'mid_pat_pos_threshold', 'all_pos_threshold']:
        # for col in ['all_inter_session_time', 'ns_occurance_rate', 'pat_occurance_rate', 'pat_pos_threshold', 'ns_pos_threshold', 'all_pos_threshold']:
        # for col in ['all_inter_session_time', 'ns_occurance_rate', 'pat_occurance_rate', 'all_pos_threshold']:
        # for col in ['all_inter_session_time', 'all_session_time_above_1min', 'ns_occurance_rate', 'pat_occurance_rate', 'pat_neg_threshold', 'ns_neg_threshold']:
        for col in ['all_inter_session_time', 
                    # 'all_session_time_above_1min', 
                    'pat_session_time_above_1min', 
                    'ns_session_time_above_1min', 
                    'pat_occurance_rate', 
                    'ns_occurance_rate', 
                    # 'ns_neg_threshold', 
                    # 'pat_neg_threshold'
                    # 'other_occurance_rate',
                    # 'start_pat_neg_threshold', 'start_ns_neg_threshold', 
                    # 'ns_neg_threshold', 
                    # 'mid_ns_neg_threshold',
                    # 'pat_neg_threshold', 
                    # 'start_all_neg_threshold', 
                    # 'mid_all_neg_threshold', 
                    # 'end_all_neg_threshold', 
                    
                    'mid_pat_neg_threshold', 
                    'start_pat_neg_threshold', 
                    'end_pat_neg_threshold', 
                    
                    'mid_ns_neg_threshold', 
                    'start_ns_neg_threshold', 
                    'end_ns_neg_threshold', 
                    
                    # 'mid_other_neg_threshold', 
                    # 'start_other_neg_threshold', 
                    # 'end_other_neg_threshold', 

                    'mid_pat_pos_threshold', 
                    'start_pat_pos_threshold', 
                    'end_pat_pos_threshold', 
                    
                    'mid_ns_pos_threshold', 
                    'start_ns_pos_threshold', 
                    'end_ns_pos_threshold', 

                    # 'ns_pos_threshold', 
                    # 'mid_ns_pos_threshold', 
                    'ns_pos_threshold',
                    'ns_neg_threshold', 
                    'pat_pos_threshold',
                    'pat_neg_threshold'
                    ]:
        # for col in ['all_inter_session_time', 'all_session_time_above_1min', 'ns_occurance_rate', 'pat_occurance_rate', 'ns_pos_threshold', 'pat_pos_threshold', 'ns_neg_threshold', 'pat_neg_threshold']:
        # for col in ['all_inter_session_time', 'end_all_inter_session_time', 'ns_occurance_rate', 'pat_occurance_rate', 'end_all_pos_threshold', 'all_pos_threshold']:
            speech_cols.append(stats_col+'_'+col)
    # speech_cols.append('mean_all_session_time_above_1min')
    # speech_cols.append('mean_ns_session_time_above_1min')
    # speech_cols.append('mean_pat_session_time_above_1min')


    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True],
        'max_depth': [5, 10, 15],
        'max_features': [3, 4, 5],
        'min_samples_split': [2, 3, 4, 5],
        'n_estimators': [50, 100, 200]
    }
    
    # param_grid = {'alpha': [1e-3, 1e-2, 1e-1, 1, 5, 10, 20]}
    # pred_opt = 'fitbit'
    for pred_opt in ['fitbit', 'speech', 'fusion']:
        ml_result_df = pd.DataFrame()
        if pred_opt == 'fitbit': feat_cols = demo_cols + fitbit_cols
        elif pred_opt == 'speech': feat_cols = demo_cols + speech_cols
        elif pred_opt == 'fusion': feat_cols = demo_cols + speech_cols + fitbit_cols
        # feat_cols = speech_cols+fitbit_cols
        # feat_cols = fitbit_cols
        for col in ["stai", "pan_PosAffect", "pan_NegAffect", "swls"]:
            row_df = pd.DataFrame(index=[col])
            
            data_df = ml_df[[col]+feat_cols]
            data_df = data_df.fillna(data_df.mean())
            y = data_df[col]
            x = data_df[feat_cols]
            # pdb.set_trace()
            x = x - x.mean() / x.std()
            # Create a based model
            rf = RandomForestClassifier(random_state=8)
            # Instantiate the grid search model
            np.random.seed(8) 
            # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_micro')
            grid_search.fit(x, y)
            row_df['f1'] = grid_search.best_score_
            row_df['1s'] = np.mean(y == 1)
            row_df['0s'] = np.mean(y == 0)
            feature_importances = grid_search.best_estimator_.feature_importances_
            
            if pred_opt == 'fusion':
                top_features = list()
                for idx in np.argsort(feature_importances)[::-1][:10]:
                    if 'mean' in feat_cols[idx]: feat = 'Avg. ' + feature_dict[feat_cols[idx].split('mean_')[1]]
                    else: feat = 'Std. ' + feature_dict[feat_cols[idx].split('std_')[1]]
                    top_features.append(feat)
                plot_feature_importances(top_features, np.sort(feature_importances)[::-1][:10], baseline=col, pred_opt=pred_opt)
            # pdb.set_trace()
            for idx in range(len(feat_cols)):
                row_df[feat_cols[idx]] = feature_importances[idx]
            ml_result_df = pd.concat([ml_result_df, row_df])
            
            
        # pdb.set_trace()
        ml_result_df.to_csv(str(Path(os.path.realpath(__file__)).parents[0].joinpath(save_setting_str+'_'+pred_opt+'.csv')))
        