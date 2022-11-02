# process fg masks
taskset 100 python3 arousal_rating/process_fg_mask.py --fg_threshold 0.5

# process arousal rating baselines
python3 arousal_rating/process_baseline.py --fg_threshold 0.5

# process ratings
python3 arousal_rating/process_rating.py --fg_threshold 0.5

# process shift-based features
python3 analysis/process_shift_feat.py --fg_threshold 0.5 --rssi_threshold 150

# process participant-based features
python3 analysis/extract_arousal_feat.py --fg_threshold 0.5 --rssi_threshold 150

# extract ml features
python3 ml/extract_ml_feat.py --fg_threshold 0.5 --rssi_threshold 150
