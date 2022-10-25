# tiles-audio-arousal
TILES arousal rating using audio data

## Table of Contents
* Installation
* Preprocessing
* Analysis
* Machine Learning Experiment
* Contact

## Preprocessing

The preprocessing includes different modality:

* Location
* Audio
* Physio

### Location

To process the location data, we run the following:

```
cd location
taskset 100 python3 process_location.py
```

The above scripts process participant-based localization metric per minute. The available location types are: 

* ns (nursing station)
* pat (patient room)
* other


### Audio

To process the audio data and get the rating, we run the following:

```
cd arousal_rating
taskset 100 python3 process_fg_mask.py
taskset 100 python3 process_baseline.py
taskset 100 python3 process_rating.py
taskset 100 python3 process_shift_feat.py
```


* process_fg_mask.py: return fg features with fg posterior above threshold (default: 0.7)
* process_baseline.py: aggregate each participant's pitch, intensity, and HF/LF features
* process_rating.py: process ratings for each speech snippet
* process_shift_feat.py: process features for each segment in a shift




