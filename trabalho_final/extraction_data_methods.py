import json

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def extract_bands_ts(samples, line, bands_to_select):
    ts_ = pd.DataFrame(json.loads(samples['time_series'][line]))
    bands_ = ["Index"] + bands_to_select
    ts_ = ts_[bands_]
    return ts_

def extract_bands(samples, bands):
    samples_ = samples.copy()
    for row in range(0, len(samples_)):
        samples_.loc[row, 'time_series'] = json.dumps(extract_bands_ts(samples_, row, bands).to_dict(orient="list"))
    return samples_

def get_band_description(band, bands_description):
    selected = {}
    for band_desc in bands_description:
        if band_desc['name'] == band:
            selected = band_desc
            break
    return selected
    
def normalize_ts(samples, line, bands_description):
    ts_ = pd.DataFrame(json.loads(samples['time_series'][line]))
    for column in ts_.columns:
        if column != "Index":
            band_desc = get_band_description(column, bands_description)
            scale = band_desc['scale']
            ts_[column] = ts_[column] * scale
    return ts_

def normalize_(samples, bands_description):
    samples_ = samples.copy()
    for row in range(0, len(samples_)):
        samples_.loc[row, 'time_series'] = json.dumps(normalize_ts(samples_, row, bands_description).to_dict(orient="list"))
    return samples_

def _set_NaN(value, missing_value):
    if value != missing_value:
        return value
    else:
        return None
    
def std_(X_):
    std = X_.std(axis=0)
    std[std == 0] = 1
    return std

def interpolate_ts(samples, line, bands_description):
    ts_ = pd.DataFrame(json.loads(samples['time_series'][line]))
    for column in ts_.columns:
        if column != "Index":
            band_desc = get_band_description(column, bands_description)
            scale = band_desc['scale']
            missing_value = band_desc['nodata'] * scale
            ts_[column] = ts_[column].apply(lambda x: _set_NaN(x, missing_value)).interpolate(
                method = 'linear',
                limit_direction = 'forward',
                order = 2
            )
    return ts_

def interpolate_single_ts(ts_, bands_description):
    for column in ts_.columns:
        if column != "Index":
            band_desc = get_band_description(column, bands_description)
            scale = band_desc['scale']
            missing_value = band_desc['nodata'] * scale
            ts_[column] = ts_[column].apply(lambda x: _set_NaN(x, missing_value)).interpolate(
                method = 'linear',
                limit_direction = 'forward',
                order = 2
            )
    return ts_

def interpolate_(samples, bands_description):
    samples_ = samples.copy()
    for row in range(0, len(samples_)):
        samples_.loc[row, 'time_series'] = json.dumps(interpolate_ts(samples_, row, bands_description).to_dict(orient="list"))
    return samples_

class SGolay:
    def __init__(self, window_size: int, polynomial_order: int, mode: str = "interp"):
        self.mode = mode
        if (window_size % 2) != 0:
            self.window_size = window_size
        else:
            raise Exception("Window size must be odd number!")
        if window_size > polynomial_order:
            self.polynomial_order = polynomial_order
        else:
            raise Exception("Window size must be higher than the polynomial order!")

    def apply(self, samples, line):
        ts_ = pd.DataFrame(json.loads(samples['time_series'][line]))
        return self.apply_ts(ts_)

    def apply_ts(self, ts_):
        for column in ts_.columns:
            if column != "Index":
                ts_[column] = savgol_filter(
                    ts_[column],
                    window_length=self.window_size,
                    polyorder=self.polynomial_order,
                    mode=self.mode
                )
        return ts_

def smooth_(samples, sgolay):
    samples_ = samples.copy()
    for row in range(0, len(samples_)):
        samples_.loc[row, 'time_series'] = json.dumps(sgolay.apply(samples_, row).to_dict(orient="list"))
    return samples_

def getAllClasses(samples):
    return pd.DataFrame({
        "class_name": np.unique(samples['label']),
        "index": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "color": ["#FF7D66", "#5480FF", "#BDBDBD", "#698891", "#487D5D", "#AB5B96", "#45A2BF", "#92D199", "#92A7D1"]
    })

def getAllClassesWLTS(wlts_):
    return pd.DataFrame({
        "class_name": list(set(wlts_["class"])),
        "index": [0, 1, 2],
        "color": ["#9BD3E8", "#C9C285", "#C0F279"]
    })

def getClass(samples, index=-1, label=''):
    classes = getAllClasses(samples)
    if index >= 0:
        result = classes[index == classes["index"]]
    if len(label):
        result = classes[label == classes["class_name"]]
    result = result.reset_index(drop = True)
    return result

def extract_features(ts_string):
    ts = json.loads(ts_string.replace('""', '"'))
    all_features = []
    for key_ in ts.keys():
        if key_ != "Index":
            all_features.append(np.array(ts[key_]))
    return np.concatenate(all_features)

### Método para codificar as labels
def encode_label(samples, label_):
    return int(getClass(samples, label = label_)['index'].iloc[0])
