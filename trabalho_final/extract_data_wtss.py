import json
import urllib

import pandas as pd
import requests


def get_ts(ts_dumps):
    ts_ = json.loads(ts_dumps)
    return pd.DataFrame(ts_)

def make_request_wtss(host, coverage, bands, start, end, longitude, latitude):
    url = (f"{urllib.parse.urljoin(host, f'time_series')}?" +
          f"coverage={coverage}" +
          f"&attributes={",".join(bands)}" +
          f"&start_date={start}" +
          f"&end_date={end}" +
          f"&latitude={latitude}" +
          f"&longitude={longitude}")
    print(url)
    response = requests.get(url).json().get('result', None)
    if response:
        ts_ = {'Index': response['timeline']}
        band_values = response['attributes']
        for data_ in band_values:
            ts_[data_['attribute']] = data_['values']
        return(json.dumps(ts_))
    else:
        return(json.dumps({}, ensure_ascii=False))     

wtss_url = "https://data.inpe.br/bdc/wtss/v4/"

cube_name = "mod13q1-6.1"
bands = ["NDVI", "EVI", "NIR_reflectance"]

samples = pd.read_csv('./samples/samples_mt.csv')

for row in range(0, len(samples)):
    start_date = samples['start_date'][row]
    end_date = samples['end_date'][row]
    longitude = samples['longitude'][row]
    latitude = samples['latitude'][row]
    
    samples.loc[row, 'time_series'] = make_request_wtss(wtss_url, cube_name, bands, start_date, end_date, longitude, latitude)

idx = samples.columns.get_loc('end_date') + 1
samples.insert(idx, 'cube', 'mod13q1-6.1')

samples.to_csv("./samples/samples_mt_time_series.csv", index = False)
