import csv
import os
import re

import pandas as pd
import geopandas as gpd
import requests

# Load an environment variable in ~/.bashrc
# https://api.census.gov/data/key_signup.html
key = os.environ.get('CENSUS_KEY')

URL = 'https://api.census.gov/data/{year}/acs/acs5'
payload = {
        'get': 'group(B11002)',
        'for': 'county:*',
        'key': key}

yr_range = range(2009, 2024)


dfs = []
for yr in yr_range:
    resp = requests.get(URL.format(year=yr), params=payload)

    # added if statement with warning in case we are working with a response != 200
    if resp.status_code != 200:
        print(f"Warning: Failed to retrieve data for {yr}. Status code: {resp.status_code}")
        continue

    # assert resp.status_code == 200
    dat = resp.json()
    df = pd.DataFrame(dat[1:], columns=dat[0])
    df['year'] = yr
    dfs.append(df)

df = pd.concat(dfs)

# Download the latest geometries for counties files into a folder called `geos/`
# https://www.census.gov/cgi-bin/geo/shapefiles/index.php

# Change 
# Used .copy() after read_file to avoid warnings
geos = gpd.read_file('geos').copy()
geos = geos.loc[:, ['GEOID', 'INTPTLAT', 'INTPTLON']]

geos['INTPTLAT'] = geos['INTPTLAT'].astype(float)
geos['INTPTLON'] = geos['INTPTLON'].astype(float)
# geos.INTPTLAT = geos.INTPTLAT.astype(float)
# geos.INTPTLON = geos.INTPTLON.astype(float)



df['GEOID'] = df.GEO_ID.apply(lambda x: re.sub(r'.+US', '', x))
# merge with geographic coordinates
bdf = df.merge(geos, on='GEOID', how='left')
# bdf = df.merge(geos, left_on='GEOID', right_on='GEOID')
bdf.drop(columns=['GEOID', 'GEO_ID'], inplace=True)

bdf.to_csv('with_geo_household_cnt.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)