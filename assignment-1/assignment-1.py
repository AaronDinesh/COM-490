# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DSLab Assignment 1 - Data Science with CO2
#
# ## Hand-in Instructions
#
# - __Due: **.03.2025 23h59 CET__
# - Create a fork of this repository under your group name, if you do not yet have a group, you can fork it under your username.
# - `./setup.sh` before you can start working on this notebook.
# - `git push` your final verion to the master branch of your group's repository before the due date.
# - Set the group name variable below, e.g. GROUP_NAME='Z9'
# - Add necessary comments and discussion to make your codes readable.
# - Let us know if you need us to install additional python packages.

# %%
GROUP_NAME='X1'

# %% [markdown]
# ## Carbosense
#
# The project Carbosense establishes a uniquely dense CO2 sensor network across Switzerland to provide near-real time information on man-made emissions and CO2 uptake by the biosphere. The main goal of the project is to improve the understanding of the small-scale CO2 fluxes in Switzerland and concurrently to contribute to a better top-down quantification of the Swiss CO2 emissions. The Carbosense network has a spatial focus on the City of Zurich where more than 50 sensors are deployed. Network operations started in July 2017.
#
# <img src="http://carbosense.wdfiles.com/local--files/main:project/CarboSense_MAP_20191113_LowRes.jpg" width="500">
#
# <img src="http://carbosense.wdfiles.com/local--files/main:sensors/LP8_ZLMT_3.JPG" width="156">  <img src="http://carbosense.wdfiles.com/local--files/main:sensors/LP8_sensor_SMALL.jpg" width="300">

# %% [markdown]
# ## Description of the assignment
#
# In this assignment, we will curate a set of **CO2 measurements**, measured from cheap but inaccurate sensors, that have been deployed in the city of Zurich from the Carbosense project. The goal of the exercise is twofold: 
#
# 1. Learn how to deal with real world sensor timeseries data, and organize them efficiently using python dataframes.
#
# 2. Apply data science tools to model the measurements, and use the learned model to process them (e.g., detect drifts in the sensor measurements). 
#
# The sensor network consists of 46 sites, located in different parts of the city. Each site contains three different sensors measuring (a) **CO2 concentration**, (b) **temperature**, and (c) **humidity**. Beside these measurements, we have the following additional information that can be used to process the measurements: 
#
# 1. The **altitude** at which the CO2 sensor is located, and the GPS coordinates (latitude, longitude).
#
# 2. A clustering of the city of Zurich in 17 different city **zones** and the zone in which the sensor belongs to. Some characteristic zones are industrial area, residential area, forest, glacier, lake, etc.
#
# ## Prior knowledge
#
# The average value of the CO2 in a city is approximately 400 ppm. However, the exact measurement in each site depends on parameters such as the temperature, the humidity, the altitude, and the level of traffic around the site. For example, sensors positioned in high altitude (mountains, forests), are expected to have a much lower and uniform level of CO2 than sensors that are positioned in a business area with much higher traffic activity. Moreover, we know that there is a strong dependence of the CO2 measurements, on temperature and humidity.
#
# Given this knowledge, you are asked to define an algorithm that curates the data, by detecting and removing potential drifts. **The algorithm should be based on the fact that sensors in similar conditions are expected to have similar measurements.** 
#
# ## To start with
#
# The following csv files in the `~/shared/data/` folder will be needed: 
#
# 1. `CO2_sensor_measurements.csv`
#     
#    __Description__: It contains the CO2 measurements `CO2`, the name of the site `LocationName`, a unique sensor identifier `SensorUnit_ID`, and the time instance in which the measurement was taken `timestamp`.
#     
# 2. `temperature_humidity.csv`
#
#    __Description__: It contains the temperature and the humidity measurements for each sensor identifier, at each timestamp `Timestamp`. For each `SensorUnit_ID`, the temperature and the humidity can be found in the corresponding columns of the dataframe `{SensorUnit_ID}.temperature`, `{SensorUnit_ID}.humidity`.
#     
# 3. `sensor_metadata_updated.csv`
#
#    __Description__: It contains the name of the site `LocationName`, the zone index `zone`, the altitude in meters `altitude`, the longitude `LON`, and the latitude `LAT`. 
#
# Import the following python packages:

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

plt.rcParams['font.family'] = 'DejaVu Sans' # prevent font not found warnings

# %% [markdown]
# ## PART I: Handling time series with pandas (10 points)
#
# The following scripts will copy the data to your home folder if it is not already present. They will output the absolute path of the data in your home folder, but for portability across different user accounts, we recommend using the shorthand _~/shared/data_ in your code.

# %%
# !./setup.sh

# %%
DATA_DIR='~/shared/data/'

# %% [markdown]
# ### a) **10/10**
#
# Merge the `CO2_sensor_measurements.csv`, `temperature_humidity.csv`, and `sensors_metadata.csv`, into a single dataframe. 
#
# * The merged dataframe contains:
#     - index: the time instance `timestamp` of the measurements
#     - columns: the location of the site `LocationName`, the sensor ID `SensorUnit_ID`, the CO2 measurement `CO2`, the `temperature`, the `humidity`, the `zone`, the `altitude`, the longitude `lon` and the latitude `lat`.
#
# | timestamp | LocationName | SensorUnit_ID | CO2 | temperature | humidity | zone | altitude | lon | lat |
# |:---------:|:------------:|:-------------:|:---:|:-----------:|:--------:|:----:|:--------:|:---:|:---:|
# |    ...    |      ...     |      ...      | ... |     ...     |    ...   |  ... |    ...   | ... | ... |
#
#
#
# * For each measurement (CO2, humidity, temperature), __take the average over an interval of 30 min__. 
#
# * If there are missing measurements, __interpolate them linearly__ from measurements that are close by in time.
#
# __Hints__: The following methods could be useful
#
# 1. ```python 
# pandas.DataFrame.resample()
# ``` 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
#     
# 2. ```python
# pandas.DataFrame.interpolate()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
#     
# 3. ```python
# pandas.DataFrame.mean()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html
#     
# 4. ```python
# pandas.DataFrame.append()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html

# %% [markdown]
# #### Load and check data

# %%
# file paths for CO2, temperature_humidity, and sensor metadata
CO2_DIR = f'{DATA_DIR}/CO2_sensor_measurements.csv'
TEMP_HUM_DIR = f'{DATA_DIR}/temperature_humidity.csv'
SENSOR_DIR = f'{DATA_DIR}/sensors_metadata_updated.csv'

# load csv files using tab separator and parse dates where applicable 
co2_df = pd.read_csv(CO2_DIR, sep='\t', parse_dates=[0])
temp_hum_df = pd.read_csv(TEMP_HUM_DIR, sep='\t', parse_dates=[0])
sensor_df = pd.read_csv(SENSOR_DIR)

# %%
# check the data
# --------------co2_df----------------
print("--------------co2_df----------------")
print("Missing values:")
print(co2_df.isnull().sum())
print("co2_df shape:", co2_df.shape)
print("Unique SensorUnit_ID in co2_df:", co2_df['SensorUnit_ID'].unique())

co2_df_check = co2_df.copy()
# group by SensorUnit_ID and count the number of unique LocationName entries
location_counts = co2_df_check.groupby('SensorUnit_ID')['LocationName'].nunique()
# identify sensors with more than one unique LocationName
inconsistent_sensors = location_counts[location_counts > 1].sum()
print("Sensors with inconsistent LocationName values:")
print(inconsistent_sensors)
print("\n")


# --------------temp_hum_df----------------
print("--------------temp_hum_df----------------")
print("Missing values:")
print(temp_hum_df.isnull().sum())
print("temp_hum_resampled shape:", temp_hum_df.shape)
print("\n")


# --------------sensor_df----------------
print("--------------sensor_df----------------")
print("Missing values:")
print(sensor_df.isnull().sum())
print("sensor_df shape:", sensor_df.shape)
print("Unique LocationName in sensor_df:", sensor_df['LocationName'].unique())

# %% [markdown]
# #### prepare co2_df before merging

# %%
# Copy the orginal df
co2_df_copy = co2_df.copy()

# Sort rows by timestamp
co2_df_copy = co2_df_copy.sort_values('timestamp')
# Set timestamp as the index
co2_df_copy = co2_df_copy.set_index('timestamp')

# Resample data every 30 minutes for each sensor; compute mean CO2 and take the first LocationName per interval
# This can be done beacuse each LocationName corresponds to a unique SensorUnit_ID
co2_resampled = co2_df_copy.groupby('SensorUnit_ID').resample('30min', origin='start_day').agg({
    'CO2': 'mean',
    'LocationName': 'first'
}).reset_index()

# Interpolate missing CO2 values within each sensor group using linear interpolation.
co2_resampled['CO2'] = co2_resampled.groupby('SensorUnit_ID')['CO2'].transform(lambda x: x.interpolate(method='linear'))

# Fill missing LocationName values within each sensor group using forward and backward fill.
co2_resampled['LocationName'] = co2_resampled.groupby('SensorUnit_ID')['LocationName'].transform(lambda x: x.ffill().bfill())

# %%
# Check co2_resampled
print("Missing values co2_resampled:")
print(co2_resampled.isnull().sum())
print("\nDtypes co2_resampled:")
print(co2_resampled.dtypes)    
co2_resampled

# %%
# Problem: some sensor start later or stop earlier
# Define the expected time range and create a common 30-minute interval index
expected_start = pd.Timestamp("2017-10-01 00:00:00")
expected_end = pd.Timestamp("2017-10-31 23:30:00")
common_index = pd.date_range(start=expected_start, end=expected_end, freq="30min")

# Find sensors that start later or stop earlier
sensor_start_times = co2_resampled.groupby("SensorUnit_ID")["timestamp"].min()
sensor_end_times = co2_resampled.groupby("SensorUnit_ID")["timestamp"].max()

late_start_sensors = sensor_start_times[sensor_start_times > expected_start].index
early_stop_sensors = sensor_end_times[sensor_end_times < expected_end].index

print(f"Sensors that start too late: {list(late_start_sensors)}")
print(f"Sensors that stop too early: {list(early_stop_sensors)}")

# For each sensor, reindex its data to include all timestamps in the common index (filling missing ones with NaN)
sensor_dfs = []
for sensor_id in co2_resampled["SensorUnit_ID"].unique():
    sensor_data = co2_resampled[co2_resampled["SensorUnit_ID"] == sensor_id].copy()
    sensor_data = sensor_data.set_index("timestamp")

    # Reindex to include all time steps, filling missing ones with NaN
    sensor_data = sensor_data.reindex(common_index)

    # # Ensure SensorUnit_ID is assigned to all rows
    sensor_data["SensorUnit_ID"] = sensor_id

    # Restore LocationName
    if not sensor_data["LocationName"].dropna().empty:
        sensor_data["LocationName"] = sensor_data["LocationName"].dropna().iloc[0]

    # Fill missing Values for start-late and stop-early cases
    if sensor_id in late_start_sensors:
        sensor_data["CO2"] = sensor_data["CO2"].bfill()  # Backfill from first recorded value

    if sensor_id in early_stop_sensors:
        sensor_data["CO2"] = sensor_data["CO2"].ffill()  # Forward-fill from last recorded value

    # Reset index to restore timestamp column
    sensor_data = sensor_data.reset_index().rename(columns={"index": "timestamp"})
    sensor_dfs.append(sensor_data)

# Combine the updated DataFrames
co2_resampled_fixed = pd.concat(sensor_dfs, ignore_index=True)

# Check before final sorting and verify that every sensor now spans the full time range
print("\nShape after fixing:", co2_resampled_fixed.shape)
print("\nMissing values after fixing:")
print(co2_resampled_fixed.isnull().sum())

# Check if all sensors now start and end at the correct time
final_start_times = co2_resampled_fixed.groupby("SensorUnit_ID")["timestamp"].min()
final_end_times = co2_resampled_fixed.groupby("SensorUnit_ID")["timestamp"].max()

print(f"\nSensors still starting after {expected_start}: {list(final_start_times[final_start_times > expected_start].index)}")
print(f"Sensors still ending before {expected_end}: {list(final_end_times[final_end_times < expected_end].index)}")

# Sort the final DataFrame by SensorUnit_ID and timestamp, then reset the index
co2_resampled_fixed = co2_resampled_fixed.sort_values(["SensorUnit_ID", "timestamp"]).reset_index(drop=True)

# %%
# check co2_df_resampled
print("Final shape after fixing:", co2_resampled_fixed.shape)
print("\nMissing values after fixing:")
print(co2_resampled_fixed.isnull().sum())

# %%
# Look between 35711 and 35712
co2_resampled[35709:35715]

# %%
# Look between 35711 and 35712
co2_resampled_fixed[35709:35715]

# %% [markdown]
# #### Prepare temp_hum_df before merging

# %%
temp_hum_df_copy = temp_hum_df.copy()

# Rename "Timestamp" to "timestamp"
temp_hum_df_copy.rename(columns={"Timestamp": "timestamp"}, inplace=True)

# Reshape the DataFrame from wide to long format.
temp_hum_melt = pd.melt(temp_hum_df_copy, id_vars=["timestamp"], 
                        var_name="sensor_measure", value_name="value")

# Split "sensor_measure" into SensorUnit_ID and measurement type.
temp_hum_melt[['SensorUnit_ID', 'measurement']] = temp_hum_melt['sensor_measure'].str.split('.', expand=True)

# Pivot the long DataFrame so that each row has separate columns for temperature and humidity.
temp_hum_pivot = temp_hum_melt.pivot(index=["timestamp", "SensorUnit_ID"], 
                                     columns="measurement", values="value").reset_index()

print("Missing values before cleaning:")
print(temp_hum_pivot.isnull().sum())

# Remove the columns index name so that "measurement" does not appear as a header level.
temp_hum_pivot.columns.name = None

# Interpolate missing values for temperature and humidity using linear interpolation for each sensor.
# Using transform ensures that the alignment with the original index is maintained.
temp_hum_pivot[['temperature', 'humidity']] = temp_hum_pivot.groupby('SensorUnit_ID')[['temperature', 'humidity']].transform(
    lambda x: x.interpolate(method='linear'))

# Resample each sensor's data to 30-minute intervals.
temp_hum_pivot.sort_values('timestamp', inplace=True)
temp_hum_pivot.set_index("timestamp", inplace=True)

temp_hum_resampled = temp_hum_pivot.groupby('SensorUnit_ID').resample('30min').agg({
    'humidity': 'mean',
    'temperature': 'mean',
}).reset_index()

temp_hum_resampled['SensorUnit_ID'] = temp_hum_resampled['SensorUnit_ID'].astype(int)

# %%
# Check temp_hum_resampled
print("Missing values after cleaning, temp_hum_resampled:")
print(temp_hum_resampled.isnull().sum())
print(temp_hum_resampled.dtypes)
# display the head of the cleaned DataFrame
temp_hum_resampled

# %%
# Compare the sets of unique SensorUnit_ID values to check if both DataFrames have the same sensor IDs.
unique_df1 = set(co2_resampled_fixed['SensorUnit_ID'].unique())
unique_df2 = set(temp_hum_resampled['SensorUnit_ID'].unique())

if unique_df1 == unique_df2:
    print("Both DataFrames have the same SensorUnit_ID values.")
else:
    print("Differences found:")
    print("In df1 but not in df2:", unique_df1 - unique_df2)
    print("In df2 but not in df1:", unique_df2 - unique_df1)

# %% [markdown]
# #### Prepare sensor_df before merging

# %%
# No missing values, interpolation not needed
sensor_df_copy = sensor_df.copy()

# Drop the unnecessary columns: ["Unnamed: 0", "X", "Y"]
sensor_df_clean = sensor_df_copy.drop(columns=["Unnamed: 0", "X", "Y"])

# Rename coordinate columns for consistency
sensor_df_clean.rename(columns={"LAT": "lat", "LON": "lon"}, inplace=True)

# %%
# Check sensor_df_clean
print("Missing values:")
print(sensor_df_clean.isnull().sum())
sensor_df_clean

# %% [markdown]
# #### Merge DataFrames together

# %%
# Merge co2_resampled_fixed and temp_hum_resampled DataFrames on timestamp and SensorUnit_ID
merged_df = pd.merge(co2_resampled_fixed, temp_hum_resampled, on=['timestamp', 'SensorUnit_ID'], how='outer')

# Merge the result with sensor_df_clean on LocationName
merged_df = pd.merge(merged_df, sensor_df_clean, on=['LocationName'], how='left')

# Set timestamp as the index
merged_df.set_index('timestamp', inplace=True)

# Reorder columns to match the desired structure
merged_df = merged_df[['LocationName', 'SensorUnit_ID', 'CO2', 'temperature', 'humidity', 'zone', 'altitude', 'lon', 'lat']]

# %%
# check merged_df
print("Missing values merged_df:")
print(merged_df.isnull().sum())
merged_df

# %% [markdown]
# ## PART II: Data visualization (15 points)

# %% [markdown]
# ### a) **5/15** 
# Group the sites based on their altitude, by performing K-means clustering. 
# - Find the optimal number of clusters using the [Elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)). 
# - Wite out the formula of metric you use for Elbow curve.
# - Perform clustering with the optimal number of clusters and add an additional column `altitude_cluster` to the dataframe of the previous question indicating the altitude cluster index. 
# - Report your findings.
#
# __Note__: [Yellowbrick](http://www.scikit-yb.org/) is a very nice Machine Learning Visualization extension to scikit-learn, which might be useful to you. 

# %% [markdown]
# __Answer:__ $ Distortion = 101222.84

# %% [markdown]
# ### Optimal number of clusters using the Elbow method

# %%
# Extract altitude data from sensor_df_clean
X = sensor_df_clean[['altitude']].values

# Find the optimal number of clusters using the Elbow method
model = KMeans(init="k-means++", random_state=42)
visualizer = KElbowVisualizer(model, k=(1, 10), metric='distortion', timings=False)
visualizer.fit(X)
visualizer.show()

# Get the optimal number of clusters and distortion score
optimal_k = visualizer.elbow_value_
distortion_score = visualizer.elbow_score_
print(f"\nOptimal number of clusters: {optimal_k}\nDistortion score: {distortion_score:.2f}")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Distortion Score Elbow for K-Means Clustering
#
# In the Elbow Method, the cluster quality is measured using the **distortion score**, which is the **sum of squared errors (SSE)**. The distortion metric in the KElbowVisualizer (by default) uses the Euclidean distance, corresponding to the sum of squared distances from each point to its assigned centroid.
# $$
# \text{Distortion} = \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2
# $$
# where:
# - $k$ is the **number of clusters**.
# - $C_i$ is the **set of points** in the $i^\text{th}$ cluster.
# - $\mu_i$ is the **centroid** of the $i^\text{th}$ cluster.
# - $x$ is a **data point** in cluster $C_i$.
#
# A **lower distortion score** means points lie **closer to their centroid**, reflecting **more compact clusters**. While the **lowest** distortion occurs when each point is its own cluster, that choice would be **overly complex**. So a good trade-off aims to reduce distortion **and** complexity (low k).

# %% [markdown]
# ### Perform K-Means with optimal k value and add the altitude cluster indexes to the DataFrame with all measurments

# %%
# Fit K-Means on sensor_df_clean
X = sensor_df_clean[['altitude']].values
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)

# Add the cluster index to sensor_df_clean
sensor_df_clean['altitude_cluster'] = kmeans.labels_

# Checks sensor_df_clean
print("Missing values in sensor_df_clean after clustering:")
print(sensor_df_clean.isnull().sum(), "\n")

print("Number of sensors per cluster:")
print(sensor_df_clean['altitude_cluster'].value_counts(), "\n")

# Merge altitude_cluster back into merged_df on 'LocationName'
merged_df_clustered = merged_df.merge(
    sensor_df_clean[['LocationName', 'altitude_cluster']],
    on='LocationName',
    how='left')

merged_df_clustered = (
    merged_df
    .reset_index()  # So 'timestamp' becomes a normal column
    .merge(
        sensor_df_clean[['LocationName', 'altitude_cluster']],
        on='LocationName', 
        how='left')
    .set_index('timestamp'))  # Put 'timestamp' back as the index

# Checks merged_df_clustered
print("Missing values in merged_df_clustered:")
print(merged_df_clustered.isnull().sum(), "\n")
merged_df_clustered

# %% [markdown]
# ### Report findings

# %%
print(f"Optimal number of clusters: {optimal_k}\nDistortion score: {distortion_score:.2f}\n")

altitude_stats = merged_df_clustered.groupby('altitude_cluster')['altitude'].agg(['min','max','mean','count'])

print("Clusters Information:")
for cluster_id, row in altitude_stats.iterrows():
    # Filter rows belonging to the current cluster
    cluster_slice = merged_df_clustered[merged_df_clustered['altitude_cluster'] == cluster_id]
    
    # Count unique sensors ('LocationName' or 'SensorUnit_ID', whichever identifies a sensor uniquely)
    unique_sensors = cluster_slice['SensorUnit_ID'].nunique()
    
    print(
        f"  Cluster {cluster_id}: "
        f"Range = [{row['min']:.2f}, {row['max']:.2f}] "
        f"Mean = {row['mean']:.2f}, "
        f"Count = {int(row['count'])}, "
        f"Unique Sensors = {unique_sensors}"
    )

# %% [markdown]
# ### b) **5/15** 
#
# Use `plotly` (or other similar graphing libraries) to create an interactive plot of the monthly median CO2 measurement for each site with respect to the altitude. 
#
# Add proper title and necessary hover information to each point, and give the same color to stations that belong to the same altitude cluster.

# %%
# Compute monthly median CO2 for each site
monthly_median_co2 = (
    merged_df_clustered
    .groupby('LocationName')               # group by site
    .resample('ME')['CO2']                 # resample by month end
    .median()                              # compute median CO2 in each month
    .reset_index()                         # flatten the multi-index
    .rename(columns={'timestamp':'end_month', 'CO2': 'median_CO2'})
)

# Verify the new columns
print("Columns in monthly_median_co2:", monthly_median_co2.columns.tolist())
print(monthly_median_co2.head())

# Merge monthly medians with altitude and cluster info
unique_locations = merged_df_clustered[['LocationName', 'altitude', 'altitude_cluster']].drop_duplicates()
plot_df = pd.merge(monthly_median_co2, unique_locations, on='LocationName', how='left')

# Convert altitude_cluster to string for discrete color mapping
plot_df['altitude_cluster'] = plot_df['altitude_cluster'].astype(str)

# Check for missing median_CO2 values and print info
print(f"Missing values: {plot_df['median_CO2'].isnull().sum()}")

# Create an interactive Plotly scatter plot
fig = px.scatter(
    plot_df,
    x='altitude',
    y='median_CO2',
    color='altitude_cluster',
    color_discrete_map={"0": "green", "1": "blue"},
    hover_data=['LocationName', 'end_month'],
    title='Monthly Median CO2 vs. Altitude'
)

fig.update_layout(
    xaxis_title='Altitude [m]',
    yaxis_title='Monthly Median CO2 [ppm]',
    width=900,
    height=700,
    template='plotly_dark'
)
fig.show()

# %% [markdown]
# ### c) **5/15**
#
# Use `plotly` (or other similar graphing libraries) to plot an interactive time-varying density heatmap of the mean daily CO2 concentration for all the stations. Add proper title and necessary hover information.
#
# __Hints:__ Check following pages for more instructions:
# - [Animations](https://plotly.com/python/animations/)
# - [Density Heatmaps](https://plotly.com/python/mapbox-density-heatmaps/)

# %%
# Compute daily mean CO2 for each station
daily_mean_df = (
    merged_df_clustered
    .groupby(['LocationName', pd.Grouper(level=0, freq='D')])['CO2']
    .mean()
    .reset_index()
    .rename(columns={'CO2': 'mean_daily_CO2'})
)

# Merge in station lat/lon if not already present
station_locs = merged_df_clustered[['LocationName', 'lon', 'lat']].drop_duplicates()

plot_df = pd.merge(daily_mean_df, station_locs, on='LocationName', how='left')

# Build a time-varying density heatmap with Plotly
center_lat = plot_df['lat'].mean()
center_lon = plot_df['lon'].mean()

min_value=300
max_value=800

fig = px.density_mapbox(
    plot_df,
    lat='lat',
    lon='lon',
    z='mean_daily_CO2',
    animation_frame='timestamp',
    center=dict(lat=center_lat, lon=center_lon), 
    color_continuous_scale="Viridis",
    range_color=[min_value, max_value],
    hover_data={'LocationName': True, 'mean_daily_CO2': True, 'lat': True, 'lon': True},
    mapbox_style='carto-positron',
    zoom=11)

fig.update_layout(
    height=800,
    width=1200,
    xaxis_title='Longitude',
    yaxis_title='Latitude',
    title='Daily mean CO2 [ppm] density over Zurich',
    template='plotly_dark'
)
fig.show()

# %% [markdown]
# ## PART III: Model fitting for data curation (35 points)

# %% [markdown]
# ### a) **5/35**
#
# The domain experts in charge of these sensors report that one of the CO2 sensors `ZSBN` is exhibiting a drift on Oct. 24. Verify the drift by visualizing the CO2 concentration of the drifting sensor and compare it with some other sensors from the network. 

# %%
# Define the date range around Oct 24
start_date = "2017-10-23"
end_date   = "2017-10-25"

# Filter the merged DataFrame for that window
df_drift = merged_df_clustered.loc[start_date:end_date].copy()

# Select sensors of interest 'ZSBN', and pick a few others ("ZHRO", "BUDF" , "ZORL") for comparison
sensors_of_interest = ["ZSBN", "ZHRO", "BUDF" , "ZORL"]
df_drift = df_drift[df_drift["LocationName"].isin(sensors_of_interest)]

# Make an interactive line plot of CO2 for these sensors
fig = px.line(
    df_drift.reset_index(),
    x="timestamp",
    y="CO2",
    color="LocationName",
    title="CO2 Measurements Around Oct 24 (Suspected Drift in ZSBN)"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="CO2 [ppm]",
    legend_title="Sensor",
    hovermode='x unified',
    template='plotly_dark'
)
fig.show()


# %% [markdown]
# ### b) **10/35**
#
# The domain experts ask you if you could reconstruct the CO2 concentration of the drifting sensor had the drift not happened. You decide to:
# - Fit a linear regression model to the CO2 measurements of the site, by considering as features the covariates not affected by the malfunction (such as temperature and humidity)
# - Create an interactive plot with `plotly` (or other similar graphing libraries):
#     - the actual CO2 measurements
#     - the values obtained by the prediction of the linear model for the entire month of October
#     - the __95% confidence interval__ obtained from cross validation: assume that the prediction error follows a normal distribution and is independent of time.
# - What do you observe? Report your findings.
#
# ### Answer
# I notice that the linear regression model acurately captures the periodic nature of the CO2 levels and also corrects for the sensor drift that is observed. However it seems to be too smooth and can't accuratly capture the "spiky-ness" in the CO2 measurements. I think this can be corrected by the introduction of some periodic non-linear high frequency features.
#
#
# __Note:__ Cross validation on time series is different from that on other kinds of datasets. The following diagram illustrates the series of training sets (in orange) and validation sets (in blue). For more on time series cross validation, there are a lot of interesting articles available online. scikit-learn provides a nice method [`sklearn.model_selection.TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
#
# ![ts_cv](https://player.slideplayer.com/86/14062041/slides/slide_28.jpg)

# %%
def additionalFeature(timestamp):
    sineTimeFeature = np.sin(2 * np.pi * timestamp.hour / 24)
    cosineTimeFeature = np.cos(2 * np.pi * timestamp.hour / 24)
    doubleSineTimeFeature = np.sin(4 * np.pi * timestamp.hour / 24)
    doubleCosineTimeFeature = np.cos(4 * np.pi * timestamp.hour / 24)

    return (sineTimeFeature, cosineTimeFeature, doubleSineTimeFeature, doubleCosineTimeFeature)

def augmentData(features, timestamp):
    sineTimeFeature, cosineTimeFeature, doubleSineTimeFeature, doubleCosineTimeFeature = additionalFeature(timestamp)

    features = np.column_stack((features, sineTimeFeature, cosineTimeFeature, doubleSineTimeFeature, doubleCosineTimeFeature))

    return features

#Extract the ZSBN data

start_date = pd.Timestamp("2017-10-01 00:00:00")
end_date   = pd.Timestamp("2017-10-25 00:00:00") 

endDateTimeForBadData = pd.Timestamp("2017-10-23 23:00:00")

ZSBNGoodData = merged_df_clustered.query("LocationName == 'ZSBN' and timestamp <= @endDateTimeForBadData")
ZSBNGoodData = ZSBNGoodData.to_numpy()


ZSBNBadData = merged_df_clustered.query("LocationName == 'ZSBN' and timestamp >= @endDateTimeForBadData and timestamp <= @end_date")
ZSBNBadData = ZSBNBadData.to_numpy()

timestamps = pd.date_range(start=start_date, end=end_date, freq='30min')

timestampsForGoodData = pd.date_range(start=start_date, end=endDateTimeForBadData, freq='30min')

#Extract the temp and humidty features
features = ZSBNGoodData[:, 3:5]
features = augmentData(features, timestampsForGoodData)
features = sklearn.preprocessing.PolynomialFeatures(degree=2).fit_transform(features)

targets = ZSBNGoodData[:, 2]

bestScore = 0
bestModel = None
bestMSE = np.inf
residuals = []

for i, (trainIdx, testIdx) in enumerate(sklearn.model_selection.TimeSeriesSplit(n_splits=5).split(features)):
    model = sklearn.linear_model.LinearRegression()
    model.fit(features[trainIdx], targets[trainIdx])    
    yPred = model.predict(features[testIdx])

    residuals.extend(targets[testIdx] - yPred)

    #Calculate MSE between the predicted and actual values
    mse = sklearn.metrics.mean_squared_error(targets[testIdx], yPred)
    score = model.score(features[testIdx], targets[testIdx])
    #Save the model with the lowest MSE
    if mse < bestMSE:
        bestMSE = mse
        bestScore = score
        bestModel = model

residualsMean = np.mean(residuals)
residualsStd = np.std(residuals, ddof=1)

#Print the best score and coefficients
print(f"Best R\u00B2: {bestScore:.4f}")
#print(f"Best model coefficients: {bestModel.coef_}")
print(f"Best MSE: {bestMSE:.4f}")
print(f"Mean of Residuals: {residualsMean}")
print(f"StDev of Residuals: {residualsStd}")

confidenceWidth = 1.96 * residualsStd / np.sqrt(len(features))

fullOctoberMonthData = merged_df_clustered.query(f"LocationName == 'ZSBN' and timestamp <= @end_date").to_numpy()
timestampsForBadData = pd.date_range(start=endDateTimeForBadData, end=end_date, freq='30min')
featuresFromBadData = ZSBNBadData[:, 3:5]
featuresFromBadData = augmentData(featuresFromBadData, timestampsForBadData)
featuresFromBadData = sklearn.preprocessing.PolynomialFeatures(degree=2).fit_transform(featuresFromBadData)
regressedTargets = model.predict(featuresFromBadData)
regressedTargets = np.append(targets, regressedTargets)	

upperConfidenceInterval = regressedTargets + confidenceWidth
lowerConfidenceInterval = regressedTargets - confidenceWidth


fig = go.Figure()

lineplot1 = go.Scatter(x=timestamps, y=regressedTargets, mode='lines', name='Corrected ZSBN CO2 Values')
lineplotOriginal = go.Scatter(x=timestamps, y=fullOctoberMonthData[:, 2], line=dict(dash='dash'), name='Original ZSBN CO2 Values')
confidenceRegion = go.Scatter(x=np.concatenate([timestamps, timestamps[::-1]]), y=np.concatenate([upperConfidenceInterval, lowerConfidenceInterval[::-1]]), fill='toself', fillcolor='rgba(255, 189, 139, 0.65)', line=dict(color='rgba(255, 189, 139, 0)'), name='95% Confidence Interval')


fig.add_trace(lineplot1)
fig.add_trace(lineplotOriginal)
fig.add_trace(confidenceRegion)

fig.update_layout(template='plotly_dark',title=dict(text="Correcting Sensor Drift in ZSBN (October 2017)"),xaxis=dict(title=dict(text="Timestamp")),yaxis=dict(title=dict(text="CO2 [ppm]")), hovermode="x unified")
fig.show()


# %% [markdown]
# ### c) **10/35**
#
# In your next attempt to solve the problem, you decide to exploit the fact that the CO2 concentrations, as measured by the sensors __experiencing similar conditions__, are expected to be similar.
#
# - Find the sensors sharing similar conditions with `ZSBN`. Explain your definition of "similar condition".
# - Fit a linear regression model to the CO2 measurements of the site, by considering as features:
#     - the information of provided by similar sensors
#     - the covariates associated with the faulty sensors that were not affected by the malfunction (such as temperature and humidity).
# - Create an interactive plot with `plotly` (or other similar graphing libraries):
#     - the actual CO2 measurements
#     - the values obtained by the prediction of the linear model for the entire month of October
#     - the __confidence interval__ obtained from cross validation
# - What do you observe? Report your findings.
#
# ### Answers
# 1) The method we choose to find drift in the sensors was to calcualte the exponential moving average as well as a Kalman Filter over the data. This would show us if there was a drift in a particular direction. The graphs were visually inspected and any graphs that had a particular trend up or down were classed as "drifting" and were fixed through a linear regression. Any sites with peaks that are unusually high were also classed as anomalous and were fixed through regression.
#
# 2) When fitting the regression data from "similar" sensors were used. In this case we choose the $N$-closest sensors to the sensor of choice that wasn't in the anomalous set we described above. The distance was calculated using the lat-lon coordinates and the Haversine distance formula.
#
# 3) For features I choose the temperature and humidity and then various frequency components based on the hour, minute and day of the timestamp (this is to allow for periodic trends). These features were expanded using a polynomial regression with degree 2 (this also adds a bias term). I also included the altitude and lat-lon coordinates.
#
# 4) We also noticed that for some sensors the linear regression provided a very poor fit and sometimes yielded a negative $R^2$ score. So in order to fix them we assumed that an exponential moving average model would provide a better fit.

# %%
# Performing two types of filtering on the data to detect drift.
# The results are then plotted

def kalmanFilter(data, Q=1e-5, R=0.1):
    n = len(data)
    xEst = np.zeros(n)
    P = np.zeros(n)
    xEst[0] = data[0]
    P[0] = 1.0

    for k in range(1, n):
        xPred = xEst[k-1]
        pPred = P[k-1] + Q
        K = pPred / (pPred + R)
        xEst[k] = xPred + K * (data[k] - xPred)
        P[k] = (1 - K) * pPred
    return xEst

def exponentialMovingAverage(data, alpha=0.1):
    emaArray = np.zeros(len(data))
    emaArray[0] = data[0]
    for i in range(1, len(data)):
        emaArray[i] = alpha * data[i] + (1 - alpha) * emaArray[i-1]
    return emaArray

# --- Extract Unique Locations and Timestamps ---
allLocations = merged_df_clustered['LocationName'].unique()
timestamps = merged_df_clustered.index.unique()

# --- Store Data for Each Location ---
data_by_location = {}
for location in allLocations:
    # Extract CO₂ values for the location
    data = merged_df_clustered.query(f"LocationName == '{location}'").to_numpy()
    co2Values = data[:, 2]  # Assuming CO₂ values are in the 3rd column
    
    # Apply Kalman Filter and EMA
    co2KalmanFiltered = kalmanFilter(co2Values.flatten())
    alpha=0.1
    co2EMAFiltered = exponentialMovingAverage(co2Values.flatten(), alpha=alpha)
    
    # Store processed data
    data_by_location[location] = {
        "co2Values": co2Values.flatten(),
        "co2KalmanFiltered": co2KalmanFiltered,
        "co2EMAFiltered": co2EMAFiltered,
        "co2_mean": np.ones_like(co2Values.flatten())*np.mean(co2Values.flatten())
    }

# --- Create Interactive Plot with Slider ---
fig = go.Figure()

# Add traces for all locations (only the first one is visible initially)
for i, location in enumerate(allLocations):
    d = data_by_location[location]
    
    fig.add_trace(go.Scatter(x=timestamps, y=d["co2Values"], mode='lines', 
                             name=f"{location} - Sensor Readings", visible=(i == 0),
                             line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=timestamps, y=d["co2KalmanFiltered"], mode='lines', 
                             name=f"{location} - Kalman Filter", visible=(i == 0),
                             line=dict(color='green')))
    fig.add_trace(go.Scatter(x=timestamps, y=d["co2_mean"], mode='lines', 
                             name=f"{location} - Mean", visible=(i == 0), 
                             line=dict(color='orange', dash='dash')))
    
    fig.add_trace(go.Scatter(x=timestamps, y=d["co2EMAFiltered"], mode='lines', 
                             name=f"{location} - EMA (α={alpha})", visible=(i == 0),
                             line=dict(color='red', dash='dot')))

steps = []
n_traces_per_loc = 4  # Number of traces per location
for i, location in enumerate(allLocations):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": f"Drift Detection for {location}"}],
        label=location
    )
    
    # Enable only the traces corresponding to the selected location
    for j in range(n_traces_per_loc):
        step["args"][0]["visible"][i * n_traces_per_loc + j] = True
    
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Location: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders,
    title=f"Drift Detection for {allLocations[0]}",
    xaxis_title="Time",
    yaxis_title="CO₂ Concentration (ppm)",
    template="plotly_dark",
    hovermode="x unified"
)

fig.show()


# %%
#Helper functions for the regression below
def augmentWithTimeFeatures(features, timestamps):
    # Extract hour from the timestamp
    hours = timestamps.hour
    # Extract day of the week from the timestamp
    days = timestamps.dayofweek
    minutes = timestamps.minute

    # #Add sinusoidal augmentations of the time data
    dayPeriodic = np.sin(2 * np.pi * hours / 24)
    halfDayPeriodic = np.sin(2 * np.pi * hours / 12)
    hourlyPeriodic = np.sin(2 * np.pi * hours)
    minutePeriodic = np.sin(2 * np.pi * hours*60)   
    days = np.sin(2 * np.pi * days / 7)

    # #Add cosine aufmentations of the time data
    dayPeriodicCos = np.cos(2 * np.pi * hours / 24)
    halfDayPeriodicCos = np.cos(2 * np.pi * hours / 12)
    hourlyPeriodicCos = np.cos(2 * np.pi * hours)
    minutePeriodicCos = np.cos(2 * np.pi * hours*60)
    daysCos = np.cos(2 * np.pi * days / 7)

    

    # Add the extracted features to the input
    features = np.column_stack((features, dayPeriodic, halfDayPeriodic, hourlyPeriodic, minutePeriodic, days, dayPeriodicCos, halfDayPeriodicCos, hourlyPeriodicCos, minutePeriodicCos, daysCos))
    return features 

def augmentWithBias(features):
    # Add a bias term to the input
    bias = np.ones((features.shape[0], 1))
    features = np.column_stack((features, bias))
    return features

def haversine(lat1, lon1, lat2, lon2):
    earthRadius = 6371 # in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return earthRadius*c

# Function to find closest N sensors
def find_closest_sensors(df, target_sensor_id, N=3, exludeList=[]):
    latLonLocations = df[['LocationName', 'lat', 'lon']].drop_duplicates("LocationName")

    target = latLonLocations[latLonLocations['LocationName'] == target_sensor_id].iloc[0]
    target_lat, target_lon = target['lat'], target['lon']

    # Compute distances to all other sensors
    latLonLocations['distanceKM'] = haversine(target_lat, target_lon, latLonLocations['lat'], latLonLocations['lon'])

    closestSensors = []
    distances = []
    # Sort and exclude the target sensor
    closest_sensors = latLonLocations[latLonLocations['LocationName'] != target_sensor_id].sort_values(by='distanceKM')

    for row in closest_sensors.itertuples():
        if row.LocationName not in exludeList:
            if len(closestSensors) == N:
                break
            closestSensors.append(row.LocationName) 
            distances.append(row.distanceKM)   

    return closestSensors,distances

def doCrossValidation(goodRegionFeatures, timestampsForGoodRegion, targets, polynomalDegree=2):
    #Do Cross Validation and select the best model
    bestScore = 0
    bestModel = None
    bestMSE = np.inf
    residuals = []

    for i, (trainIdx, testIdx) in enumerate(sklearn.model_selection.TimeSeriesSplit(n_splits=7).split(goodRegionFeatures)):
        model = sklearn.linear_model.LinearRegression()
        #model = sklearn.pipeline.make_pipeline(sklearn.preprocessing.PolynomialFeatures(degree=polynomalDegree), model)  
        model.fit(goodRegionFeatures[trainIdx], targets[trainIdx])    
        yPred = model.predict(goodRegionFeatures[testIdx])

        residuals.extend(targets[testIdx] - yPred)

        #Calculate MSE between the predicted and actual values
        mse = sklearn.metrics.mean_squared_error(targets[testIdx], yPred)
    
        score = model.score(goodRegionFeatures[testIdx], targets[testIdx])
        #Save the model with the lowest MSE
        if mse < bestMSE:
            bestMSE = mse
            bestScore = score
            bestModel = model

    residualsMean = np.mean(residuals)
    residualsStd = np.std(residuals, ddof=1)

    #Print the best score and coefficients
    print(f"Best R\u00B2: {bestScore:.4f}")
    #print(f"Best model coefficients: {model.coef_}")
    print(f"Best MSE: {bestMSE:.4f}")
    print(f"Mean of Residuals: {residualsMean}")
    print(f"StDev of Residuals: {residualsStd}")

    confidenceWidth = 1.96 * residualsStd / np.sqrt(len(goodRegionFeatures))

    return bestModel, confidenceWidth

def plotWithConfidence(timestamps, regressedTargets, originalData, confidenceWidth, title):
    upperConfidenceInterval = regressedTargets + confidenceWidth
    lowerConfidenceInterval = regressedTargets - confidenceWidth

    fig = go.Figure()

    lineplot1 = go.Scatter(x=timestamps, y=regressedTargets, mode='lines', name='Corrected CO2 Values')
    lineplotOriginal = go.Scatter(x=timestamps, y=originalData, line=dict(dash='dash'), name='Original CO2 Values')
    confidenceRegion = go.Scatter(x=np.concatenate([timestamps, timestamps[::-1]]), y=np.concatenate([upperConfidenceInterval, lowerConfidenceInterval[::-1]]), fill='toself', fillcolor='rgba(255, 189, 139, 0.65)', line=dict(color='rgba(255, 189, 139, 0)'), name='95% Confidence Interval')

    fig.add_trace(lineplot1)
    fig.add_trace(lineplotOriginal)
    fig.add_trace(confidenceRegion)
    fig.update_layout(template='plotly_dark',title=dict(text=title),xaxis=dict(title=dict(text="Timestamp")),yaxis=dict(title=dict(text="CO2 [ppm]")), hovermode="x unified")
    fig.show()

def extractFeaturesAndAugment(array, polynomalDegree=2, start_date = pd.Timestamp("2017-10-01 00:00:00")):
    features = array[:, 3:5]
    altitdue = array[:, 6]
    latlon = array[:, 7:9]
    features = augmentWithTimeFeatures(features, pd.date_range(start=start_date, periods=array.shape[0], freq='30min'))
    features = sklearn.preprocessing.PolynomialFeatures(degree=2).fit_transform(features)
    features = np.column_stack((features, altitdue, latlon))
    return features



# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI,
#############################################

dfWithFaultyData = merged_df_clustered.copy()

start_date = pd.Timestamp("2017-10-01 00:00:00")
end_date   = pd.Timestamp("2017-10-31 23:30:00")
timestamps = pd.date_range(start=start_date, end=end_date, freq='30min')

#Fixing ZRHO
print("--------Attempting to fix ZHRO--------")

#Find the 3-closest sensors to ZHRO
closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZHRO", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

#Extract all the data for closest sensors
goodRegionFeatures = [] 
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])


#Get the good data from ZHRO and append it to the list
ZHROGoodDataTimestamp = pd.Timestamp("2017-10-19 16:30:00")
ZHROGoodData = dfWithFaultyData.query("LocationName == 'ZHRO' and timestamp >= @ZHROGoodDataTimestamp").to_numpy() 
ZHROFeatures = extractFeaturesAndAugment(ZHROGoodData)
goodRegionFeatures.append(ZHROFeatures)
goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])

ZHROGoodTargets = dfWithFaultyData.query("LocationName == 'ZHRO' and timestamp >= @ZHROGoodDataTimestamp").to_numpy()[:, 2]
goodRegionTargets.append(ZHROGoodTargets)
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

#Do Cross Validation and select the best model
bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
ZHROData = dfWithFaultyData.query("LocationName == 'ZHRO' and timestamp < @ZHROGoodDataTimestamp").to_numpy()
ZHRODataFull = dfWithFaultyData.query("LocationName == 'ZHRO'").to_numpy()
ZHROFeatures = extractFeaturesAndAugment(ZHROData)
regressedTargets = bestModel.predict(ZHROFeatures)
regressedTargets = np.append(regressedTargets, ZHROGoodTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZHRO", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZHRODataFull[:, 2], confidenceWidth, "Correcting Sensor Drift in ZHRO (October 2017)")


# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI,
#Fixing ZSBN
print("--------Attempting to fix ZSBN--------")
start_date = pd.Timestamp("2017-10-01 00:00:00")
end_date   = pd.Timestamp("2017-10-25 00:00:00") 

endDateTimeForBadData = pd.Timestamp("2017-10-23 23:00:00")



#Find the 3-closest sensors to ZSBN
closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZSBN", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

#Extract all the data for closest sensors
goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

#Get the good data from ZSBN and append it to the list

ZSBNGoodData = merged_df_clustered.query("LocationName == 'ZSBN' and timestamp <= @endDateTimeForBadData")
ZSBNGoodData = ZSBNGoodData.to_numpy()


ZSBNBadData = merged_df_clustered.query("LocationName == 'ZSBN' and timestamp >= @endDateTimeForBadData and timestamp <= @end_date")
ZSBNBadData = ZSBNBadData.to_numpy()

ZSBNData = merged_df_clustered.query("LocationName == 'ZSBN'")
ZSBNData = ZSBNData.to_numpy()


timestamps = pd.date_range(start=start_date, end=end_date, freq='30min')

timestampsForGoodData = pd.date_range(start=start_date, end=endDateTimeForBadData, freq='30min')

#Extract the temp and humidty features
features = extractFeaturesAndAugment(ZSBNGoodData)
goodRegionFeatures.append(features)
goodRegionTargets.append(ZSBNGoodData[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

#Do Cross Validation and select the best model
bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestampsForGoodData, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
ZSBNFeatures = extractFeaturesAndAugment(ZSBNBadData)
regressedTargets = bestModel.predict(ZSBNFeatures)
regressedTargets = np.append(ZSBNGoodData[:, 2], regressedTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZSBN", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZSBNData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZSBN (October 2017)")

# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI,
#Fixing ZPFW
print("--------Attempting to fix ZPFW--------")

#Find the 3-closest sensors to ZPFW
closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZPFW", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

#Extract all the data for closest sensors
goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

#convert the features in to a numpy array
ZPFWGoodDataTimestamp = pd.Timestamp("2017-10-19 16:30:00")
ZPFWGoodData = dfWithFaultyData.query("LocationName == 'ZPFW' and timestamp >= @ZPFWGoodDataTimestamp").to_numpy()
ZPFWFeatures = extractFeaturesAndAugment(ZPFWGoodData)
goodRegionFeatures.append(ZPFWFeatures)
goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])

ZPFWGoodTargets = dfWithFaultyData.query("LocationName == 'ZPFW' and timestamp >= @ZPFWGoodDataTimestamp").to_numpy()[:, 2]
goodRegionTargets.append(ZPFWGoodTargets)
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

#Do Cross Validation and select the best model
bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
ZPFWData = dfWithFaultyData.query("LocationName == 'ZPFW' and timestamp < @ZPFWGoodDataTimestamp").to_numpy()
ZPFWFullData = dfWithFaultyData.query("LocationName == 'ZPFW'").to_numpy()
ZPFWFeatures = extractFeaturesAndAugment(ZPFWData)
regressedTargets = bestModel.predict(ZPFWFeatures)
regressedTargets = np.append(regressedTargets, ZPFWGoodTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZPFW", "CO2"] = regressedTargets
assert len(regressedTargets) == len(ZPFWFullData)
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZPFWFullData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZPFW (October 2017)")

# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI

#Fixing ZTBN
print("--------Attempting to fix ZTBN--------")
closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZTBN", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

#Data less than this is good
ZTBNGoodDataTimestamp = pd.Timestamp("2017-10-24 18:30:00")

ZTBNGoodData = dfWithFaultyData.query("LocationName == 'ZTBN' and timestamp <= @ZTBNGoodDataTimestamp").to_numpy()
ZTBNFeatures = extractFeaturesAndAugment(ZTBNGoodData)
goodRegionFeatures.append(ZTBNFeatures)
goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])

ZTBNGoodTargets = dfWithFaultyData.query("LocationName == 'ZTBN' and timestamp <= @ZTBNGoodDataTimestamp").to_numpy()[:, 2]
goodRegionTargets.append(ZTBNGoodTargets)
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
ZTBNData = dfWithFaultyData.query("LocationName == 'ZTBN' and timestamp > @ZTBNGoodDataTimestamp").to_numpy()
ZTBNFullData = dfWithFaultyData.query("LocationName == 'ZTBN'").to_numpy()
ZTBNFeatures = extractFeaturesAndAugment(ZTBNData)
regressedTargets = bestModel.predict(ZTBNFeatures)
regressedTargets = np.append(ZTBNGoodTargets, regressedTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZTBN", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZTBNFullData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZTBN (October 2017)")


# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI
#ZSTL
print("--------Attempting to fix ZSTL--------")
closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZSTL", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
ZSTLData = dfWithFaultyData.query("LocationName == 'ZSTL'").to_numpy()
ZSTLFeatures = extractFeaturesAndAugment(ZSTLData)
regressedTargets = bestModel.predict(ZSTLFeatures)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZSTL", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZSTLData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZSTL (October 2017)")


# %%
# Fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI

#ZBRC
print("--------Attempting to fix ZSTL--------")
closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZBRC", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
ZBRCData = dfWithFaultyData.query("LocationName == 'ZBRC'").to_numpy()
ZBRCFeatures = extractFeaturesAndAugment(ZBRCData)
regressedTargets = bestModel.predict(ZBRCFeatures)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZBRC", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZBRCData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZBRC (October 2017)")


# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI
#WMOO
print("--------Attempting to fix WMOO--------")
WMOOData = dfWithFaultyData.query("LocationName == 'WMOO'").to_numpy()

alpha = 0.1

meanCO2 = WMOOData[:, 2].mean()

regressedTargets = exponentialMovingAverage(WMOOData[:, 2], alpha=alpha)

print("--------Using the EMA Model to correct WMOO--------")
#Computing Residual sum of squares
residuals = WMOOData[:, 2] - regressedTargets
mae = np.sum(np.abs(residuals)) / len(regressedTargets)
score = 1 - np.sum(residuals**2) / np.sum((WMOOData[:, 2] - meanCO2)**2)
print(f"The R\u00B2 value for WMOO is: {score}")
print(f"The MSE is: {mae}")
confidenceWidth = 1.96 * np.std(residuals) / np.sqrt(len(residuals))
print(f"95% Confidence Width: {confidenceWidth}")

#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "WMOO", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, WMOOData[:, 2], confidenceWidth, f"Correcting Sensor Drift in WMOO with EMA alpha={alpha} (October 2017)")


print("--------Using the Polynomial Model to correct WMOO--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "WMOO", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
WMOOData = dfWithFaultyData.query("LocationName == 'WMOO'").to_numpy()
WMOOFeatures = extractFeaturesAndAugment(WMOOData)
regressedTargets = bestModel.predict(WMOOFeatures)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "WMOO", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, WMOOData[:, 2], confidenceWidth, "Correcting Sensor Drift in WMOO (October 2017)")

# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI
#BSCR
print("--------Attempting to fix BSCR--------")
BSCRData = dfWithFaultyData.query("LocationName == 'BSCR'").to_numpy()

alpha = 0.1

meanCO2 = BSCRData[:, 2].mean()

regressedTargets = exponentialMovingAverage(BSCRData[:, 2], alpha=alpha)
print("--------Using the EMA Model to correct BSCR--------")
#Computing Residual sum of squares
residuals = BSCRData[:, 2] - regressedTargets
mae = np.sum(np.abs(residuals)) / len(regressedTargets)
score = 1 - np.sum(residuals**2) / np.sum((BSCRData[:, 2] - meanCO2)**2)
print(f"The R\u00B2 value for BSCR is: {score}")
print(f"The MSE is: {mae}")
confidenceWidth = 1.96 * np.std(residuals) / np.sqrt(len(residuals))
print(f"95% Confidence Width: {confidenceWidth}")


#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "BSCR", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, BSCRData[:, 2], confidenceWidth, f"Correcting Sensor Drift in BSCR with EMA alpha={alpha} (October 2017)")


print("--------Using the Polynomial Model to correct BSCR--------")
closestSensors, distances = find_closest_sensors(dfWithFaultyData, "BSCR", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])
goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
BSCRData = dfWithFaultyData.query("LocationName == 'BSCR'").to_numpy()
BSCRFeatures = extractFeaturesAndAugment(BSCRData)
regressedTargets = bestModel.predict(BSCRFeatures)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "BSCR", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, BSCRData[:, 2], confidenceWidth, "Correcting Sensor Drift in BSCR (October 2017)")

# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI
#RCTZ
print("--------Attempting to fix RCTZ--------")
closestSensors, distances = find_closest_sensors(dfWithFaultyData, "RCTZ", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
RCTZData = dfWithFaultyData.query("LocationName == 'RCTZ'").to_numpy()
RCTZFeatures = extractFeaturesAndAugment(RCTZData)
regressedTargets = bestModel.predict(RCTZFeatures)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "RCTZ", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, RCTZData[:, 2], confidenceWidth, "Correcting Sensor Drift in RCTZ (October 2017)")

# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI
#SZGL
print("--------Attempting to fix SZGL--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "SZGL", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
SZGLData = dfWithFaultyData.query("LocationName == 'SZGL'").to_numpy()
SZGLFeatures = extractFeaturesAndAugment(SZGLData)
regressedTargets = bestModel.predict(SZGLFeatures)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "SZGL", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, SZGLData[:, 2], confidenceWidth, "Correcting Sensor Drift in SZGL (October 2017)")

# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI

#SMHK
print("--------Attempting to fix SMHK--------")
SMHKData = dfWithFaultyData.query("LocationName == 'SMHK'").to_numpy()

alpha = 0.1

meanCO2 = SMHKData[:, 2].mean()

print("--------Using the EMA Model to correct SMHK--------")
regressedTargets = exponentialMovingAverage(SMHKData[:, 2], alpha=alpha)

#Computing Residual sum of squares
residuals = SMHKData[:, 2] - regressedTargets
residualsMean = residuals.mean()
residualsStd = residuals.std()
mae = np.sum(np.abs(residuals)) / len(regressedTargets)
mse = np.sum(np.square(residuals)) / len(regressedTargets)
score = 1 - np.sum(residuals**2) / np.sum((SMHKData[:, 2] - meanCO2)**2)

confidenceWidth = 1.96 * residualsStd / np.sqrt(len(regressedTargets))

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Best R\u00B2: {score}")

plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, SMHKData[:, 2], confidenceWidth, f"Correcting Sensor Drift in SMHK with Exponential Moving Average alpha={alpha} (October 2017)")

print("--------Using the Polynomial Model to correct SMHK--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "SMHK", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
SMHKData = dfWithFaultyData.query("LocationName == 'SMHK'").to_numpy()
SMHKFeatures = extractFeaturesAndAugment(SMHKData)
regressedTargets = bestModel.predict(SMHKFeatures)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "SMHK", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, SMHKData[:, 2], confidenceWidth, "Correcting Sensor Drift in SMHK (October 2017)")

# %%
#Sites to fix ZHRO, (ZSBN), ZPFW, ZTBN, ZSTL, ZBRC, WMOO, BSCR, RCTZ, SZGL, SMHK, UTLI
#UTLI
print("--------Attempting to fix UTLI--------")
UTLIData = dfWithFaultyData.query("LocationName == 'UTLI'").to_numpy()

alpha = 0.05

meanCO2 = UTLIData[:, 2].mean()

regressedTargets = exponentialMovingAverage(UTLIData[:, 2], alpha=alpha)
print("--------Using the EMA Model to correct UTLI--------")
#Computing Residual sum of squares
residuals = UTLIData[:, 2] - regressedTargets
residualsMean = residuals.mean()
residualsStd = residuals.std()
mae = np.sum(np.abs(residuals)) / len(regressedTargets)
mse = np.sum(np.square(residuals)) / len(regressedTargets)
score = 1 - np.sum(residuals**2) / np.sum((UTLIData[:, 2] - meanCO2)**2)

confidenceWidth = 1.96 * residualsStd / np.sqrt(len(regressedTargets))

print(f"Best R\u00B2: {score}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"95% Confidence Width: {confidenceWidth}")
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, UTLIData[:, 2], confidenceWidth, f"Correcting Sensor Drift in UTLI with Exponential Moving Average alpha={alpha} (October 2017)")

print("--------Using the Polynomial Model to correct UTLI--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "UTLI", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

bestModel, confidenceWidth = doCrossValidation(goodRegionFeatures, timestamps, goodRegionTargets, polynomalDegree=2)
print(f"95% Confidence Width: {confidenceWidth}")
UTLIData = dfWithFaultyData.query("LocationName == 'UTLI'").to_numpy()
UTLIFeatures = extractFeaturesAndAugment(UTLIData)
regressedTargets = bestModel.predict(UTLIFeatures)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "UTLI", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, UTLIData[:, 2], confidenceWidth, "Correcting Sensor Drift in UTLI (October 2017)")


# %% [markdown]
# ### d) **10/35**
#
# Now, instead of feeding the model with all features, you want to do something smarter by using linear regression with fewer features.
#
# - Start with the same sensors and features as in question c)
# - Leverage at least two different feature selection methods
# - Create similar interactive plot as in question c)
# - Describe the methods you choose and report your findings
#
#

# %% [markdown]
# ## Preamble
# We found that only using the features provided in the dataset was not enough to model the data. We found that these gave us low (or in some cases negative) $R^2$ score. So instead of performing feature selection on the provided features, we decided to perform it on the features we developed in Q3c. The feature we selected were:
# 1) Temperature
# 2) Humidity
# 3) Time-dependent features (hour and day of the week)
# 4) Latitude and Longitude
# 5) Altitude
#
# The temperature and humidity were chosen since those were the features suggested in the question. We also noticed that there was a periodic component to the $CO_2$ data. This makes sense since we expect that $CO_2$ emissions due to cars and people would be periodic. So to model this trend we decided to make use of sinusoidal encodings of the hour and the day. This will allow us to capture daily trends as well as hourly, half-hourly and minute trends. These features were then expanded through the use of a polynomial augmentation with degree 2. We also surmised that there was a spatial correlation to the $CO_2$ emissions. We expect that $CO_2$ sensors that are closer together should see similar trends in $CO_2$ levels as traffic passes by. This is why we decided to include the Latitude and Longitiude coordinates for each sensor. Finally we also suspected that the altitude of each sensor played a role in the recorded $CO_2$ levels. We expect that the readings recorded from a high altitude sensors would be different than one from a sensor at a low altitude in the city. It should be noted that in our testing we saw significant degradation in our model from a polynomial expansion on the spatial features and so they were left untouched.
#
# As seen in the previous question, we used an exponential moving average for some of the sensors as they provided a better fit of the data rather than a linear regression model. In this question we will investigate to see if feature pruning might improve our models
#
#
#
# ## Method 1: Mutual Information Regression
#
# This takes a statistical signal processing approach to feature selection. Here we compute the Mutual information between the features and the target. With the mutual information being defined as:
# $$
# \begin{align*}
#     I(X; Y) = \sum_{x,y}{p(x, y)\log\frac{p(x, y)}{p(x)p(y)}}
# \end{align*}
# $$
# This measures the reduction in the uncertainty of $Y$ given that we know $X$. First the method estimates the probability distribution of $X$ and $Y$. Then using the fomula above we can compute the Mutual Information between the features and the target. The features with the highest Mutual Information scores are selected. The ones with scores close to 0 can be dropped as they provide very little information about the target. This method has the benefit of working for both linear and non-linear models. We then iteratively compute the $R^2$ scores for increasing number of features until the best model is found.
#
# We noticed that this method allowed us to drastically reduce the number of features needed to model some of the sites. However this method didn't work well on sites that had extremely irregular data such was (WMOO). The data in these sites had seemingly random out of the ordinary peaks that would last for an hour. This made the Mutual Information method less effective. Showing that this method could be sensitive to large outliers in the data.
#
# One other thing that was interesting to note is that, in most of the cases the Mutual Information method decided to keep some of the sinusoidal encoding of the time features in the final model. Meaning that these encodings provide a lot of information on the $CO_2$ levels. This highlights the importance of the time of day and the day itself in modelling $CO_2$ levels

# %%
def doCrossValidationWithScore(goodRegionFeatures, timestampsForGoodRegion, targets, polynomalDegree=2):
    #Do Cross Validation and select the best model
    bestScore = 0
    bestModel = None
    bestMSE = np.inf
    residuals = []

    for i, (trainIdx, testIdx) in enumerate(sklearn.model_selection.TimeSeriesSplit(n_splits=7).split(goodRegionFeatures)):
        model = sklearn.linear_model.LinearRegression()
        #model = sklearn.pipeline.make_pipeline(sklearn.preprocessing.PolynomialFeatures(degree=polynomalDegree), model)  
        model.fit(goodRegionFeatures[trainIdx], targets[trainIdx])    
        yPred = model.predict(goodRegionFeatures[testIdx])

        residuals.extend(targets[testIdx] - yPred)

        #Calculate MSE between the predicted and actual values
        mse = sklearn.metrics.mean_squared_error(targets[testIdx], yPred)
    
        score = model.score(goodRegionFeatures[testIdx], targets[testIdx])
        #Save the model with the lowest MSE
        if mse < bestMSE:
            bestMSE = mse
            bestScore = score
            bestModel = model

    residualsMean = np.mean(residuals)
    residualsStd = np.std(residuals, ddof=1)

    #Print the best score and coefficients
    #print(f"Best R\u00B2: {bestScore:.4f}")
    #print(f"Best model coefficients: {model.coef_}")
    #print(f"Best MSE: {bestMSE:.4f}")
    #print(f"Mean of Residuals: {residualsMean}")
    #print(f"StDev of Residuals: {residualsStd}")

    confidenceWidth = 1.96 * residualsStd / np.sqrt(len(goodRegionFeatures))

    return bestModel, confidenceWidth, bestScore


# %%
from sklearn.feature_selection import mutual_info_regression
print("--------Attempting to fix ZHRO--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZHRO", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

#convert the features in to a numpy array
ZHROGoodDataTimestamp = pd.Timestamp("2017-10-19 16:30:00")
ZHROGoodData = dfWithFaultyData.query("LocationName == 'ZHRO' and timestamp >= @ZHROGoodDataTimestamp").to_numpy()
ZHROFeatures = extractFeaturesAndAugment(ZHROGoodData)
goodRegionFeatures.append(ZHROFeatures)
goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])

ZHROGoodTargets = ZHROGoodData[:, 2]
goodRegionTargets.append(ZHROGoodTargets)
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []
models = []
supports = []
confidenceWidths = []
MSEs = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]): 
    #print(f"--------Testing {numFeat} features--------")
    support = selectorScoresSorted[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)

scoreMaxIdx = np.argmax(scores)

bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZHRO")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZHROData = dfWithFaultyData.query("LocationName == 'ZHRO' and timestamp < @ZHROGoodDataTimestamp").to_numpy()
ZHROFullData = dfWithFaultyData.query("LocationName == 'ZHRO'").to_numpy()
ZHROFeatures = extractFeaturesAndAugment(ZHROData)
regressedTargets = bestModel.predict(ZHROFeatures[:, bestSupport])
regressedTargets = np.append(regressedTargets, ZHROGoodTargets)
assert len(regressedTargets) == len(ZHROFullData[:, 2])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZHRO", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZHROFullData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZHRO (October 2017)")


# %%
#ZSBN
print("--------Attempting to fix ZSBN--------")

start_date = pd.Timestamp("2017-10-01 00:00:00")
end_date   = pd.Timestamp("2017-10-25 00:00:00") 

endDateTimeForBadData = pd.Timestamp("2017-10-23 23:00:00")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZSBN", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

#Extract all the data for closest sensors
goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

#Get the good data from ZSBN and append it to the list

ZSBNGoodData = merged_df_clustered.query("LocationName == 'ZSBN' and timestamp <= @endDateTimeForBadData")
ZSBNGoodData = ZSBNGoodData.to_numpy()


ZSBNBadData = merged_df_clustered.query("LocationName == 'ZSBN' and timestamp >= @endDateTimeForBadData and timestamp <= @end_date")
ZSBNBadData = ZSBNBadData.to_numpy()

ZSBNData = merged_df_clustered.query("LocationName == 'ZSBN'")
ZSBNData = ZSBNData.to_numpy()


timestamps = pd.date_range(start=start_date, end=end_date, freq='30min')

timestampsForGoodData = pd.date_range(start=start_date, end=endDateTimeForBadData, freq='30min')

#Extract the temp and humidty features
features = extractFeaturesAndAugment(ZSBNGoodData)
goodRegionFeatures.append(features)
goodRegionTargets.append(ZSBNGoodData[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []
models = []
supports = []
confidenceWidths = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    support = selectorScoresSorted[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]

    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZPFW")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZSBNFeatures = extractFeaturesAndAugment(ZSBNBadData)
regressedTargets = bestModel.predict(ZSBNFeatures[:, bestSupport])
regressedTargets = np.append(ZSBNGoodData[:, 2], regressedTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZSBN", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZSBNData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZSBN (October 2017)")

# %%
print("--------Attempting to fix ZPFW--------")
closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZPFW", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

ZPFWGoodDataTimestamp = pd.Timestamp("2017-10-19 16:30:00")
ZPFWGoodData = dfWithFaultyData.query("LocationName == 'ZPFW' and timestamp >= @ZPFWGoodDataTimestamp").to_numpy()
ZPFWFeatures = extractFeaturesAndAugment(ZPFWGoodData)
goodRegionFeatures.append(ZPFWFeatures)
goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])

ZPFWGoodTargets = dfWithFaultyData.query("LocationName == 'ZPFW' and timestamp >= @ZPFWGoodDataTimestamp").to_numpy()[:, 2]
goodRegionTargets.append(ZPFWGoodTargets)
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []
models = []
supports = []
confidenceWidths = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    support = selectorScoresSorted[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]

    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZPFW")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZPFWData = dfWithFaultyData.query("LocationName == 'ZPFW' and timestamp < @ZPFWGoodDataTimestamp").to_numpy()
ZPFWFullData = dfWithFaultyData.query("LocationName == 'ZPFW'").to_numpy()
ZPFWFeatures = extractFeaturesAndAugment(ZPFWData)
regressedTargets = bestModel.predict(ZPFWFeatures[:, bestSupport])
regressedTargets = np.append(regressedTargets, ZPFWGoodTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZPFW", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZPFWFullData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZPFW (October 2017)")

# %%
print("--------Attempting to fix ZTBN--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZTBN", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

#Data less than this is good
ZTBNGoodDataTimestamp = pd.Timestamp("2017-10-24 18:30:00")

ZTBNGoodData = dfWithFaultyData.query("LocationName == 'ZTBN' and timestamp <= @ZTBNGoodDataTimestamp").to_numpy()
ZTBNFeatures = extractFeaturesAndAugment(ZTBNGoodData)
goodRegionFeatures.append(ZTBNFeatures)
goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])

ZTBNGoodTargets = dfWithFaultyData.query("LocationName == 'ZTBN' and timestamp <= @ZTBNGoodDataTimestamp").to_numpy()[:, 2]
goodRegionTargets.append(ZTBNGoodTargets)
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []
models = []
supports = []    
confidenceWidths = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    support = selectorScoresSorted[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]

    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZTBN")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}") 
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZTBNData = dfWithFaultyData.query("LocationName == 'ZTBN' and timestamp > @ZTBNGoodDataTimestamp").to_numpy()
ZTBNFullData = dfWithFaultyData.query("LocationName == 'ZTBN'").to_numpy()
ZTBNFeatures = extractFeaturesAndAugment(ZTBNData)
regressedTargets = bestModel.predict(ZTBNFeatures[:, bestSupport])
regressedTargets = np.append(ZTBNGoodTargets, regressedTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZTBN", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZTBNFullData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZTBN (October 2017)")

# %%
print("--------Attempting to fix ZSTL--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZSTL", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])


goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    support = selectorScoresSorted[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]

    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZSTL")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZSTLData = dfWithFaultyData.query("LocationName == 'ZSTL'").to_numpy()
ZSTLFeatures = extractFeaturesAndAugment(ZSTLData)
regressedTargets = bestModel.predict(ZSTLFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZSTL", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZSTLData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZSTL (October 2017)")

# %%
print("--------Attempting to fix WMOO--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "WMOO", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    support = selectorScoresSorted[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]

    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for WMOO")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

WMOOData = dfWithFaultyData.query("LocationName == 'WMOO'").to_numpy()
WMOOFeatures = extractFeaturesAndAugment(WMOOData)
regressedTargets = bestModel.predict(WMOOFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "WMOO", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, WMOOData[:, 2], confidenceWidth, "Correcting Sensor Drift in WMOO (October 2017)")

# %%
print("--------Attempting to fix BSCR--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "BSCR", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    support = selectorScoresSorted[-numFeat:]    
    supports.append(support)
    features = goodRegionFeatures[:, support]

    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for BSCR")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

BSCRData = dfWithFaultyData.query("LocationName == 'BSCR'").to_numpy()
BSCRFeatures = extractFeaturesAndAugment(BSCRData)
regressedTargets = bestModel.predict(BSCRFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "BSCR", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, BSCRData[:, 2], confidenceWidth, "Correcting Sensor Drift in BSCR (October 2017)")

# %%
print("--------Attempting to fix ZBRC--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZBRC", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])


goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
    support = np.argsort(selectorScores)[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]

    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot    
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZBRC")    
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")    
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZBRCData = dfWithFaultyData.query("LocationName == 'ZBRC'").to_numpy()
ZBRCFeatures = extractFeaturesAndAugment(ZBRCData)
regressedTargets = bestModel.predict(ZBRCFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZBRC", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZBRCData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZBRC (October 2017)")

# %%
print("--------Attempting to fix RCTZ--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "RCTZ", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])


goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    support = selectorScoresSorted[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot    
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for RCTZ")    
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")    
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

RCTZData = dfWithFaultyData.query("LocationName == 'RCTZ'").to_numpy()
RCTZFeatures = extractFeaturesAndAugment(RCTZData)
regressedTargets = bestModel.predict(RCTZFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "RCTZ", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, RCTZData[:, 2], confidenceWidth, "Correcting Sensor Drift in RCTZ (October 2017)")

# %%
print("--------Attempting to fix SZGL--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "SZGL", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    support = selectorScoresSorted[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot    
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for SZGL")    
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")    
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

SZGLData = dfWithFaultyData.query("LocationName == 'SZGL'").to_numpy()
SZGLFeatures = extractFeaturesAndAugment(SZGLData)
regressedTargets = bestModel.predict(SZGLFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "SZGL", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, SZGLData[:, 2], confidenceWidth, "Correcting Sensor Drift in SZGL (October 2017)")

# %%
print("--------Attempting to fix SMHK--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "SMHK", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    support = selectorScoresSorted[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot    
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for SMHK")    
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")    
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

SMHKData = dfWithFaultyData.query("LocationName == 'SMHK'").to_numpy()
SMHKFeatures = extractFeaturesAndAugment(SMHKData)
regressedTargets = bestModel.predict(SMHKFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "SMHK", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, SMHKData[:, 2], confidenceWidth, "Correcting Sensor Drift in SMHK (October 2017)")

# %%
print("--------Attempting to fix UTLI--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "UTLI", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
selectorScores = mutual_info_regression(goodRegionFeatures, goodRegionTargets)
selectorScoresSorted = np.argsort(selectorScores)
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")    
    support = selectorScoresSorted[-numFeat:]
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot    
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for UTLI")    
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")    
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

UTLIData = dfWithFaultyData.query("LocationName == 'UTLI'").to_numpy()
UTLIFeatures = extractFeaturesAndAugment(UTLIData)
regressedTargets = bestModel.predict(UTLIFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "UTLI", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, UTLIData[:, 2], confidenceWidth, "Correcting Sensor Drift in UTLI (October 2017)")

# %% [markdown]
# ## Method 2: Recursive Feature Elimination
#
# The RFE algorithm perform iterative line fits on the data and drops the least important features. "Importance" in the case of a linear model is determined by its respective coefficient. RFE keeps dropping the least important features until the desired number of features are reached.
#
# We then iterativly performed RFE on the features we had selected and augmented in question c), allowing fewer and fewer features to be dropped each iteration. For each of these iterations we computed the $R^2$ score of the resulting model. The features that maximised the $R^2$ score were selected. In some cases this showed a dramatic imporvement. For example take the ZTBN site, RFE allowed us to drop the feature count from 93 down to 8.
#
# It should also be noted that RFE faired much better than the mutual information method on sites where there were large outliers in the data such as WMOO. Meaning that if there are large outliers in the data, this method could be more effective. Also for most if not all of the sites, the RFE method decided to keep some of the sinusoidal encodings of the time features in the final model. This highlights the importance of the time of day and the day itself in modelling $CO_2$ levels

# %%
#Fixing ZHRO
print("--------Attempting to fix ZHRO--------")

from sklearn.feature_selection import RFE

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZHRO", N=3, exludeList=["ZHRO", "ZSBN", "ZHRO", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])


goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

#convert the features in to a numpy array
ZHROGoodDataTimestamp = pd.Timestamp("2017-10-19 16:30:00")
ZHROGoodData = dfWithFaultyData.query("LocationName == 'ZHRO' and timestamp >= @ZHROGoodDataTimestamp").to_numpy()
ZHROFeatures = extractFeaturesAndAugment(ZHROGoodData)
goodRegionFeatures.append(ZHROFeatures)
goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])

ZHROGoodTargets = dfWithFaultyData.query("LocationName == 'ZHRO' and timestamp >= @ZHROGoodDataTimestamp").to_numpy()[:, 2]
goodRegionTargets.append(ZHROGoodTargets)
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []
models = []
supports = []
confidenceWidths = []
MSEs = []
for numFeat in range(2, goodRegionFeatures.shape[1]): 
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)

scoreMaxIdx = np.argmax(scores)

bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZHRO")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZHROData = dfWithFaultyData.query("LocationName == 'ZHRO' and timestamp < @ZHROGoodDataTimestamp").to_numpy()
ZHROFullData = dfWithFaultyData.query("LocationName == 'ZHRO'").to_numpy()
ZHROFeatures = extractFeaturesAndAugment(ZHROData)
regressedTargets = bestModel.predict(ZHROFeatures[:, bestSupport])
regressedTargets = np.append(regressedTargets, ZHROGoodTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZHRO", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZHROFullData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZHRO (October 2017)")

# %%
print("--------Attempting to fix ZSBN--------")

start_date = pd.Timestamp("2017-10-01 00:00:00")
end_date   = pd.Timestamp("2017-10-25 00:00:00") 

endDateTimeForBadData = pd.Timestamp("2017-10-23 23:00:00")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZSBN", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

#Extract all the data for closest sensors
goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

#Get the good data from ZSBN and append it to the list

ZSBNGoodData = merged_df_clustered.query("LocationName == 'ZSBN' and timestamp <= @endDateTimeForBadData")
ZSBNGoodData = ZSBNGoodData.to_numpy()


ZSBNBadData = merged_df_clustered.query("LocationName == 'ZSBN' and timestamp >= @endDateTimeForBadData and timestamp <= @end_date")
ZSBNBadData = ZSBNBadData.to_numpy()

ZSBNData = merged_df_clustered.query("LocationName == 'ZSBN'")
ZSBNData = ZSBNData.to_numpy()


timestamps = pd.date_range(start=start_date, end=end_date, freq='30min')

timestampsForGoodData = pd.date_range(start=start_date, end=endDateTimeForBadData, freq='30min')

#Extract the temp and humidty features
features = extractFeaturesAndAugment(ZSBNGoodData)
goodRegionFeatures.append(features)
goodRegionTargets.append(ZSBNGoodData[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []
models = []
supports = []
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]

    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZPFW")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZSBNFeatures = extractFeaturesAndAugment(ZSBNBadData)
regressedTargets = bestModel.predict(ZSBNFeatures[:, bestSupport])
regressedTargets = np.append(ZSBNGoodData[:, 2], regressedTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZSBN", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZSBNData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZSBN (October 2017)")

# %%
#ZPFW
print("--------Attempting to fix ZPFW--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZPFW", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

ZPFWGoodDataTimestamp = pd.Timestamp("2017-10-19 16:30:00")
ZPFWGoodData = dfWithFaultyData.query("LocationName == 'ZPFW' and timestamp >= @ZPFWGoodDataTimestamp").to_numpy()
ZPFWFeatures = extractFeaturesAndAugment(ZPFWGoodData)
goodRegionFeatures.append(ZPFWFeatures)
goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])

ZPFWGoodTargets = dfWithFaultyData.query("LocationName == 'ZPFW' and timestamp >= @ZPFWGoodDataTimestamp").to_numpy()[:, 2]
goodRegionTargets.append(ZPFWGoodTargets)
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []
models = []
supports = []
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZPFW")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZPFWData = dfWithFaultyData.query("LocationName == 'ZPFW' and timestamp < @ZPFWGoodDataTimestamp").to_numpy()
ZPFWFullData = dfWithFaultyData.query("LocationName == 'ZPFW'").to_numpy()
ZPFWFeatures = extractFeaturesAndAugment(ZPFWData)
regressedTargets = bestModel.predict(ZPFWFeatures[:, bestSupport])
regressedTargets = np.append(regressedTargets, ZPFWGoodTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZPFW", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZPFWFullData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZPFW (October 2017)")

# %%
print("--------Attempting to fix ZTBN--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZTBN", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

#Data less than this is good
ZTBNGoodDataTimestamp = pd.Timestamp("2017-10-24 18:30:00")

ZTBNGoodData = dfWithFaultyData.query("LocationName == 'ZTBN' and timestamp <= @ZTBNGoodDataTimestamp").to_numpy()
ZTBNFeatures = extractFeaturesAndAugment(ZTBNGoodData)
goodRegionFeatures.append(ZTBNFeatures)
goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])

ZTBNGoodTargets = dfWithFaultyData.query("LocationName == 'ZTBN' and timestamp <= @ZTBNGoodDataTimestamp").to_numpy()[:, 2]
goodRegionTargets.append(ZTBNGoodTargets)
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []
models = []
supports = []    
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZTBN")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}") 
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZTBNData = dfWithFaultyData.query("LocationName == 'ZTBN' and timestamp > @ZTBNGoodDataTimestamp").to_numpy()
ZTBNFullData = dfWithFaultyData.query("LocationName == 'ZTBN'").to_numpy()
ZTBNFeatures = extractFeaturesAndAugment(ZTBNData)
regressedTargets = bestModel.predict(ZTBNFeatures[:, bestSupport])
regressedTargets = np.append(ZTBNGoodTargets, regressedTargets)
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZTBN", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZTBNFullData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZTBN (October 2017)")

# %%
print("--------Attempting to fix ZSTL--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZSTL", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])


goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZSTL")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZSTLData = dfWithFaultyData.query("LocationName == 'ZSTL'").to_numpy()
ZSTLFeatures = extractFeaturesAndAugment(ZSTLData)
regressedTargets = bestModel.predict(ZSTLFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZSTL", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZSTLData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZSTL (October 2017)")

# %%
print("--------Attempting to fix WMOO--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "WMOO", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]
 
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for WMOO")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

WMOOData = dfWithFaultyData.query("LocationName == 'WMOO'").to_numpy()
WMOOFeatures = extractFeaturesAndAugment(WMOOData)
regressedTargets = bestModel.predict(WMOOFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "WMOO", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, WMOOData[:, 2], confidenceWidth, "Correcting Sensor Drift in WMOO (October 2017)")



# %%
print("--------Attempting to fix BSCR--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "BSCR", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]

    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for BSCR")
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")
fig.show()

print("###### Final Results ######")    
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

BSCRData = dfWithFaultyData.query("LocationName == 'BSCR'").to_numpy()
BSCRFeatures = extractFeaturesAndAugment(BSCRData)
regressedTargets = bestModel.predict(BSCRFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "BSCR", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, BSCRData[:, 2], confidenceWidth, "Correcting Sensor Drift in BSCR (October 2017)")

# %%
print("--------Attempting to fix ZBRC--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "ZBRC", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])


goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot    
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for ZBRC")    
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")    
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

ZBRCData = dfWithFaultyData.query("LocationName == 'ZBRC'").to_numpy()
ZBRCFeatures = extractFeaturesAndAugment(ZBRCData)
regressedTargets = bestModel.predict(ZBRCFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "ZBRC", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, ZBRCData[:, 2], confidenceWidth, "Correcting Sensor Drift in ZBRC (October 2017)")

# %%
print("--------Attempting to fix RCTZ--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "RCTZ", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])


goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot    
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for RCTZ")    
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")    
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

RCTZData = dfWithFaultyData.query("LocationName == 'RCTZ'").to_numpy()
RCTZFeatures = extractFeaturesAndAugment(RCTZData)
regressedTargets = bestModel.predict(RCTZFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "RCTZ", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, RCTZData[:, 2], confidenceWidth, "Correcting Sensor Drift in RCTZ (October 2017)")

# %%
print("--------Attempting to fix SZGL--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "SZGL", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])


goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)    
    models.append(bestModel)
    scores.append(score)
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot    
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for SZGL")    
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")    
fig.show()    

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

SZGLData = dfWithFaultyData.query("LocationName == 'SZGL'").to_numpy()
SZGLFeatures = extractFeaturesAndAugment(SZGLData)
regressedTargets = bestModel.predict(SZGLFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "SZGL", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, SZGLData[:, 2], confidenceWidth, "Correcting Sensor Drift in SZGL (October 2017)")

# %%
print("--------Attempting to fix SMHK--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "SMHK", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)    
    models.append(bestModel)
    scores.append(score)    
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)    
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot    
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for SMHK")    
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")    
fig.show()

print("###### Final Results ######")
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

SMHKData = dfWithFaultyData.query("LocationName == 'SMHK'").to_numpy()
SMHKFeatures = extractFeaturesAndAugment(SMHKData)
regressedTargets = bestModel.predict(SMHKFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "SMHK", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, SMHKData[:, 2], confidenceWidth, "Correcting Sensor Drift in SZGL (October 2017)")

# %%
print("--------Attempting to fix UTLI--------")

closestSensors, distances = find_closest_sensors(dfWithFaultyData, "UTLI", N=3, exludeList=["ZHRO", "ZSBN", "ZPFW", "ZTBN", "ZSTL", "ZBRC", "WMOO", "BSCR", "RCTZ", "SZGL", "SMHK", "UTLI"])

goodRegionFeatures = []
goodRegionTargets = []

for sensor in closestSensors:
    data = dfWithFaultyData.query(f"LocationName == '{sensor}'").to_numpy()
    features = extractFeaturesAndAugment(data)
    goodRegionFeatures.append(features)
    goodRegionTargets.append(data[:, 2])

goodRegionFeatures = np.vstack([np.array(x) for x in goodRegionFeatures])
goodRegionTargets = np.concatenate(goodRegionTargets, axis=0)

scores = []    
models = []
supports = []    
confidenceWidths = []
for numFeat in range(2, goodRegionFeatures.shape[1]):
    #print(f"--------Testing {numFeat} features--------")
    rfe = RFE(sklearn.linear_model.LinearRegression(), n_features_to_select=numFeat)
    rfe.fit(goodRegionFeatures, goodRegionTargets)
    support = rfe.get_support()
    supports.append(support)
    features = goodRegionFeatures[:, support]
    
    #Do Cross Validation and select the best model
    bestModel, confidenceWidth, score = doCrossValidationWithScore(features, timestamps, goodRegionTargets, polynomalDegree=2)
    confidenceWidths.append(confidenceWidth)    
    models.append(bestModel)
    scores.append(score)    
    #print(f"95% Confidence Width: {confidenceWidth}")

scoreMaxIdx = np.argmax(scores)    
bestSupport = supports[scoreMaxIdx]
bestModel = models[scoreMaxIdx]
confidenceWidth = confidenceWidths[scoreMaxIdx]

#Plot all the scores using a darkmode plot    
fig = px.line(x=np.arange(2, goodRegionFeatures.shape[1]), y=scores, title="Plot of R\u00b2 Scores vs Features for UTLI")    
fig.update_layout(template="plotly_dark", xaxis_title="# Features", yaxis_title="R\u00b2 Scores", hovermode="x unified")    
fig.show()

print("###### Final Results ######")    
print(f"Using {scoreMaxIdx+2} features")
print(f"Best R\u00b2 Score: {scores[scoreMaxIdx]}")
print(f"Final 95% Confidence Width: {confidenceWidth}")

UTLIData = dfWithFaultyData.query("LocationName == 'UTLI'").to_numpy()
UTLIFeatures = extractFeaturesAndAugment(UTLIData)
regressedTargets = bestModel.predict(UTLIFeatures[:, bestSupport])
#dfWithFaultyData.loc[dfWithFaultyData["LocationName"] == "UTLI", "CO2"] = regressedTargets
plotWithConfidence(pd.date_range(start=start_date, periods=len(regressedTargets), freq='30min'), regressedTargets, UTLIData[:, 2], confidenceWidth, "Correcting Sensor Drift in UTLI (October 2017)")

# %% [markdown]
# # That's all, folks!
# ### Checking to see if all my changes were saved
