# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
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
'''sensor_id_to_check = 1117
print(f"\nSample data for sensor {sensor_id_to_check}:")
print(co2_resampled_fixed[co2_resampled_fixed["SensorUnit_ID"] == sensor_id_to_check].head(20))
co2_resampled_fixed'''

# %%
co2_resampled[35709:35715]

# %%
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

# what's best: int or str?
temp_hum_resampled['SensorUnit_ID'] = temp_hum_resampled['SensorUnit_ID'].astype(int)

# %%
# Check temp_hum_resampled
print("Missing values temp_hum_resampled:")
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
# A **lower distortion score** means points lie **closer to their centroid**, reflecting **more compact clusters**. While the **lowest** distortion occurs when each point is its own cluster, that choice would be **overly complex**. So a good trade-off aims to reduce distortion and complexity (low k).

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
    
    # Count unique sensors (use 'LocationName' or 'SensorUnit_ID', whichever identifies a sensor uniquely)
    unique_sensors = cluster_slice['SensorUnit_ID'].nunique()
    
    print(
        f"  Cluster {cluster_id}: "
        f"Range = {row['min']:.2f} - {row['max']:.2f}, "
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

# Check for missing median_CO2 values and print info (do not drop them)
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
    height=700
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
    center=dict(lat=center_lat, lon=center_lon),  # or pick specific numbers
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
    title='Daily mean CO2 [ppm] density over Zurich')
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

# Select sensors of interest 'ZSBN', and pick a few others (like 'ZGHD', 'ZORL') for comparison
sensors_of_interest = ["ZSBN", "ZGHD", "ZORL"]
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
    hovermode='x unified'
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
# __Note:__ Cross validation on time series is different from that on other kinds of datasets. The following diagram illustrates the series of training sets (in orange) and validation sets (in blue). For more on time series cross validation, there are a lot of interesting articles available online. scikit-learn provides a nice method [`sklearn.model_selection.TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
#
# ![ts_cv](https://player.slideplayer.com/86/14062041/slides/slide_28.jpg)

# %%
#Extract the ZSBN data
print(merged_df_clustered)

ZSBNDataMask = merged_df_clustered['LocationName'] == "ZSBN"
ZSBNData = merged_df_clustered[ZSBNDataMask].to_numpy()

#Extract the temp and humidty features
features = ZSBNData[:, 3:5]
targets = ZSBNData[:, 2]

model = sklearn.linear_model.LinearRegression()
model.fit(features, targets)
print("The model coefficients are: ", model.coef_)
print("The R\u00B2 score is: ", model.score(features, targets))

#Generting the plotly plot of measured and fit CO2 Data.




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

# %%

# %% [markdown]
# ### d) **10/35**
#
# Now, instead of feeding the model with all features, you want to do something smarter by using linear regression with fewer features.
#
# - Start with the same sensors and features as in question c)
# - Leverage at least two different feature selection methods
# - Create similar interactive plot as in question c)
# - Describe the methods you choose and report your findings

# %% [markdown]
# __Method 1: 

# %%

# %% [markdown]
# __Method 2:

# %%

# %% [markdown]
# # That's all, folks!
