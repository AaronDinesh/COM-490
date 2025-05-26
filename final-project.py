# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
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
# # __Final Assignment: Robust Journey Planning__
# ---
#
# ## Hand-in Instructions
#
# - A short (7min max) video presentation of your work is due before __May 22nd, 23:59 CEST__
# - In-progress code (not necessarily functional) is due before __May 26th, 23:59__, final code is due before __May 30, 23:59 CEST__
#

# %% [markdown]
# # Introduction

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # I. Data Cleaning and Pre-processing

# %% [markdown]
# ## Setting up namespace, Spark session, and imports:

# %%
# ## Configuration and Imports
import os
import pwd
import numpy as np
import sys

from pyspark.sql import SparkSession
from random import randrange
import pyspark.sql.functions as F
#np.bool = np.bool_


username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS=os.getenv('HADOOP_FS', None)
groupName = 'X1'

print(os.getenv('SPARK_HOME'))
print(f"hadoopFSs={hadoopFS}")
print(f"username={username}")
print(f"group={groupName}")


# %%
try:
    if 'spark' in locals() and spark: # Check if spark exists and is not None
        print("Stopping existing SparkSession...")
        spark.stop()
        print("SparkSession stopped.")
except NameError:
    print("No existing SparkSession to stop.") # spark variable doesn't exist yet
except Exception as e:
    print(f"Error stopping SparkSession: {e}")

print("Creating new SparkSession...")



spark = SparkSession\
            .builder\
            .appName(pwd.getpwuid(os.getuid()).pw_name)\
            .config('spark.ui.port', randrange(4040, 4440, 5))\
            .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path)) \
            .config('spark.jars', f'{hadoopFS}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar')\
            .config('spark.sql.extensions', 'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions')\
            .config('spark.sql.catalog.iceberg', 'org.apache.iceberg.spark.SparkCatalog')\
            .config('spark.sql.catalog.iceberg.type', 'hadoop')\
            .config('spark.sql.catalog.iceberg.warehouse', f'{hadoopFS}/data/com-490/iceberg/')\
            .config('spark.sql.catalog.spark_catalog', 'org.apache.iceberg.spark.SparkSessionCatalog')\
            .config('spark.sql.catalog.spark_catalog.type', 'hadoop')\
            .config('spark.sql.catalog.spark_catalog.warehouse', f'{hadoopFS}/user/{username}/final_project/warehouse')\
            .config("spark.sql.warehouse.dir", f'{hadoopFS}/user/{username}/final_project/spark/warehouse')\
            .config('spark.eventLog.gcMetrics.youngGenerationGarbageCollectors', 'G1 Young Generation')\
            .config("spark.executor.memory", "10g")\
            .config("spark.executor.cores", "2")\
            .config("spark.executor.instances", "2")\
            .master('yarn')\
            .getOrCreate()

# %%
spark.sparkContext

# %%
spark.sql(f'CREATE SCHEMA IF NOT EXISTS spark_catalog.{username}')
spark.sql(f'USE spark_catalog.{username}')
spark.sql(f'SHOW CATALOGS').show(truncate=False)

# %% [markdown]
# ## Exploring tables and schemas:

# %%
spark.sql(f'SHOW SCHEMAS').show(truncate=False)

# %%
spark.sql(f'SHOW TABLES').show(truncate=False)

# %%
spark.sql(f'SHOW SCHEMAS IN iceberg').show(truncate=False)

# %%
spark.sql(f'SHOW TABLES IN iceberg.sbb').show(truncate=False)

# %%
spark.sql(f'SHOW TABLES IN iceberg.geo').show(truncate=False)

# %%
spark.sql(f'SELECT * FROM iceberg.sbb.stop_times LIMIT 1').show(truncate=False,vertical=True)

# %%
spark.sql(f'SELECT * FROM iceberg.sbb.trips LIMIT 1').show(truncate=False,vertical=True)

# %%
spark.sql(f'SELECT * FROM iceberg.sbb.routes LIMIT 1').show(truncate=False,vertical=True)

# %%
spark.sql(f'SELECT * FROM iceberg.sbb.calendar LIMIT 1').show(truncate=False,vertical=True)

# %%
spark.sql(f'SELECT * FROM iceberg.sbb.calendar_dates LIMIT 1').show(truncate=False,vertical=True)

# %%
spark.sql(f'SELECT * FROM iceberg.sbb.istdaten LIMIT 1').show(truncate=False,vertical=True)

# %% [markdown]
# ## Loading Weather data:

# %%
spark.read.options(header=True).csv(f'/data/com-490/csv/weather_stations').withColumns({
      'lat': F.col('lat').cast('double'),
      'lon': F.col('lon').cast('double'),
    }).createOrReplaceTempView("weather_stations")

# %%
spark.sql(f'SELECT * FROM weather_stations').printSchema()

# %%
spark.sql(f'SHOW TABLES').show(truncate=False)

# %%
spark.sql(f'SELECT * FROM weather_stations LIMIT 5').toPandas()

# %% [markdown]
# ## Define Project Parameters:

# %%

# ---
# ## Define Project Parameters
# These parameters control the scope of the data processing.
# ---

# Region Filtering 
region_names = ['Lausanne', 'Ouest lausannois']

# Timetable Filtering
target_pub_date_start = "2024-07-01" # Start of a recent week 
target_pub_date_end = "2024-07-07"   # End of that week
target_day_of_week = "monday"        # Target day for timetable (e.g., 'monday', 'tuesday', etc.)

# Istdaten Filtering (for delay modeling)
historical_start_date = "2024-01-01"
historical_end_date = "2024-12-31"

# Business Hours Filtering (seconds since midnight)
# 7:00:00 to 17:59:59 (inclusive of 17:00 hour for arrivals)
business_hours_start_sec = 7 * 3600
business_hours_end_sec = 17 * 3600 + 59 * 60 + 59

# Walking parameters
MAX_WALKING_DISTANCE_METERS = 500
WALKING_SPEED_MPS = 50 / 60 # 50 meters per minute -> meters per second

print(f"Region Names: {region_names}")
print(f"Target Timetable Pub Date Range: {target_pub_date_start} to {target_pub_date_end}")
print(f"Target Timetable Day: {target_day_of_week}")
print(f"Historical Istdaten Range: {historical_start_date} to {historical_end_date}")
print(f"Business Hours (seconds): {business_hours_start_sec} to {business_hours_end_sec}")
print(f"Max Walking Distance: {MAX_WALKING_DISTANCE_METERS}m")
print(f"Walking Speed: {WALKING_SPEED_MPS:.2f} m/s")


# %% [markdown]
# ## Defining some Helper Functions for later:

# %%
def haversine_distance_udf(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]: return None
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_gtfs_time_udf(time_str):
    if time_str is None: return None
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except Exception: return None


# %% [markdown]
# ## Loading Data:
#

# %%
stops = spark.table("iceberg.sbb.stops")
trips = spark.table("iceberg.sbb.trips")
transfers = spark.table("iceberg.sbb.transfers")
routes = spark.table("iceberg.sbb.routes")
calendar = spark.table("iceberg.sbb.calendar")
calendar_dates = spark.table("iceberg.sbb.calendar_dates")
stop_times = spark.table("iceberg.sbb.stop_times")
istdaten = spark.table("iceberg.sbb.istdaten")
geo = spark.table("iceberg.geo.shapes")
weather = spark.table("weather_stations")

# %% [markdown]
# ## Initial filtering of data:
#

# %% [markdown]
# ### Filter Calendar to only include mon-fri trips

# %%
calendar_f = calendar.filter((calendar.monday == True) & (calendar.tuesday == True)
                                & (calendar.wednesday == True) & (calendar.thursday == True) 
                                & (calendar.friday == True) & (calendar.saturday == False) 
                                & (calendar.sunday == False))

# %% [markdown]
# ### Join trips table to calendar

# %% [markdown]
# Join trips table to filtered calendar table to only include weekday trips

# %%
# Get the filtered calendar_ids
ids = calendar_f.select(calendar_f.service_id).distinct()

# Filter the trips df to only include mon-fri schedules and keep only relevant columns and join with routes df to show transportation medium.
trips_f = trips.join(ids, 'service_id', 'inner').select(['service_id', 'route_id', 'trip_id'])
trips_f = trips_f.join(routes, 'route_id', 'inner').select(['service_id', 'route_id', 'trip_id','route_desc','route_type'])
trips_f.show(3)

# %% [markdown]
# ### Filter stops table

# %% [markdown]
# Filter stops table to only include stops in the Lausanne region

# %% [markdown]
# #### Setup Trino for Geospacial queries:

# %%
import os
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable .*")

# %%
import base64 as b64
import json
import time
import re
import pandas as pd

def getUsername():
    payload = os.environ.get('EPFL_COM490_TOKEN').split('.')[1]
    payload=payload+'=' * (4 - len(payload) % 4)
    obj = json.loads(b64.urlsafe_b64decode(payload))
    if (time.time() > int(obj.get('exp')) - 3600):
        raise Exception('Your credentials have expired, please restart your Jupyter Hub server:'
                        'File>Hub Control Panel, Stop My Server, Start My Server.')
    time_left = int((obj.get('exp') - time.time())/3600)
    return obj.get('sub'), time_left


# %%
username, validity_h = getUsername()
hadoopFS = os.environ.get('HADOOP_FS')
namespace = 'iceberg.' + username
sharedNS = 'iceberg.com490_iceberg'

if not re.search('[A-Z][0-9]', groupName):
    raise Exception(f'Invalid group name {groupName}')

print(f"you are: {username}")
print(f"credentials validity: {validity_h} hours left.")
print(f"shared namespace is: {sharedNS}")
print(f"your namespace is: {namespace}")
print(f"your group is: {groupName}")

# %%
import trino
from contextlib import closing
from urllib.parse import urlparse
from trino.dbapi import connect
from trino.auth import BasicAuthentication, JWTAuthentication

trinoAuth = JWTAuthentication(os.environ.get('EPFL_COM490_TOKEN'))
trinoUrl  = urlparse(os.environ.get('TRINO_URL'))
Query=[]

print(f"Warehouse URL: {trinoUrl.scheme}://{trinoUrl.hostname}:{trinoUrl.port}/")

conn = connect(
    host=trinoUrl.hostname,
    port=trinoUrl.port,
    auth=trinoAuth,
    http_scheme=trinoUrl.scheme,
    verify=True
)

print('Connected!')

# %% [markdown]
# #### Perform the stop geofiltering using trino:

# %% [markdown]
# lausanne_region = geo.select('wkb_geometry')\
#             .filter(F.lower(F.col("name")).isin([name.lower() for name in region_names]))
# lausanne_region.show(3)


# %%
query = f"""
WITH lausanne_region AS (
    SELECT wkb_geometry
    FROM {sharedNS}.geo
    WHERE LOWER(name) IN ('lausanne', 'ouest lausannois')
),

geo_filtered_stops AS (
    SELECT 
        s.stop_id AS stop_id,
        s.stop_name,
        s.stop_lat,
        s.stop_lon,
        REGEXP_EXTRACT(stop_id, '(\\d+)') AS bpuic,
        ROW_NUMBER() OVER (PARTITION BY REGEXP_EXTRACT(stop_id, '(\\d+)') ORDER BY pub_date DESC) AS rn
    FROM {sharedNS}.sbb_stops s
    JOIN lausanne_region g
      ON ST_Contains(
          ST_GeomFromBinary(g.wkb_geometry),
          ST_Point(s.stop_lon, s.stop_lat)
      )
    WHERE pub_date >= DATE '2024-07-01'
      AND pub_date < DATE '2024-07-08'
      AND stop_id IS NOT NULL
      AND REGEXP_EXTRACT(stop_id, '(\\d+)') IS NOT NULL
)

SELECT stop_id,stop_name, stop_lat, stop_lon

FROM geo_filtered_stops
WHERE rn = 1
"""

# %%
# Fetch with Trino, convert to Pandas
with closing(conn.cursor()) as cur:
    cur.execute(query)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()

pdf = pd.DataFrame(rows, columns=cols)

 # %%
 # Convert to Spark DataFrame
stops_lausanne = spark.createDataFrame(pdf)

# (optional) inspect
stops_lausanne.printSchema()
print("Rows fetched:", stops_lausanne.count())

# %%
stops_lausanne
# ### Getting stops with realtime data for Lausanne region:


# %%
from pyspark.sql.functions import col, regexp_extract, lit, when
# 1) Build the ist_july DataFrame
ist_july_df = istdaten.filter((col("operating_day") >= lit("2024-07-01")) &\
          (col("operating_day") <  lit("2024-08-01")))\
      .select(regexp_extract(col("bpuic").cast("string"), "(\\d+)", 1).alias("bpuic"))\
      .distinct()

# %%
stops_lausanne = stops_lausanne.select(\
          col("stop_id"),\
          col("stop_name"),\
          col("stop_lat"),\
          col("stop_lon"),\
          regexp_extract(col("stop_id"), "(\\d+)", 1).alias("bpuic")\
      )\
      .distinct()

# %%
stops_lausanne.show(3)

# %%
stops_lausanne.count()

# %%
# rename the ist_july bpuic so we can see NULLs on misses
ist_july_key = ist_july_df.select(col("bpuic").alias("bpuic_rt"))

stops_lausanne = (
   (lhs:=stops_lausanne)  # your original stops-with-bpuic
      .join(ist_july_key, lhs.bpuic == ist_july_key.bpuic_rt, how="left")
      .withColumn("has_realtime", col("bpuic_rt").isNotNull())
      .drop("bpuic_rt")       # no longer needed
)

# %%
stops_lausanne.printSchema()
stops_lausanne.show(5)

# %%
from pyspark.sql.functions import avg, col
result = (\
    stops_lausanne\
      .select((1 - avg(col("has_realtime").cast("double"))) * 100)\
      .first()[0]\
)

print(f"{result:.2f}% of stops have no realtime data")

# %%
# Find the stop with ID 8592108
stop_8592108 = stops_lausanne.filter(col("stop_id") == "8592108")

# Show the result
stop_8592108.show(truncate=False)

# %%
ist_july_df.filter(col('bpuic')=="8592108").show(truncate=False)

# %% [markdown]
# ### Filtering Stop times:
#

# %% [markdown]
# #### Filtering weekday trips:

# %%
import plotly.express as px
fig = px.scatter_map(data_frame=stops_lausanne, lat="stop_lat", lon="stop_lon", color="has_realtime", hover_name="stop_name")

fig.show()

# %% [markdown]
# #### Filtering stop times in the Laus

# %%
stops_lausanne_rt = stops_lausanne.filter(col("has_realtime") == True)


# %%
stops_lausanne_rt.writeTo(f"""{namespace}.sbb_stops_lausanne_region""")\
    .using("iceberg")

# %% [markdown]
# # II. Public Transport Network Model

# %%
# %pip install networkx

# %%
# !pip install networkx

# %%
# ensure the personal spark catalog exist
username, _ = getUsername()
spark.sql(f"CREATE SCHEMA IF NOT EXISTS spark_catalog.{username}")

# %%
stops_lausanne_rt.createOrReplaceTempView("stops_lausanne_rt")

# sanity check
spark.sql("SHOW TABLES").filter("tableName = 'stops_lausanne_rt'").show()

# %%
business_hours_start = '07:00:00'
business_hours_end = '17:59:59'

# Read the shared stop_times from the iceberg.sbb catalog
shared_st = spark.table("iceberg.sbb.stop_times")

# Read lausanne stops from the spark_catalog
lausanne_st = spark.table(f"stops_lausanne_rt").select("stop_id")

# filter by pub_date, region, business hours (start, end)
stop_times_df = (
    shared_st
      .filter((col("pub_date") >= lit("2024-07-01")) &
              (col("pub_date") <  lit("2024-07-05")))
      .join(lausanne_st, on="stop_id", how="inner")
      .filter((col("departure_time") >= lit(business_hours_start)) &
              (col("arrival_time")   <= lit(business_hours_end)))
      .select("trip_id", "stop_id", "departure_time", "arrival_time"))

# %%
#sanity check
print("tot filtered rows:", stop_times_df.count())
stop_times_df.show(5, truncate=False)

# %%
'''business_hours_start = '07:00:00'
business_hours_end = '17:59:59'

query = f"""
SELECT 
    trip_id,
    stop_id,
    departure_time,
    arrival_time
FROM {sharedNS}.sbb_stop_times
WHERE 
    pub_date BETWEEN DATE '2024-07-01' AND DATE '2024-07-05'
    AND stop_id IN (SELECT stop_id FROM {namespace}.sbb_stops_lausanne_region)
    AND departure_time >= '{business_hours_start}' 
    AND arrival_time <='{business_hours_end}'
"""


with closing(conn.cursor()) as cur:
    cur.execute(query)
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()

stop_times_df = spark.createDataFrame(pd.DataFrame(rows, columns=cols))'''

# %%
joined_df = stops_lausanne.join(stop_times_df, 'stop_id', 'inner')


# %%
joined_df.printSchema()
joined_df.show(5)

# %%
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import math

haversine_distance_udf = udf(haversine_distance_udf, DoubleType())

# %%
pairwise_distances = stops_lausanne.alias("a").crossJoin(stops_lausanne.alias("b")) \
    .filter(col("a.stop_id") != col("b.stop_id")) \
    .select(
        col("a.stop_id").alias("stop_id_a"),
        col("b.stop_id").alias("stop_id_b"),
        haversine_distance_udf(
            col("a.stop_lat"), col("a.stop_lon"),
            col("b.stop_lat"), col("b.stop_lon")
        ).alias("distance")
    )

stop_to_stop_df = pairwise_distances.filter(col("distance") <= 500)

stop_to_stop_df.show(5)
stop_to_stop_df.printSchema()


# %%
from pyspark.sql.functions import to_timestamp

lausanne_stop_times = joined_df.withColumn(
    "departure_td", to_timestamp("departure_time", "HH:mm:ss")
).withColumn(
    "arrival_td", to_timestamp("arrival_time", "HH:mm:ss")
)

# %% jupyter={"source_hidden": true}
# old graph version 
# import networkx as nx
# from datetime import timedelta


# filtered_stop_times_list = lausanne_stop_times.select(
#     "trip_id", "stop_id", "departure_time", "arrival_time"
# ).collect()

# stop_to_stop_list = stop_to_stop_df.select(
#     "stop_id_a", "stop_id_b", "distance"
# ).collect()

# stop_times = [
#     {
#         "trip_id": row["trip_id"],
#         "stop_id": row["stop_id"],
#         "departure_td": timedelta(hours=int(row["departure_time"][:2]),
#                                   minutes=int(row["departure_time"][3:5]),
#                                   seconds=int(row["departure_time"][6:8])),
#         "arrival_td": timedelta(hours=int(row["arrival_time"][:2]),
#                                 minutes=int(row["arrival_time"][3:5]),
#                                 seconds=int(row["arrival_time"][6:8]))
#     }
#     for row in filtered_stop_times_list
# ]

# G = nx.DiGraph()

# from collections import defaultdict
# trip_groups = defaultdict(list)

# for row in stop_times:
#     trip_groups[row["trip_id"]].append(row)

# for trip_id, group in trip_groups.items():
#     sorted_group = sorted(group, key=lambda x: x["departure_td"])
#     previous_stop = None

#     for row in sorted_group:
#         current_stop = (row['stop_id'], row['departure_td'])
#         G.add_node(current_stop)

#         if previous_stop:
#             travel_time = (row['arrival_td'] - previous_stop[1]).total_seconds() / 60.0
#             G.add_edge(previous_stop, current_stop, weight=travel_time, trip_id=trip_id)

#         previous_stop = (row['stop_id'], row['arrival_td'])

# node_list = list(G.nodes)
# for row in stop_to_stop_list:
#     stop_a, stop_b, distance = row['stop_id_a'], row['stop_id_b'], row['distance']
#     walking_time_min = distance / (WALKING_SPEED_MPS * 60)

#     for node in node_list:
#         if node[0] == stop_a:
#             departure_time = node[1]
#             arrival_time = departure_time + timedelta(minutes=walking_time_min)
#             G.add_edge(node, (stop_b, arrival_time), weight=walking_time_min, trip_id='WALK')


# print(f"Nodes: {len(G.nodes)}")
# print(f"Edges: {len(G.edges)}")


# %%
# this is the new version where travel_time = (arrival at b) - (deparrture from a) 
# so now the edge weigth is pure travel time 
# not travel time + wait time at a 

# NOTE: Takes about 3 minutes to run

import networkx as nx
from datetime import timedelta
from collections import defaultdict


WALKING_SPEED_MPS = 50 / 60 

# Collecting data from Spark 
# lots of tries and accepts cause I had a few bugs 
filtered_stop_times_list = []
if 'lausanne_stop_times' in locals() and lausanne_stop_times is not None:
    try:
        filtered_stop_times_list = lausanne_stop_times.select(
            "trip_id", "stop_id", "departure_time", "arrival_time"
        ).collect()
        print(f"Collected {len(filtered_stop_times_list)} rows for filtered_stop_times_list.")
    except Exception as e:
        print(f"Error collecting filtered_stop_times_list: {e}")
else:
    print("'lausanne_stop_times' not found or is None")

stop_to_stop_list = []
if 'stop_to_stop_df' in locals() and stop_to_stop_df is not None:
    try:
        stop_to_stop_list = stop_to_stop_df.select(
            "stop_id_a", "stop_id_b", "distance" 
        ).collect()
        print(f"Collected {len(stop_to_stop_list)} rows for stop_to_stop_list.")
    except Exception as e:
        print(f"Error collecting stop_to_stop_list: {e}")
else:
    print("'stop_to_stop_df' not found or is None")



# Data prep for PT
stop_times_data_for_graph = [] 
if filtered_stop_times_list:
    try:
        stop_times_data_for_graph = [
            {
                "trip_id": r["trip_id"],
                "stop_id": r["stop_id"],
                "departure_td": timedelta(
                    hours=int(r["departure_time"][:2]),
                    minutes=int(r["departure_time"][3:5]),
                    seconds=int(r["departure_time"][6:8])
                ),
                "arrival_td": timedelta(
                    hours=int(r["arrival_time"][:2]),
                    minutes=int(r["arrival_time"][3:5]),
                    seconds=int(r["arrival_time"][6:8])
                )
            }
            for r in filtered_stop_times_list
        ]
        print(f"PT Data Prep: Converted {len(stop_times_data_for_graph)} entries to timedelta format.")
    except Exception as e:
        print(f"PT Data Prep: Error during timedelta conversion: {e}")
        stop_times_data_for_graph = [] 
else:
    print("PT Data Prep: 'filtered_stop_times_list' is empty. No PT data to process.")

# Grabbing trips 
trip_groups_graph = defaultdict(list)
if stop_times_data_for_graph: 
    for row_data in stop_times_data_for_graph:
        trip_groups_graph[row_data["trip_id"]].append(row_data)
    print(f"PT Data Prep: Grouped PT data into {len(trip_groups_graph)} trip_ids.")
else:
    print("PT Data Prep: 'stop_times_data_for_graph' is empty. 'trip_groups_graph' will be empty.")

# Building Graph
G = nx.DiGraph()

attempted_pt_edges = 0
non_negative_pt_edge_attempts = 0
actually_added_new_pt_edges = 0 
walking_edge_addition_attempts = 0

# Public transport edges 
if trip_groups_graph:
    print("Building PT edges...")
    for trip_id, group in trip_groups_graph.items():
        sorted_group = sorted(group, key=lambda x: x["departure_td"])
        for i in range(len(sorted_group) - 1):
            stop_a_data = sorted_group[i]
            stop_b_data = sorted_group[i+1]
            from_node = (stop_a_data['stop_id'], stop_a_data['departure_td'])
            to_node   = (stop_b_data['stop_id'], stop_b_data['arrival_td'])
            G.add_node(from_node); G.add_node(to_node)
            travel_time_delta = stop_b_data['arrival_td'] - stop_a_data['departure_td']
            travel_time_minutes = travel_time_delta.total_seconds() / 60.0
            attempted_pt_edges +=1
            if travel_time_minutes >= 0:
                non_negative_pt_edge_attempts +=1
                edge_existed_before = G.has_edge(from_node, to_node)
                G.add_edge(from_node, to_node, 
                           weight=travel_time_minutes, 
                           trip_id=trip_id,
                           dep_time=from_node[1], # this is stop_a_data['departure_td']
                           arr_time=to_node[1]    # this is stop_b_data['arrival_td']
                          ) 
                if not edge_existed_before: actually_added_new_pt_edges += 1
    print("Finished building PT edges.")
else:
    print("No PT trips to process for graph building.")
                
# print(f"\n Double Checking Our Data")
# print(f"Total PT segments processed (attempted_pt_edges): {attempted_pt_edges}")
# print(f"PT segments with non-negative travel time (non_negative_pt_edge_attempts): {non_negative_pt_edge_attempts}")
# print(f"Unique PT edges added to graph (actually_added_new_pt_edges): {actually_added_new_pt_edges}")
# print(f"Nodes after PT processing: {len(G.nodes())}") 
# print(f"Edges after PT processing (G.edges()): {len(G.edges())}") 

# Building walking edges 
node_list_for_walking = list(G.nodes()) 

if node_list_for_walking and stop_to_stop_list: 
    print(f"Building walking edges. Nodes available: {len(node_list_for_walking)}, Walk definitions: {len(stop_to_stop_list)}")
    for walk_info_row in stop_to_stop_list:
        try:
            # Handle Spark Row or dict
            stop_a_id = walk_info_row['stop_id_a'] if isinstance(walk_info_row, dict) else walk_info_row.stop_id_a
            stop_b_id = walk_info_row['stop_id_b'] if isinstance(walk_info_row, dict) else walk_info_row.stop_id_b
            distance_meters = float(walk_info_row['distance'] if isinstance(walk_info_row, dict) else walk_info_row.distance) 
        except (TypeError, KeyError, ValueError, AttributeError) as e:
            # print(f"Skipping walk_info_row due to data issue: {walk_info_row}, Error: {e}")
            continue

        if distance_meters < 0: continue # Distance cannot be negative
            
        walking_speed_mpm = WALKING_SPEED_MPS * 60
        if walking_speed_mpm == 0: continue
        
        walking_time_min = distance_meters / walking_speed_mpm # Time in minutes
        
        if walking_time_min < 0 : continue # Time cannot be negative

        for source_node_walk_start in node_list_for_walking: 
            if source_node_walk_start[0] == stop_a_id:
                walk_departure_time = source_node_walk_start[1]
                walk_arrival_time_at_b = walk_departure_time + timedelta(minutes=walking_time_min)
                destination_node_for_walk = (stop_b_id, walk_arrival_time_at_b)

                walking_edge_addition_attempts +=1
                G.add_edge(source_node_walk_start,
                           destination_node_for_walk,
                           weight=walking_time_min,
                           trip_id='WALK',
                           dep_time=walk_departure_time,   
                           arr_time=walk_arrival_time_at_b,
                           distance_m=distance_meters)      
                                                    
    print("Finished building walking edges.")
else:
    if not node_list_for_walking: print("No nodes from PT to start walking from for walking edge creation.")
    if not stop_to_stop_list: print("Warning: 'stop_to_stop_list' is empty. No walking definitions to process.")

print(f"\n Double Checking Our Data")
print(f"Total walking edge addition attempts: {walking_edge_addition_attempts}") 
print(f"Total Nodes in Graph: {len(G.nodes())}")
print(f"Total Edges in Graph: {len(G.edges())}")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # III. Predictive Model

# %%
# import libraries
from pyspark.sql import functions as F
from pyspark.sql.functions import unix_timestamp, col, lit
from pyspark.sql.functions import hour, percentile_approx
from pyspark.sql.functions import to_timestamp, dayofweek, dayofyear
from pyspark.sql.functions import regexp_extract
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import regexp_extract, min as Fmin, count as Fcount
from pyspark.sql.functions import sin, cos, lit, col
import math
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

# %% [markdown]
# ## Historical delay data

# %%
# historical data
hist_ist = (
    istdaten
      .filter(
          (col("operating_day") >= lit(historical_start_date)) &
          (col("operating_day") <= lit(historical_end_date)))
      .withColumn("arr_delay_s", unix_timestamp("arr_actual") - unix_timestamp("arr_time"))
      .withColumn("dep_delay_s", unix_timestamp("dep_actual") - unix_timestamp("dep_time"))
      .filter((col("arr_delay_s") > -120) & (col("arr_delay_s") < 3600))
      .filter((col("dep_delay_s") > -120) & (col("dep_delay_s") < 3600))
      .filter((hour(col("dep_actual")) >= lit(7)) & (hour(col("dep_actual")) < lit(18)))
      .cache())

# %%
# check the time range filter
hist_ist.select(hour(col("dep_actual")).alias("h")).distinct().orderBy("h").show()

# %%
# sanity check, 1% of data sample
sample = hist_ist.sample(fraction=0.01, seed=42)

print("Sample size:", sample.count())

# compute a few approximate quantiles
arr_q = sample.approxQuantile("arr_delay_s", [0.1, 0.5, 0.9], 0.01)
dep_q = sample.approxQuantile("dep_delay_s", [0.1, 0.5, 0.9], 0.01)
print(f"arr_delay_s quantiles (10%,50%,90%): {arr_q}")
print(f"dep_delay_s quantiles (10%,50%,90%): {dep_q}")

sample.select("operating_day","arr_delay_s","dep_delay_s").show(5, truncate=False)

# %%
# hourly delays percentile
hourly_stats = (
    hist_ist
      .withColumn("dep_hour", hour(col("dep_actual")))
      .groupBy("dep_hour")
      .agg(
        percentile_approx(col("arr_delay_s"), [0.5,0.75,0.9,0.95]).alias("arr_qs"),
        percentile_approx(col("dep_delay_s"), [0.5,0.75,0.9,0.95]).alias("dep_qs"),
        F.count("*").alias("n_trips"))
      .orderBy("dep_hour")
      .cache())

# trigger 
#hourly_stats.head(1)

# %%
#sanity check
# are all 24 hours present?
distinct_hours = hourly_stats.select("dep_hour").distinct().orderBy("dep_hour")
print("Hours covered:", [r.dep_hour for r in distinct_hours.collect()])
hourly_stats.show(24, truncate=False)

# %% [markdown]
# ## Feature engineering

# %%
# time features
hist_feat = (
    hist_ist
        .withColumn("sched_dep_hour", hour(to_timestamp(col("dep_time"), "yyyy-MM-dd HH:mm:ss")))
        .withColumn("act_dep_hour", hour(to_timestamp(col("dep_actual"), "yyyy-MM-dd HH:mm:ss")))
        .withColumn("dow",((dayofweek(col("operating_day")) + 5) % 7) + 1) #days of week
        .withColumn("day_of_year",dayofyear(col("operating_day")))
        .cache())

#trigger (for cache)
#hist_feat.head(1)

# %%
#sanity check
sample_feat = hist_feat.sample(fraction=0.01, seed=42)
sample_feat.printSchema()

print("\nSample rows with time features:")
sample_feat.select("dep_time","dep_actual","sched_dep_hour","act_dep_hour","operating_day","dow").limit(5).show(truncate=False)

hours = sorted(r.sched_dep_hour for r in sample_feat.select("sched_dep_hour").distinct().collect())
days  = sorted(r.dow for r in sample_feat.select("dow").distinct().collect())
print(f"\nsched_dep_hour values in sample: {hours}")
print(f"dow values in sample: {days}")


# %%
# take interesting column
hist_fact = hist_feat.select(
    "bpuic",
    "trip_id",
    "arr_time",
    "dep_time",
    "arr_delay_s",
    "sched_dep_hour",
    "act_dep_hour",
    "dow",
    "day_of_year").cache()

# %%
# stops_geo: bpuic, name, coords
stops_geo = (spark.table("iceberg.sbb.stops")
         .withColumn("bpuic", regexp_extract("stop_id", "(\\d+)", 1).cast("int"))
         .select("bpuic", "stop_name", "stop_lat", "stop_lon"))

# trip and route
trips_df  = spark.table("iceberg.sbb.trips").select("trip_id", "route_id")
routes_df = spark.table("iceberg.sbb.routes").select("route_id", "route_desc", "route_type")

# %%
hist_route_station = (
    hist_fact
      .join(stops_geo, on="bpuic", how="left")
      .join(trips_df, on="trip_id", how="left")
      .join(routes_df, on="route_id", how="left")
      .withColumn("scheduled_tt", unix_timestamp("arr_time") - unix_timestamp("dep_time"))
      .withColumn("delay", col("arr_delay_s"))
      .cache())

hist_route_station.createOrReplaceTempView("hist_route_station")

# %%
# sanity check
hist_route_station.printSchema()

# %%
# transfers per origin stop
transfers = (
    spark.table("iceberg.sbb.transfers")
      .withColumn("from_bpuic", regexp_extract("from_stop_id", "(\\d+)", 1).cast("int"))
      .withColumn("to_bpuic", regexp_extract("to_stop_id", "(\\d+)", 1).cast("int"))
      .select("from_bpuic", "to_bpuic", "min_transfer_time"))

transfer_stats = (
    transfers
        .groupBy("from_bpuic")
        .agg(
            Fmin("min_transfer_time").alias("min_transfer_time"),
            Fcount("to_bpuic").alias("transfer_degree")))

# join transfer
feat_with_transfer = (
    hist_route_station
      .join(transfer_stats, hist_route_station.bpuic == transfer_stats.from_bpuic, how="left")
      .drop("from_bpuic")
      .na.fill({"min_transfer_time": 2, "transfer_degree": 0}))

# weather
feat_with_weather = (
    feat_with_transfer
        .withColumn("temperature", lit(0.0))
        .withColumn("precipitation", lit(0.0))
        .withColumn("wind_speed", lit(0.0))
        .withColumn("hour_sin", sin(2 * lit(math.pi) * col("sched_dep_hour") / lit(24)))
        .withColumn("hour_cos", cos(2 * lit(math.pi) * col("sched_dep_hour") / lit(24)))
        .withColumn("doy_sin", sin(2 * lit(math.pi) * col("day_of_year") / lit(365)))
        .withColumn("doy_cos", cos(2 * lit(math.pi) * col("day_of_year") / lit(365))))

# final features col
final_features = (
    feat_with_weather
      .select(
         "trip_id", "bpuic", "route_id", "route_desc", "route_type",
         "scheduled_tt", "delay", "sched_dep_hour", "act_dep_hour", "dow", 
         "day_of_year", "hour_sin", "hour_cos", "doy_sin", "doy_cos", 
         "stop_lat", "stop_lon", "transfer_degree", "min_transfer_time",
         "temperature", "precipitation", "wind_speed").dropDuplicates())

final_features.createOrReplaceTempView("segment_features")

# %%
final_features.printSchema()

# %%
# baseline quantiles for route, bpuic, sched_dep_hour
baseline_qs = (
    final_features
      .groupBy("route_id", "bpuic", "sched_dep_hour")
      .agg(
          percentile_approx("delay", 0.50).alias("q50"),
          percentile_approx("delay", 0.75).alias("q75"),
          percentile_approx("delay", 0.90).alias("q90"),
          percentile_approx("delay", 0.95).alias("q95"),
          F.count("*").alias("n_obs"))
      .cache())

baseline_qs.createOrReplaceTempView("baseline_quantiles")

# %%
baseline_qs.printSchema()

# %% [markdown]
# ## Hybrid Delay Model

# %%
from hashlib import sha256

TRAINING = True

cleaned_region_names = sorted(["".join(x.lower().split()) for x in region_names])
joined_region_names = "".join(cleaned_region_names)
region_hash = sha256(joined_region_names.encode('utf-8')).hexdigest()
model_hdfs_path = f"{hadoopFS}/user/com-490/group/{GroupName}/{region_hash}/best_model"

print(f"Attempting to load the RandomForest PipelineModel from: {model_path_rf_to_load}")

try:
    # Load the PipelineModel
    resid_model = PipelineModel.load(model_path_rf_to_load)
    print(f"PipelineModel loaded successfully from {model_path_rf_to_load}")
    TRAINING = False
except Exception as e:
    print(f"Could not find a PipelineModel to load for the regions {region_names} at {model_path_rf_to_load}")
    print("Retraining the model...")


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml            import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types     import DoubleType

# %%
# prepare hybrid dataframe
# join in your baseline quantiles (q50, q75, ecc) that you computed earlier
data = final_features.join(
    baseline_qs.select("route_id","bpuic","sched_dep_hour","q50","q75","q90","q95"),
    on=["route_id","bpuic","sched_dep_hour"], how="left").withColumn("resid_label", col("delay") - col("q50"))

# %%
# split the dataset
train, valid, test = data.randomSplit([0.7,0.15,0.15], seed=42)

# %%
# sample test, take a small 5% sample of train to verify pipeline
# train_small = train.sample(fraction=0.05, seed=42)

# feature_cols = [
#     "scheduled_tt","sched_dep_hour","act_dep_hour",
#     "dow","day_of_year","hour_sin","hour_cos","doy_sin","doy_cos",
#     "stop_lat","stop_lon","transfer_degree","min_transfer_time",
#     "temperature","precipitation","wind_speed",
#     "q50"]  # include baseline median

# assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# rf_small = RandomForestRegressor(
#     labelCol="resid_label", featuresCol="features",
#     numTrees=10, maxDepth=5, seed=42)

# pipeline_small = Pipeline(stages=[assembler, rf_small])

# if TRAINING:
#     # fit and predict on the small sample
#     model_small = pipeline_small.fit(train_small)
#     preds_small = model_small.transform(valid).withColumn("pred_delay", col("q50")+col("prediction"))
# else:
#     print("Skipping Training the model")

# %%
# if TRAINING:
#     preds_small.select("delay","pred_delay").show(5)

# %%
if TRAINING:
    # train on full dataset
    rf = RandomForestRegressor(
        labelCol="resid_label", featuresCol="features",
        numTrees=100, maxDepth=10, seed=42)

    pipeline = Pipeline(stages=[assembler, rf])
    resid_model = pipeline.fit(train)
    print("train residual model complete")
else:
    print("Skipping Training the model")

# %% [markdown]
# ## Inital Validation
# Here we calculate the performance of our model before doing hyperparameter tuning. Since we are doing regression, our performance metric is Root Mean Squared Error (RMSE).

# %%
if TRAINING:
    # calibrate quatiles offset on validation
    from pyspark.ml.evaluation import RegressionEvaluator
    # point predictions + compute residuals on val
    vp = resid_model.transform(valid) \
          .withColumn("pred_delay", col("q50") + col("prediction")) \
          .withColumn("residual", col("delay") - col("pred_delay"))

    # approximate desired percentiles of residual
    qs = [0.5, 0.75, 0.9, 0.95]
    deltas = vp.stat.approxQuantile("residual", qs, 0.001)
    delta_by_q = dict(zip(qs, deltas))
    print("Residual shifts:", delta_by_q)

    # build UDFs to shift point forecast for each quantile
    def shift_udf(delta):
        return udf(lambda p: float(p + delta), DoubleType())

    shifters = {q: shift_udf(delta_by_q[q]) for q in qs}

    rmse_results = {}
    for q_value, shifter_udf in shifters.items():
        calibrated_pred_col_name = f"pred_{int(q_value * 100)}" # e.g., "pred_50", "pred_75"

        # Apply the shifter UDF to the 'pred_delay' column
        vp = vp.withColumn(calibrated_pred_col_name, shifter_udf(col("pred_delay")))

        # Initialize the evaluator for this specific calibrated prediction
        evaluator = RegressionEvaluator(
            labelCol="delay",  # The actual true delay
            predictionCol=calibrated_pred_col_name,
            metricName="rmse"
        )

        # Compute RMSE
        rmse = evaluator.evaluate(vp)
        rmse_results[calibrated_pred_col_name] = rmse
        print(f"RMSE for {calibrated_pred_col_name} against actual delay: {rmse}")

    # Display the calibrated predictions and actual delay for a few rows
    vp.select("delay", "pred_delay", "pred_50", "pred_75", "pred_90", "pred_95").show(5, truncate=False)

    # Print all RMSE results
    print("\nSummary of RMSE for calibrated quantile predictions:")
    for pred_col, rmse_val in rmse_results.items():
        print(f" - {pred_col}: {rmse_val}")
else:
    print("Skipping Validation step since TRAINING is false")

    

# %% [markdown]
# ### Hyperparameter Tuning
#
# Now that we can train the model, we conduct some hyperparamter tuning experiments to further improve the model. We will
# use the train and test set to perform the tuning and then evaluate the final model on the validation set.

# %%
if TRAINING:
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml import Pipeline
    from pyspark.sql import functions as F
    import numpy as np # ensure numpy is imported for np.bool_ if needed by older pyspark versions
    np.bool = np.bool_ # For compatibility if older pyspark version uses it.


    # We define the grid of parameters that we want to optimize over
    paramGrid_rf = (ParamGridBuilder()
                    .addGrid(rf.numTrees, [20, 50, 100])      # Number of trees
                    .addGrid(rf.maxDepth, [5, 10, 15])        # Maximum depth of the trees
                    .addGrid(rf.maxBins, [32, 64])            # Max number of bins for discretizing 
                    .build())

    print("RandomForestRegressor, ParamGrid, and Pipeline defined.")
    print(f"RandomForest labelCol: {rf.getLabelCol()}, featuresCol: {rf.getFeaturesCol()}")
    print(f"Number of models to train in CrossValidator for RF: {len(paramGrid_rf)}")

    evaluator_rf = RegressionEvaluator(labelCol="resid_label", predictionCol="prediction", metricName="rmse")

    cv_rf = CrossValidator(estimator=pipeline_rf,
                           estimatorParamMaps=paramGrid_rf,
                           evaluator=evaluator_rf,
                           numFolds=3,  # Use 3 folds for cross-validation. Adjust as needed.
                           seed=42,
                           parallelism=4) # Number of models to train in parallel

    print(f"Evaluator metric for RF: {evaluator_rf.getMetricName()}")
    print(f"CrossValidator numFolds for RF: {cv_rf.getNumFolds()}")
    print("Starting CrossValidator model fitting for RandomForestRegressor on the training data...")
    cvModel_rf = cv_rf.fit(train)
    print("CrossValidator model fitting for RandomForestRegressor completed.")
    resid_model = cvModel_rf.bestModel
else:
    print("Skipping Cross Validation step since TRAINING is false")

# %%
if TRAINING:
    # calibrate quatiles offset on validation

    # point predictions + compute residuals on val
    vp = resid_model.transform(valid) \
          .withColumn("pred_delay", col("q50") + col("prediction")) \
          .withColumn("residual", col("delay") - col("pred_delay"))

    # approximate desired percentiles of residual
    qs = [0.5, 0.75, 0.9, 0.95]
    deltas = vp.stat.approxQuantile("residual", qs, 0.001)
    delta_by_q = dict(zip(qs, deltas))
    print("Residual shifts:", delta_by_q)

    # build UDFs to shift point forecast for each quantile
    def shift_udf(delta):
        return udf(lambda p: float(p + delta), DoubleType())

    shifters = {q: shift_udf(delta_by_q[q]) for q in qs}

    rmse_results = {}

    for q_value, shifter_udf in shifters.items():
        calibrated_pred_col_name = f"pred_{int(q_value * 100)}" # e.g., "pred_50", "pred_75"

        # Apply the shifter UDF to the 'pred_delay' column
        vp = vp.withColumn(calibrated_pred_col_name, shifter_udf(col("pred_delay")))

        # Initialize the evaluator for this specific calibrated prediction
        evaluator = RegressionEvaluator(
            labelCol="delay",  # The actual true delay
            predictionCol=calibrated_pred_col_name,
            metricName="rmse"
        )

        # Compute RMSE
        rmse = evaluator.evaluate(vp)
        rmse_results[calibrated_pred_col_name] = rmse
        print(f"RMSE for {calibrated_pred_col_name} against actual delay: {rmse}")

    # Display the calibrated predictions and actual delay for a few rows
    vp.select("delay", "pred_delay", "pred_50", "pred_75", "pred_90", "pred_95").show(5, truncate=False)

    # Print all RMSE results
    print("\nSummary of RMSE for calibrated quantile predictions:")
    for pred_col, rmse_val in rmse_results.items():
        print(f" - {pred_col}: {rmse_val}")
else:
    print("Skipping validation of best model since TRAINING is false")
 


# %%
if TRAINING:
  # apply to test set and show final predictions

  tp = resid_model.transform(test) \
       .withColumn("pred_delay", col("q50") + col("prediction"))

  rmse_results = {}
  for q in qs:
    tp = tp.withColumn(f"pred_{int(q*100)}", shifters[q](col("pred_delay")))

  for q_value, shifter_udf in shifters.items():
    calibrated_pred_col_name = f"pred_{int(q_value * 100)}" # e.g., "pred_50", "pred_75"

    # Initialize the evaluator for this specific calibrated prediction
    evaluator = RegressionEvaluator(
        labelCol="delay",  # The actual true delay
        predictionCol=calibrated_pred_col_name,
        metricName="rmse"
    )

    # Compute RMSE
    rmse = evaluator.evaluate(tp)
    rmse_results[calibrated_pred_col_name] = rmse
    print(f"RMSE for {calibrated_pred_col_name} against actual delay: {rmse}")

  # Print all RMSE results
  print("\nSummary of RMSE for calibrated quantile predictions:")
  for pred_col, rmse_val in rmse_results.items():
      print(f" - {pred_col}: {rmse_val}")

  tp.select("delay","pred_delay","pred_50","pred_75","pred_90","pred_95") \
    .show(5, truncate=False)
else:
  print("Skipping testing of best model since TRAINING is false")

# %%
from hashlib import sha256

# Saving the best model
if TRAINING:
    cleaned_region_names = sorted(["".join(x.lower().split()) for x in region_names])
    joined_region_names = "".join(cleaned_region_names)
    region_hash = sha256(joined_region_names.encode('utf-8')).hexdigest()

    model_hdfs_path = f"{hadoopFS}/user/com-490/group/{GroupName}/{region_hash}/best_model"
    print(f"Attempting to save the trained RandomForest PipelineModel to: {model_path_rf}")

    try:
        # To overwrite if the model path already exists
        bestPipelineModel_rf.write().overwrite().save(model_path_rf)
        print(f"PipelineModel saved successfully to {model_path_rf}")
    except Exception as e:
        print(f"Error saving model: {e}")

    


# %%


# %%
'''#assemble your features
feature_cols = [
    "scheduled_tt", "sched_dep_hour", "act_dep_hour",
    "dow", "day_of_year", "hour_sin", "hour_cos",
    "doy_sin", "doy_cos", "stop_lat", "stop_lon",
    "transfer_degree", "min_transfer_time",
    "temperature", "precipitation", "wind_speed"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

#define the model (tune numTrees and maxDepth)
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="delay",
    predictionCol="prediction",
    numTrees=100,
    maxDepth=10,
    seed=42)

# build the pipeline and fit the model
rf_pipeline = Pipeline(stages=[assembler, rf])
rf_model = rf_pipeline.fit(train_df)
print("RandomForest training complete")'''

# %%

# %%

# %%

# %%


# %% [markdown]
# ## Model validation

# %%

# %%

# %%

# %%

# %%

# %%


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # IV. Route Planning Algorithm
# %%
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date as DateObject 
from heapq import heappush, heappop 
import math
from collections import defaultdict
from IPython.display import display, clear_output
import ipywidgets as widgets
import plotly.express as px
import plotly.graph_objects as go

class RobustJourneyPlanner:
    def __init__(self, graph, delay_model=None, max_walking_distance=500, walking_speed=50/60):
        self.graph = graph
        self.delay_model = delay_model
        self.max_walking_distance = max_walking_distance
        self.walking_speed_mps = walking_speed

    def get_stop_info(self, stop_df): 
        if stop_df is None: 
            print("Warning: get_stop_info received None for stop_df.")
            return {}
        try:
            required_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
            actual_cols = stop_df.columns
            if not ("stop_id" in actual_cols and "stop_name" in actual_cols):
                print("Warning: stop_df is missing 'stop_id' or 'stop_name'")
                return {}
            if not all(c in actual_cols for c in required_cols):
                print(f"Warning: stop_df missing some geo/detail columns. Using available: {required_cols}. Found: {actual_cols}")
                cols_to_select = [c for c in required_cols if c in actual_cols]
                if not ("stop_id" in cols_to_select and "stop_name" in cols_to_select):
                     print("Error: Critical 'stop_id' or 'stop_name' still missing after column selection.")
                     return {}
                stops_pd = stop_df.select(*cols_to_select).toPandas()
            else:
                stops_pd = stop_df.select(*required_cols).toPandas()
            stops_pd.dropna(subset=['stop_id'], inplace=True)
            return stops_pd.set_index("stop_id").to_dict('index')
        except Exception as e: 
            print(f"Error in get_stop_info: {e}"); 
            import traceback; traceback.print_exc(); 
            return {}

    def _calculate_walking_time(self, lat1, lon1, lat2, lon2):
        R = 6371000 #very detailed walking time LOL 
        if None in [lat1, lon1, lat2, lon2]: return float('inf'), float('inf')
        try: lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
        except ValueError: return float('inf'), float('inf')
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1); d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist = R * c
        time_s = float('inf') if self.walking_speed_mps == 0 else dist / self.walking_speed_mps
        return dist, time_s

    # DUMMY VALUES CAUSE MODEL NOT IMPLEMENTED YET 
    def _get_delay_distribution(self, trip_id, dep_stop_id, dep_time_td, day_of_week=None):
        if self.delay_model is None: return {"q50": 0, "q75": 0, "q90": 0, "q95": 0}
        return {"q50": 0, "q75": 0, "q90": 0, "q95": 0}


    def _has_significant_physical_stop_loop(self, path_nodes_list):
        if not path_nodes_list or len(path_nodes_list) < 3:
            return False
        stop_ids_in_path = [node[0] for node in path_nodes_list]
        for i in range(len(stop_ids_in_path) - 1):
            current_stop = stop_ids_in_path[i]
            for j in range(i + 2, len(stop_ids_in_path)): 
                if stop_ids_in_path[j] == current_stop:
                    is_actual_loop = False
                    for k_loop_check in range(i + 1, j):
                        if stop_ids_in_path[k_loop_check] != current_stop:
                            is_actual_loop = True
                            break
                    if is_actual_loop:
                        # print(f"DEBUG Loop Path segment ...{path_nodes_list[i]}...{path_nodes_list[j]}...")
                        return True
        return False

    def find_robust_paths(self, source_stop_id, target_stop_id,
                         departure_time_td=None, 
                         latest_arrival_constraint_td=None, 
                         confidence_level=0.75,
                         max_options=3):
        if departure_time_td is None and latest_arrival_constraint_td is None:
            print("Error: At least departure time or arrival constraint must be specified.")
            return []
        
        effective_earliest_departure_td = departure_time_td
        if departure_time_td is None and latest_arrival_constraint_td is not None:
            print("Warning: Only arrival constraint specified. Using a wide departure window for forward search (suboptimal).")
            effective_earliest_departure_td = latest_arrival_constraint_td - timedelta(hours=4) 
            if effective_earliest_departure_td < timedelta(0): effective_earliest_departure_td = timedelta(0)
        elif departure_time_td is None: 
            effective_earliest_departure_td = timedelta(0)

        confidence_map = {0.5: "q50", 0.75: "q75", 0.9: "q90", 0.95: "q95"}
        quantile_key = confidence_map.get(confidence_level, "q75")

        found_options = self._forward_dijkstra_for_options(
            source_stop_id, target_stop_id, effective_earliest_departure_td,
            latest_arrival_constraint_td, quantile_key, max_options
        )
        
        # Filtering again by original departure_time_td if it was specified and we used a wider hack window for "arrive by"
        if departure_time_td is not None: 
            found_options = [opt for opt in found_options if opt["departure_time"] >= departure_time_td]
        
        # Sorting final selected options for presentation
        found_options.sort(key=lambda x: (x["arrival_time"], x["departure_time"], x["transfers"]))
        return found_options[:max_options] # Making sure we adhere to max_options

    def _forward_dijkstra_for_options(self, source_stop_id, target_stop_id,
                                     earliest_departure_td, latest_arrival_constraint_td,
                                     quantile_key, num_options_desired):
        start_event_nodes = []
        for node in self.graph.nodes(): 
            if node[0] == source_stop_id and isinstance(node[1], timedelta) and \
               node[1] >= earliest_departure_td:
                try: 
                    _ = next(self.graph.successors(node)) 
                    if node not in start_event_nodes:
                        start_event_nodes.append(node)
                except StopIteration: pass 
        
        if not start_event_nodes: return []
        start_event_nodes.sort(key=lambda x: x[1])

        queue = []
        for start_node in start_event_nodes:
            heappush(queue, (start_node[1].total_seconds(), 0.0, start_node, [start_node]))

        visited_earliest_arrival = {}
        completed_options_raw = [] 
        
        max_iterations = 750000; iterations = 0 

        # Heuristic: How many times we've hit the target. Stop if we have plenty of raw options.
        target_hit_count = 0

        while queue and target_hit_count < num_options_desired * 5 and iterations < max_iterations: 
            iterations += 1 
            current_arrival_sec, current_total_walk_dist_m, current_event_node, current_path_nodes = heappop(queue)
            current_arrival_td = timedelta(seconds=current_arrival_sec)
            current_stop_id, _ = current_event_node 

            # Pruning based on visited states
            if current_event_node in visited_earliest_arrival and \
               current_arrival_td >= visited_earliest_arrival[current_event_node]:
                continue
            visited_earliest_arrival[current_event_node] = current_arrival_td

            # If current node is the target
            if current_stop_id == target_stop_id:
                if latest_arrival_constraint_td and current_arrival_td > latest_arrival_constraint_td:
                    continue
                actual_departure_node = current_path_nodes[0]; actual_departure_time_td = actual_departure_node[1]
                total_travel_time_minutes = (current_arrival_td - actual_departure_time_td).total_seconds() / 60.0
                if total_travel_time_minutes < -1e-7: continue 

                # Checking for loops before adding to raw options
                if self._has_significant_physical_stop_loop(current_path_nodes):
                    # print(f"DEBUG: Pruning loopy path to target: {[n[0] for n in current_path_nodes]}")
                    continue

                option_details = {"departure_stop": source_stop_id, "departure_time": actual_departure_time_td,
                                  "arrival_stop": target_stop_id, "arrival_time": current_arrival_td,
                                  "travel_time_minutes": total_travel_time_minutes,
                                  "walking_distance_meters": current_total_walk_dist_m,
                                  "transfers": self._count_transfers(current_path_nodes),
                                  "path": current_path_nodes, # List of (stop_id, time_td) tuples
                                  "confidence_level": self._quantile_key_to_confidence(quantile_key)}
                completed_options_raw.append(option_details)
                target_hit_count +=1
            
            # Heuristic pruning if current path is already much later than constraint
            if latest_arrival_constraint_td and current_arrival_td > latest_arrival_constraint_td + timedelta(minutes=60) : 
                pass 

            for successor_event_node in self.graph.successors(current_event_node):
                edge_data = self.graph.get_edge_data(current_event_node, successor_event_node)
                if not edge_data: continue
                
                segment_departure_td = current_event_node[1] 
                segment_scheduled_arrival_td = successor_event_node[1] 
                trip_id_seg = edge_data.get("trip_id", "")
                
                if segment_scheduled_arrival_td < segment_departure_td: continue
                
                new_path_nodes_candidate = current_path_nodes + [successor_event_node]

                # More robust simple loop check (A->B->A) for PT before pushing to heap
                if trip_id_seg != "WALK" and len(new_path_nodes_candidate) >= 3:
                    if new_path_nodes_candidate[-1][0] == new_path_nodes_candidate[-3][0]: # Successor stop is same as grandparent stop
                        continue 
                
                delay_on_segment_seconds = 0
                if trip_id_seg != "WALK":
                    delay_dist = self._get_delay_distribution(trip_id_seg, current_event_node[0], segment_departure_td)
                    delay_on_segment_seconds = delay_dist.get(quantile_key, 0)
                
                actual_arrival_at_successor_td = segment_scheduled_arrival_td + timedelta(seconds=delay_on_segment_seconds)

                if latest_arrival_constraint_td and actual_arrival_at_successor_td > latest_arrival_constraint_td:
                    continue

                new_total_walk_dist_m = current_total_walk_dist_m
                if trip_id_seg == "WALK": 
                    new_total_walk_dist_m += edge_data.get("distance_m", 0) 
                if new_total_walk_dist_m > self.max_walking_distance: continue
                
                # Pruning for already visited successor state
                if successor_event_node in visited_earliest_arrival and \
                   actual_arrival_at_successor_td >= visited_earliest_arrival[successor_event_node]:
                    continue
                
                heappush(queue, (actual_arrival_at_successor_td.total_seconds(),
                                     new_total_walk_dist_m, successor_event_node, new_path_nodes_candidate))
        
        if iterations >= max_iterations: 
            print(f"Warning: Dijkstra search reached max iterations ({max_iterations}).")
        
        # Post processing logic for diversity and no repetition
        if not completed_options_raw: return []

        # 1. Filtering out paths with significant loops from the raw options
        non_loopy_options = [opt for opt in completed_options_raw if not self._has_significant_physical_stop_loop(opt["path"])]
        if not non_loopy_options: 
            print("Warning: All found paths had significant loops based on current check.")
            # Fallback to raw options if filtering removes everything, then sort and pick
            non_loopy_options = completed_options_raw 
            if not non_loopy_options: return [] # Still nothing

        # 2. Sorting by a primary metric like arrival time, then departure, then transfers
        non_loopy_options.sort(key=lambda x: (x["arrival_time"], x["departure_time"], x["transfers"]))
        
        final_selected_options = []
        # Heuristic for diversity: we wanna pick options with different departure "buckets"
        # and different "main characteristics" (e.g., number of transfers, or key intermediate trips)
        
        selected_path_footprints = set() # for us to store a signature of selected paths

        for option in non_loopy_options:
            if len(final_selected_options) >= num_options_desired:
                break

            # Create a simple footprint for the path (e.g., departure time bucket, num_transfers, main trip ids)
            dep_bucket = int(option["departure_time"].total_seconds() / (15 * 60)) # 15-min departure buckets
            
            # Get dominant PT trip_ids (e.g., first 2 non-walk trip_ids)
            pt_trips_in_path = []
            for k_path in range(len(option["path"]) - 1):
                edge_data_fp = self.graph.get_edge_data(option["path"][k_path], option["path"][k_path+1])
                if edge_data_fp and edge_data_fp.get("trip_id") != "WALK":
                    pt_trips_in_path.append(edge_data_fp.get("trip_id"))
                if len(pt_trips_in_path) >= 2: break
            
            footprint = (dep_bucket, option["transfers"], tuple(pt_trips_in_path))

            if footprint not in selected_path_footprints:
                final_selected_options.append(option)
                selected_path_footprints.add(footprint)
        
        # If still not enough options, filling with the best remaining ones (that are not identical paths)
        if len(final_selected_options) < num_options_desired:
            existing_raw_paths_in_final = {tuple(opt['path']) for opt in final_selected_options}
            for option in non_loopy_options: # Iterate again through all non-loopy sorted options
                if len(final_selected_options) >= num_options_desired:
                    break
                if tuple(option['path']) not in existing_raw_paths_in_final:
                    final_selected_options.append(option)
                    existing_raw_paths_in_final.add(tuple(option['path'])) # trynna avoid duplicates so we use raw paths

        final_selected_options.sort(key=lambda x: (x["arrival_time"], x["departure_time"], x["transfers"]))
        return final_selected_options[:num_options_desired]


    def _count_transfers(self, path_nodes_list):
        if not path_nodes_list or len(path_nodes_list) <= 1: return 0
        transfers = 0; last_pt_trip_id = None
        for i in range(len(path_nodes_list) - 1):
            node_u, node_v = path_nodes_list[i], path_nodes_list[i+1]
            if not (isinstance(node_u, tuple) and len(node_u) == 2 and isinstance(node_v, tuple) and len(node_v) == 2): continue
            edge_data = self.graph.get_edge_data(node_u, node_v)
            if not edge_data: continue
            segment_trip_id = edge_data.get("trip_id")
            if segment_trip_id != "WALK" and segment_trip_id is not None: # Current segment is PT
                if last_pt_trip_id is not None and segment_trip_id != last_pt_trip_id:
                    transfers += 1 
                last_pt_trip_id = segment_trip_id
            # If segment_trip_id is "WALK", last_pt_trip_id remains unchanged,
            # so the next PT segment will compare against the PT trip before the walk.
        return transfers

    # UI stuff starts

    def _reconstruct_path(self, reverse_path, target_node): 
        return reverse_path 

    def _quantile_key_to_confidence(self, quantile_key):
        quantile_map = {"q50": 0.5, "q75": 0.75, "q90": 0.9, "q95": 0.95}
        return quantile_map.get(quantile_key, 0.75)

    def format_results(self, paths, stop_info):
        formatted_paths = []
        if not stop_info: print("Warning: stop_info not provided to format_results."); stop_info = {}
        for i, path_result_dict in enumerate(paths):
            path_detail_nodes = path_result_dict.get("path", []) 
            segments = []
            if not path_detail_nodes or len(path_detail_nodes) < 1:
                if len(path_detail_nodes) == 1: 
                     curr_stop_id, curr_time_td = path_detail_nodes[0]
                     curr_stop_name = stop_info.get(curr_stop_id, {}).get("stop_name", curr_stop_id)
                     curr_time_str = self._format_time(curr_time_td)
                     segments.append({"from_stop": curr_stop_name, "to_stop": curr_stop_name, "departure": curr_time_str, "arrival": curr_time_str, "duration_mins": 0.0, "type": "AT_STOP", "trip_id": "N/A"})
            else: 
                for j in range(len(path_detail_nodes) - 1):
                    curr_node, next_node = path_detail_nodes[j], path_detail_nodes[j+1]
                    curr_stop_id, curr_time_td = curr_node; next_stop_id, next_time_td = next_node
                    edge_data = self.graph.get_edge_data(curr_node, next_node) or {}
                    trip_id_seg = edge_data.get("trip_id", "UNKNOWN")
                    transport_type = "WALK" if trip_id_seg == "WALK" else "TRANSIT"
                    curr_stop_name = stop_info.get(curr_stop_id, {}).get("stop_name", curr_stop_id)
                    next_stop_name = stop_info.get(next_stop_id, {}).get("stop_name", next_stop_id)
                    curr_time_str = self._format_time(curr_time_td); next_time_str = self._format_time(next_time_td)
                    segment_duration_mins = (next_time_td - curr_time_td).total_seconds() / 60.0
                    segments.append({"from_stop": curr_stop_name, "to_stop": next_stop_name, "departure": curr_time_str, "arrival": next_time_str, "duration_mins": round(segment_duration_mins, 1), "type": transport_type, "trip_id": trip_id_seg})
            route_summary = {"route_id": i + 1, "departure": self._format_time(path_result_dict["departure_time"]), "arrival": self._format_time(path_result_dict["arrival_time"]), "travel_time_mins": round(path_result_dict.get("travel_time_minutes", 0), 1), "transfers": path_result_dict.get("transfers", 0), "walking_distance": round(path_result_dict.get("walking_distance_meters", 0)), "confidence": f"{int(path_result_dict.get('confidence_level', 0.75) * 100)}%", "segments": segments, "_raw_path_nodes": path_detail_nodes}
            formatted_paths.append(route_summary)
        return formatted_paths

    def _format_time(self, time_td):
        if not isinstance(time_td, timedelta): return "00:00:00"
        total_seconds = time_td.total_seconds(); sign = "-" if total_seconds < 0 else ""; total_seconds = abs(total_seconds)
        h = int(total_seconds // 3600); m = int((total_seconds % 3600) // 60); s = int(total_seconds % 60)
        return f"{sign}{h:02d}:{m:02d}:{s:02d}"

    def visualize_path(self, path_dict, stop_info):
        path_nodes_viz = path_dict.get("_raw_path_nodes");
        if not path_nodes_viz or not stop_info: print("Warning: Cannot visualize path."); return go.Figure()
        stop_map_data_viz = []; unique_stops_for_map_markers = {} 
        for node_viz in path_nodes_viz: 
            sid_viz = node_viz[0]; info_viz = stop_info.get(sid_viz)
            if info_viz and 'stop_lat' in info_viz and 'stop_lon' in info_viz:
                if sid_viz not in unique_stops_for_map_markers: 
                    stop_map_data_viz.append({"stop_id": sid_viz, "stop_name": info_viz.get("stop_name", sid_viz), "lat": info_viz["stop_lat"], "lon": info_viz["stop_lon"]})
                    unique_stops_for_map_markers[sid_viz] = True 
        if not stop_map_data_viz: print("Warning: No valid stops for visualization."); return go.Figure()
        df_map = pd.DataFrame(stop_map_data_viz);
        if df_map.empty: print("Warning: Map DataFrame empty."); return go.Figure()
        path_segments_for_map = []
        if len(path_nodes_viz) >= 2:
            for j in range(len(path_nodes_viz) - 1):
                curr_n_map, next_n_map = path_nodes_viz[j], path_nodes_viz[j+1]
                curr_sid_map, next_sid_map = curr_n_map[0], next_n_map[0]
                curr_geo, next_geo = stop_info.get(curr_sid_map), stop_info.get(next_sid_map)
                if curr_geo and 'stop_lat' in curr_geo and 'stop_lon' in curr_geo and \
                   next_geo and 'stop_lat' in next_geo and 'stop_lon' in next_geo:
                    edge_d_map = self.graph.get_edge_data(curr_n_map, next_n_map) or {}
                    tid_map = edge_d_map.get("trip_id", "UNKNOWN")
                    color = 'red' if tid_map == 'WALK' else 'blue'; seg_type = 'Walk' if tid_map == 'WALK' else 'Transit'
                    path_segments_for_map.append({'lat': [float(curr_geo["stop_lat"]), float(next_geo["stop_lat"])], 'lon': [float(curr_geo["stop_lon"]), float(next_geo["stop_lon"])], 'segment_type': seg_type, 'color': color})
        fig = px.scatter_mapbox(df_map, lat='lat', lon='lon', hover_name='stop_name', text=df_map['stop_name'].fillna('') if 'stop_name' in df_map.columns else None, size=[8]*len(df_map), zoom=11, center={"lat": df_map['lat'].mean(), "lon": df_map['lon'].mean()} if not df_map.empty else {"lat":46.52, "lon":6.63}, mapbox_style="carto-positron") 
        if 'stop_name' in df_map.columns: fig.update_traces(textposition='top right')
        for seg in path_segments_for_map:
            fig.add_trace(go.Scattermapbox(lat=seg['lat'], lon=seg['lon'], mode='lines', line=dict(width=4 if seg['segment_type']=='Transit' else 2, color=seg['color']), name=seg['segment_type'], hoverinfo='skip'))
        legend_names = set(); fig.for_each_trace(lambda trace: trace.update(showlegend=False) if (trace.name in legend_names) else legend_names.add(trace.name))
        fig.update_layout(title=f"Route: Dep @ {path_dict['departure']}, Arr @ {path_dict['arrival']}", legend_title_text='Segment Type', height=600, margin={"r":0,"t":50,"l":0,"b":0})
        return fig


def create_journey_planner_ui(planner, stops_df_for_ui): # stops_df_for_ui is Spark DF
    if stops_df_for_ui is None: return widgets.VBox([widgets.Label("Error: Stop data for UI not provided.")])
    
    stop_info_for_planner_methods = planner.get_stop_info(stops_df_for_ui) 

    try:
        stops_pd_ui = stops_df_for_ui.select("stop_id", "stop_name").toPandas().dropna(subset=['stop_name']).drop_duplicates(subset=['stop_name'])
        stop_names_map_ui = dict(zip(stops_pd_ui['stop_name'], stops_pd_ui['stop_id']))
        if not stop_names_map_ui: raise ValueError("No stop names for UI dropdowns.")
    except Exception as e:
        print(f"Error preparing UI stop names: {e}"); return widgets.VBox([widgets.Label("Error: UI stop names.")])
        
    dep_dropdown = widgets.Dropdown(options=sorted(stop_names_map_ui.keys()), description='From:', layout=widgets.Layout(width='350px'))
    arr_dropdown = widgets.Dropdown(options=sorted(stop_names_map_ui.keys()), description='To:', layout=widgets.Layout(width='350px'))
    date_picker_ui = widgets.DatePicker(description='Date:', value=datetime.now().date(), layout=widgets.Layout(width='220px'))
    dep_time_text = widgets.Text(description='Depart (HH:MM):', value='08:00', placeholder='HH:MM', layout=widgets.Layout(width='220px'))
    arr_time_text = widgets.Text(description='Arrive By (HH:MM):', value='', placeholder='HH:MM (optional)', layout=widgets.Layout(width='220px'))
    use_dep_time_cb = widgets.Checkbox(value=True, description='Use Dep Time', indent=False, layout=widgets.Layout(width='150px'))
    use_arr_time_cb = widgets.Checkbox(value=False, description='Use Arr Constraint', indent=False, layout=widgets.Layout(width='180px'))
    conf_slider = widgets.SelectionSlider(options=[('50%',0.5),('75%',0.75),('90%',0.9),('95%',0.95)],value=0.75,description='Confidence:',layout=widgets.Layout(width='300px'))
    max_opts_slider = widgets.IntSlider(value=3,min=1,max=10,step=1,description='Max Options:',layout=widgets.Layout(width='300px'))
    search_btn = widgets.Button(description='Search Options',button_style='success',layout=widgets.Layout(width='180px'))
    
    results_out = widgets.Output(); map_out = widgets.Output(); 
    _cached_formatted_paths = [] # Moved here to be in the broader scope for callbacks
    
    route_selection_dropdown = widgets.Dropdown(description='View Route on Map:', disabled=True, layout=widgets.Layout(width='450px'))

    def on_route_selected_for_map(change): # This callback needs access to _cached_formatted_paths
        map_out.clear_output(wait=True)
        selected_route_label_or_value = change.new 
        
        if not _cached_formatted_paths: return

        # The 'value' of the dropdown will be the index if options are (label, index)
        # The 'label' attribute will be the string label.
        # We set .options as list of (label, index) tuples, so change.new will be the index (the value part)
        selected_idx = selected_route_label_or_value 
        
        try:
            if isinstance(selected_idx, int) and 0 <= selected_idx < len(_cached_formatted_paths):
                selected_path_data = _cached_formatted_paths[selected_idx]
                with map_out: display(planner.visualize_path(selected_path_data, stop_info_for_planner_methods))
            # else if selected_idx is None (e.g. dropdown cleared) do nothing
        except Exception as e:
             with map_out: print(f"Error selecting route for map (idx: {selected_idx}): {e}")

    route_selection_dropdown.observe(on_route_selected_for_map, names='value') # Observe 'value'

    def on_search_clicked(b):
        nonlocal _cached_formatted_paths 
        results_out.clear_output(wait=True); map_out.clear_output(wait=True)
        route_selection_dropdown.options = [] 
        route_selection_dropdown.value = None # Reset value
        route_selection_dropdown.disabled = True
        _cached_formatted_paths = []

        with results_out:
            try:
                src_name=dep_dropdown.value; tgt_name=arr_dropdown.value
                if not src_name or not tgt_name: print("Select stops."); return
                src_id=stop_names_map_ui.get(src_name); tgt_id=stop_names_map_ui.get(tgt_name)
                if not src_id or not tgt_id: print("Invalid stops."); return
                dep_td_val, arr_constr_td_val = None, None; query_parts = [f"Search: {src_name} to {tgt_name}"]
                dep_time_specified_and_valid = False
                if use_dep_time_cb.value and dep_time_text.value.strip():
                    try:
                        h,m=map(int,dep_time_text.value.split(':')); 
                        if not (0<=h<=23 and 0<=m<=59): raise ValueError("Time component out of range.")
                        dep_td_val=timedelta(hours=h,minutes=m)
                        query_parts.append(f"depart at/after {planner._format_time(dep_td_val)}"); dep_time_specified_and_valid=True
                    except ValueError: print(f"Invalid Dep Time: '{dep_time_text.value}'. Use HH:MM."); return
                elif use_dep_time_cb.value and not dep_time_text.value.strip(): print("Dep time selected, but no time entered."); return
                arr_constraint_specified_and_valid = False
                if use_arr_time_cb.value and arr_time_text.value.strip():
                    try:
                        h_a,m_a=map(int,arr_time_text.value.split(':')); 
                        if not (0<=h_a<=23 and 0<=m_a<=59): raise ValueError("Time component out of range.")
                        arr_constr_td_val=timedelta(hours=h_a,minutes=m_a)
                        query_parts.append(f"arrive by {planner._format_time(arr_constr_td_val)}"); arr_constraint_specified_and_valid=True
                    except ValueError: print(f"Invalid Arr Time: '{arr_time_text.value}'. Use HH:MM."); return
                elif use_arr_time_cb.value and not arr_time_text.value.strip(): print("Arr constraint selected, but no time entered."); return
                if not (use_dep_time_cb.value and dep_time_specified_and_valid) and \
                   not (use_arr_time_cb.value and arr_constraint_specified_and_valid):
                    if not use_dep_time_cb.value and not use_arr_time_cb.value: print("Please specify & enable a time criteria.")
                    return 
                conf_val=conf_slider.value; max_opts_val=max_opts_slider.value
                query_parts.append(f"Conf: {int(conf_val*100)}%"); print(", ".join(query_parts))
                paths_list = planner.find_robust_paths(src_id, tgt_id, dep_td_val, arr_constr_td_val, conf_val, max_opts_val)
                if not paths_list: print("No trip options found."); return
                _cached_formatted_paths = planner.format_results(paths_list, stop_info_for_planner_methods)
                
                dropdown_options_list = []
                for route_idx, route in enumerate(_cached_formatted_paths): 
                    print(f"\n--- Option {route['route_id']} ---")
                    print(f"Dep: {route['departure']} Arr: {route['arrival']} (Travel: {route['travel_time_mins']:.1f} min)")
                    print(f"Transfers: {route['transfers']}, Walk: {route['walking_distance']}m, Conf: {route['confidence']}")
                    dropdown_label = f"Option {route['route_id']} (Dep: {route['departure']}, Arr: {route['arrival']})"
                    dropdown_options_list.append((dropdown_label, route_idx)) 
                    
                    # CORRECTED SEGMENT PRINTING LOGIC; had issues with no printing after 2 options
                    if route.get('segments'): 
                        print("  Segments:")
                        for seg in route['segments']:
                            print(f"    {seg['from_stop']} ({seg['departure']}) → {seg['to_stop']} ({seg['arrival']}) "
                                  f"[{seg['type']}, {seg['duration_mins']:.1f}m, Trip: {seg['trip_id'] if seg['type']=='TRANSIT' else 'N/A'}]")
                    elif route['departure'] == route['arrival'] and route.get('segments') and route['segments'][0]['type'] == "AT_STOP":
                         print(f"    {route['segments'][0]['from_stop']} ({route['departure']}) - At destination (no travel).")
                    elif route['departure'] == route['arrival']: # Fallback for source=target if segments is empty
                         print(f"    {stop_info_for_planner_methods.get(src_id,{}).get('stop_name',src_id)} ({route['departure']}) - At destination (no travel).")

                if _cached_formatted_paths:
                    route_selection_dropdown.options = dropdown_options_list
                    route_selection_dropdown.disabled = False
                    if dropdown_options_list: # If there are options, set a default value to trigger map
                        route_selection_dropdown.value = dropdown_options_list[0][1] # Setting first option by default
                    with map_out: display(planner.visualize_path(_cached_formatted_paths[0], stop_info_for_planner_methods))
                else: route_selection_dropdown.disabled = True
            except Exception as e: print(f"Error during search: {e}"); import traceback; traceback.print_exc()
    
    search_btn.on_click(on_search_clicked)
    time_input_controls = widgets.VBox([widgets.HBox([use_dep_time_cb, dep_time_text]), widgets.HBox([use_arr_time_cb, arr_time_text])])
    input_ui_layout = widgets.VBox([widgets.HBox([dep_dropdown, arr_dropdown]),date_picker_ui, time_input_controls, widgets.HBox([conf_slider, max_opts_slider]),search_btn, route_selection_dropdown])
    tabs_output = widgets.Tab([results_out, map_out]); tabs_output.set_title(0, 'Options'); tabs_output.set_title(1, 'Map')
    display(widgets.VBox([input_ui_layout, tabs_output]))


# %%
def implement_robust_journey_planner(graph, stops_df, delay_model=None):
    """
    Implement the robust journey planner.
    
    Args:
        graph: NetworkX DiGraph representing the time-dependent transport network
        stops_df: DataFrame with stop information
        delay_model: Model for predicting delays (can be None initially)
        
    Returns:
        RobustJourneyPlanner instance and interactive UI
    """
    # Creating the planner
    planner = RobustJourneyPlanner(
        graph=graph,
        delay_model=delay_model,
        max_walking_distance=MAX_WALKING_DISTANCE_METERS,
        # walking_speed=WALKING_SPEED_MPS, 
    )
    
    ui = create_journey_planner_ui(planner, stops_df)
    
    return planner, ui


# %%
planner, ui = implement_robust_journey_planner(G, stops_lausanne_rt)
display(ui)

# Takes about 30 seconds to initialize and start


# %%
#  SOME COMMENTS 

# so if super closer, maybe we can tell it to have a walking threshold 

# waiting times?
# walking optional ?

# I realized they arent the shortest paths, and IDK if thats me or the data
# but algo isnt YEN (k shortest paths)


# MINE : Depart Source 07:00 -> Arrive Target 08:30 (Total Travel Time: 90 mins) --> Arrival time at destination focus
# YEN B: Depart Source 07:30 -> Arrive Target 08:40 (Total Travel Time: 70 mins) --> Total travel time focus 

# Algo desc:
    # Dijkstra-based
    # forward search
    # k-earliest arrival paths
    # heuristic for path diversity and loop reduction
        # For ex: departure_time_threshold_for_diversity (dont leave at the same time)
                  # and _has_significant_physical_stop_loop


# later we'll incorporate a delay model

# currently using dummy delay values 
# specifically in this function

#  planner = RobustJourneyPlanner(
#         graph=graph,
#         delay_model=delay_model,
#         max_walking_distance=MAX_WALKING_DISTANCE_METERS,
#         # walking_speed=WALKING_SPEED_MPS, 
#         # delay_model = None --> default 
#         # can modify to add any other model calibrations if needed 
#     )

# def _get_delay_distribution(self, trip_id, dep_stop_id, dep_time_td, day_of_week=None):
#     if self.delay_model is None: return {"q50": 0, "q75": 0, "q90": 0, "q95": 0}
#     return {"q50": 0, "q75": 0, "q90": 0, "q95": 0}







# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # V. Validation
# %%

# %%

# %%


# %%
spark.stop()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # OTHER STUFF, CHECK WHAT TO KEEP

# %% [markdown]
# ---
# ⚠️ **Note**: all the data used in this homework is described in the [FINAL-PREVIEW](./final-preview.md) document, which can be found in this repository. The document describes the final project due for the end of this semester.
#
# For this notebook you are free to use the following tables, which can all be found under the _iceberg.com490_iceberg_ namespace shared by the class (you may use the sharedNS variable).
# - You can list the tables with the command `f"SHOW TABLES IN {sharedNS}"`.
# - You can see the details of each table with the command `f"DESCRIBE {sharedNS}.{table_name}"`.
#
# ---
# For your convenience we also define useful python variables:
#
# * _hadoop_fs_
#     * The HDFS server, in case you need it for hdfs, pandas or pyarrow commands.
# * _username_:
#     * Your user id (EPFL gaspar id), use it as your personal namespace for your private tables.
# * _sharedNS_:
#     * The namespace of the tables shared by the class. **DO NOT** modify or drop tables in this namespace, or drop the namespace.
# * _namespace_:
#     * Your personal namespace.

# %% [markdown]
# <div style="font-size: 100%" class="alert alert-block alert-warning">
#     <b>Fair cluster Usage:</b>
#     <br>
#     As there are many of you working with the cluster, we encourage you to prototype your queries on small data samples before running them on whole datasets. Do not hesitate to partion your tables, and LIMIT the output of your queries to a few rows to begin with. You are also free to test your queries using alternative solutions such as <i>DuckDB</i>.
#     <br><br>
#     You may lose your session if you remain idle for too long or if you interrupt a query. If that happens you will not lose your tables, but you may need to reconnect to the warehouse.
#     <br><br>
#     <b>Try to use as much SQL as possible and avoid using pandas operations.</b>
# </div>

# %%
import os
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable .*")

# %%
import base64 as b64
import json
import time
import re

def getUsername():
    payload = os.environ.get('EPFL_COM490_TOKEN').split('.')[1]
    payload=payload+'=' * (4 - len(payload) % 4)
    obj = json.loads(b64.urlsafe_b64decode(payload))
    if (time.time() > int(obj.get('exp')) - 3600):
        raise Exception('Your credentials have expired, please restart your Jupyter Hub server:'
                        'File>Hub Control Panel, Stop My Server, Start My Server.')
    time_left = int((obj.get('exp') - time.time())/3600)
    return obj.get('sub'), time_left


# %%
username, validity_h = getUsername()
hadoopFS = os.environ.get('HADOOP_FS')
namespace = 'iceberg.' + username
sharedNS = 'iceberg.com490_iceberg'

if not re.search('[A-Z][0-9]', groupName):
    raise Exception('Invalid group name {groupName}')

print(f"you are: {username}")
print(f"credentials validity: {validity_h} hours left.")
print(f"shared namespace is: {sharedNS}")
print(f"your namespace is: {namespace}")
print(f"your group is: {groupName}")

# %% [markdown]
# ---

# %%
import trino
from contextlib import closing
from urllib.parse import urlparse
from trino.dbapi import connect
from trino.auth import BasicAuthentication, JWTAuthentication

trinoAuth = JWTAuthentication(os.environ.get('EPFL_COM490_TOKEN'))
trinoUrl  = urlparse(os.environ.get('TRINO_URL'))
Query=[]

print(f"Warehouse URL: {trinoUrl.scheme}://{trinoUrl.hostname}:{trinoUrl.port}/")

conn = connect(
    host=trinoUrl.hostname,
    port=trinoUrl.port,
    auth=trinoAuth,
    http_scheme=trinoUrl.scheme,
    verify=True
)

print('Connected!')

# %%
import pandas as pd

pd.read_sql(f"""SHOW TABLES IN {sharedNS}""", conn)


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Part I. 10 Points

# %% [markdown]
# ### a) Declare an SQL result generator - 2/10
#
# Complete the Python generator below to execute a single query or a list of queries, returning the results row by row.
#
# The generator should implement an out-of-core approach, meaning it should limit memory usage by fetching results incrementally, rather than loading all data into memory at once.

# %%
def sql_fetch(queries, conn, batch_size=100, with_column_names=True):
    if isinstance(queries, str):
        queries = [queries]

    with closing(conn.cursor()) as cur:
        for query in queries:
            cur.execute(query)
            if with_column_names:
                col_names = [desc[0] for desc in cur.description]
                yield col_names  # ⚠️ yield column names first

            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    yield row


# %% [markdown]
# ### b) Explore SBB data - 3/10
#
# Explore the _{sharedNS}.sbb_istdaten_, _{sharedNS}.sbb_stops_, and _{sharedNS}.sbb_stop_times_ tables.
#
# Identify the field(s) used across all three tables to represent stop locations. Analyze their value ranges, format patterns, null and invalid values, and identify any years when null or invalid values are more prevalent. Use this information to implement the necessary transformations for reliably joining the tables on these stop locations.

# %%
### TODO
# Inspect the first couple lines of the three tables:
from itertools import islice
from IPython.display import display

def pretty_table(table_name, query, conn, limit=5):
    gen = sql_fetch(query, conn, with_column_names=True)
    col_names = next(gen)  # first yield = column names
    rows = list(islice(gen, limit))
    df = pd.DataFrame(rows, columns=col_names)
    print(f"\n📄 Preview of table: {table_name}")
    print('-' * 75)
    display(df)
    print('-' * 75)

tables = ['sbb_istdaten', 'sbb_stops', 'sbb_stop_times']

for table in tables:
    query = f"SELECT * FROM {sharedNS}.{table}"
    pretty_table(table, query, conn,5)


# %% [markdown]
# The stop fields in each table are:
#
# <b>sbb_istdaten:</b>
# - bpuic
# - stop_name
#
# <b>sbb_stops:</b>
# - stop_id
# - stop_name
#
# <b>sbb_stop_times:</b>
# - stop_id

# %%
# Preparing the queries:

def basic_info(table, cols, sample = False, sample_rate = 0.1):
    query = f"SELECT COUNT(*) as total_rows"
    for col in cols:
        # distinct counts
        query += f", COUNT(DISTINCT {col} ) as unique_{col}"
        # nulls
        query += f", COUNT(*) FILTER (WHERE {col} IS NULL OR TRIM(CAST({col} AS VARCHAR)) = '') as null_{col}"
        # Max and Min if numeric
        query += f", MIN(TRY(CAST({col} AS INTEGER))) AS min_value_{col}"
        query += f", MAX(TRY(CAST({col} AS INTEGER))) AS max_value_{col}"
    query += f" FROM {sharedNS}.{table}"
    if sample:
        query += f" TABLESAMPLE SYSTEM ({sample_rate})"
    return query


# %%
query = basic_info("sbb_istdaten", ["bpuic", "stop_name"])
pretty_table('sbb_istdaten', query, conn)

# %%
query = basic_info("sbb_stops", ["stop_id", "stop_name"])
pretty_table('sbb_stops', query, conn)

# %%
query = basic_info("sbb_stop_times", ["stop_id"])
pretty_table('sbb_stop_times', query, conn)

# %% [markdown]
# We can see sbb_istdaten has almost 3 billion rows, big data indeed. The bpuic has significantly less null or empty values than the stop names (97 vs ~200 million), so perhaps we should merge our tables on this value, we can see that the buic seems to start with 85 and range from 6 to 9 digits, but there could be typos somewhere. We can also comfirm from the min and max not being NULL that as mentionned in the column description, the stop_name attribute is not always purely alphabetical, some of the stop_names contain the buic (ones before 2021). There are ~25 thousand distinct buics, and ~30 thousand distict stop names, pointing to some inconsistent naming of stop names. 
#
# sbb_stops is smaller with  roughly 12 million rows, while sbb_stop_times is the biggest table with over 3 billion rows! In the sbb_stops and sbb_stop_times data, we notice zero null or empty stop_ids, although perhaps some unconventional number has been chosen to represent the NULL stop and therefore bypasses initial detection. We also notice that both datatables have the same min and max stop_id which is encouraging for joining them together, however it is not the same as sbb_istdaten. sbb_stops has over double the amount of unique stop_ids (\~93,000) compared to unique stop_names (\~42,000), which suggests that perhaps two ID conventions have been used. sbb_stop_times has less unique stop_ids with (\~57,000).

# %%
import matplotlib.pyplot as plt
def null_by_year(table,date_col, other_cols, sample = False, sample_rate = 1):
    query = f"SELECT YEAR({date_col}) AS year, COUNT(*) AS total_rows"
    for col in other_cols:
        query += f""", ROUND(
            100.0 * COUNT(*) FILTER(
                WHERE {col} IS NULL OR TRIM(CAST({col} AS VARCHAR)) = ''
            ) / COUNT(*),
            3
        ) AS {col}_null_percentage"""
    query += f" FROM {sharedNS}.{table}"
    if sample:
        query += f" TABLESAMPLE SYSTEM ({sample_rate})"
    query += f" GROUP BY 1 ORDER BY 1"
    return query

def to_pd_table(table_name, query, conn, limit=200):
    gen = sql_fetch(query, conn, with_column_names=True)
    col_names = next(gen)  # First yield is column names
    rows = list(islice(gen, limit))  # Then fetch up to 'limit' rows
    df = pd.DataFrame(rows, columns=col_names)
    return df

def plot_null_percentages_by_year(
    df, table_name,
    columns=None,
    year_col="year",
    xlabel="Year",
    ylabel="Percentage (%)",
    figsize=(6, 4)
):
    df_plot = df.copy()

    if columns is None:
        columns = [col for col in df_plot.columns if col.endswith("_null_percentage")]

    if year_col not in df_plot.columns:
        raise ValueError(f"Year column '{year_col}' not found in DataFrame.")

    df_plot.set_index(year_col, inplace=True)

    plt.figure(figsize=figsize)

    for col in columns:
        if col in df_plot.columns:
            label = col.replace("_null_percentage", "")
            plt.plot(df_plot.index, df_plot[col], marker='o', label=label)
        else:
            print(f"Column '{col}' not found in DataFrame — skipping.")

    title = f"Null/Empty Percentage per Year for {table_name}"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(title="Column")
    plt.tight_layout()
    plt.xticks(df["year"], df["year"].astype(int))
    plt.show()


# %%
query = null_by_year("sbb_istdaten","operating_day", ["bpuic","stop_name"])
df_ist = to_pd_table("sbb_istdaten", query, conn)
df_ist

# %%
plot_null_percentages_by_year(df_ist,"sbb_istdaten")

# %% [markdown]
# We can see in the graph above that no year seems to have a particularly high percentage of nulls, with 2021 being the year with the lowest percentage of nulls. As seen previously, there are a lot more null stop_names than stop_ids.
#
# Let us now begin the joining process by joining sbb_istdaten and sbb_stops by stop name to see how well they match up, we can then compare the id naming conventions and test to see if we get a better match that way.

# %%
# First count the number of matching distinct stop names
# Note that we only sample the ist table, as it is orders of magnitude bigger
query = f"""
WITH
distinct_ist_stop_names AS (
    SELECT DISTINCT LOWER(TRIM(CAST(stop_name AS VARCHAR))) AS stop_name
    FROM {sharedNS}.sbb_istdaten
--    TABLESAMPLE SYSTEM (1)
),
distinct_stops_stop_names AS (
    SELECT DISTINCT LOWER(TRIM(CAST(stop_name AS VARCHAR))) AS stop_name
    FROM {sharedNS}.sbb_stops 
)

SELECT
    COUNT(*) AS matched_stop_names,
    (SELECT COUNT(*) FROM distinct_ist_stop_names) AS unique_ist_stop_names,
    (SELECT COUNT(*) FROM distinct_stops_stop_names) AS unique_stops_stop_names
FROM distinct_ist_stop_names i
JOIN distinct_stops_stop_names s ON i.stop_name = s.stop_name
"""
pretty_table("sampled_stop_name_match_count", query, conn)


# %% [markdown]
# We see that overall the names seem to match somewhat well on the ist part at least, although stops has significantly more unique stop names, let's look at some individual matches to inspect stop_id:  

# %%
# Visually inspect some sample matches

query = f"""
WITH sampled_istdaten AS (
    SELECT * 
    FROM {sharedNS}.sbb_istdaten
    TABLESAMPLE SYSTEM (1)  
)

SELECT DISTINCT 
    LOWER(TRIM(CAST(i.stop_name AS VARCHAR))) AS ist_stop_name,
    LOWER(TRIM(CAST(s.stop_name AS VARCHAR))) AS stops_stop_name,
    i.bpuic,
    s.stop_id
FROM sampled_istdaten i
JOIN {sharedNS}.sbb_stops s
  ON LOWER(TRIM(CAST(i.stop_name AS VARCHAR))) = LOWER(TRIM(CAST(s.stop_name AS VARCHAR)))
LIMIT 40
"""
pretty_table("stop_name_match_visual_inspection", query, conn, limit=40)


# %% [markdown]
# Visually inspecting the matches, we can see three main types of formatting conventions for stop_ids in the sbb_stops:
# - {buic}
# - Parent{buic}
# - {buic}:\*:\*
#   
# Let us count how many rows of sbb_stops follow each format

# %%
query = fr"""
WITH distinct_stop_ids AS (
    SELECT DISTINCT stop_id
    FROM {sharedNS}.sbb_stops
)

SELECT
  CASE
    WHEN REGEXP_LIKE(stop_id, '^Parent\d+$') THEN 'Parentbpuic'
    WHEN REGEXP_LIKE(stop_id, '^\d+:.*') THEN 'bpuic[:...]'
    WHEN REGEXP_LIKE(stop_id, '^\d+$') THEN 'bpuic'
    ELSE 'Other'
  END AS stop_id_format,
  COUNT(*) AS count
FROM distinct_stop_ids
GROUP BY 1
ORDER BY count DESC
"""
pretty_table("refined stop_id formats for sbb_stops", query, conn)


# %% [markdown]
# It looks like the majority of IDs do follow this format!
# Lets do a sanity check with the other two tables

# %%
query = fr"""
WITH distinct_stop_ids AS (
    SELECT DISTINCT stop_id
    FROM {sharedNS}.sbb_stop_times
)

SELECT
  CASE
    WHEN REGEXP_LIKE(stop_id, '^Parent\d+$') THEN 'Parentbpuic'
    WHEN REGEXP_LIKE(stop_id, '^\d+:.*') THEN 'bpuic[:...]'
    WHEN REGEXP_LIKE(stop_id, '^\d+$') THEN 'bpuic'
    ELSE 'Other'
  END AS stop_id_format,
  COUNT(*) AS count
FROM distinct_stop_ids
GROUP BY 1
ORDER BY count DESC
"""
pretty_table("refined stop_id formats for sbb_stop_times", query, conn)

# %%
query = fr"""
WITH distinct_stop_ids AS (
    SELECT DISTINCT bpuic
    FROM {sharedNS}.sbb_istdaten
)

SELECT
  CASE
    WHEN REGEXP_LIKE(CAST(bpuic AS VARCHAR), '^Parent\d+$') THEN 'Parentbpuic'
    WHEN REGEXP_LIKE(CAST(bpuic AS VARCHAR), '^\d+:.*') THEN 'bpuic[:...]'
    WHEN REGEXP_LIKE(CAST(bpuic AS VARCHAR), '^\d+$') THEN 'bpuic'
    ELSE 'Other'
  END AS stop_id_format,
  COUNT(*) AS count
FROM distinct_stop_ids
GROUP BY 1
ORDER BY count DESC
"""
pretty_table("refined stop_id formats for sbb_istdaten", query, conn)


# %% [markdown]
# We see that istdaten and stop_times almost exactly fall in the given categories! Let's check how many matches we now get on stop_id for all three tables:

# %%
query = f"""
WITH
-- Distinct bpuic from istdaten (sampled for performance)
distinct_ist_bpuic AS (
    SELECT DISTINCT CAST(bpuic AS VARCHAR) AS bpuic
    FROM {sharedNS}.sbb_istdaten
  --  TABLESAMPLE SYSTEM (1)
),

-- Extract bpuic from sbb_stops.stop_id
distinct_stops_bpuic AS (
    SELECT DISTINCT REGEXP_EXTRACT(stop_id, '(\\d+)') AS bpuic
    FROM {sharedNS}.sbb_stops
    WHERE stop_id IS NOT NULL
),

-- Extract bpuic from sbb_stop_times.stop_id
distinct_times_bpuic AS (
    SELECT DISTINCT REGEXP_EXTRACT(stop_id, '(\\d+)') AS bpuic
    FROM {sharedNS}.sbb_stop_times
    WHERE stop_id IS NOT NULL
)

-- Count unique and overlapping bpuics
SELECT
    (SELECT COUNT(*) FROM distinct_ist_bpuic) AS unique_ist_bpuic,
    (SELECT COUNT(*) FROM distinct_stops_bpuic) AS unique_stops_bpuic,
    (SELECT COUNT(*) FROM distinct_times_bpuic) AS unique_times_bpuic,

    -- Match in all three
    (SELECT COUNT(*) 
     FROM distinct_ist_bpuic i
     JOIN distinct_stops_bpuic s ON i.bpuic = s.bpuic
     JOIN distinct_times_bpuic t ON i.bpuic = t.bpuic
    ) AS matched_all
"""
pretty_table("bpuic_match_summary_across_tables", query, conn)


# %% [markdown]
# We see that we get an almost perfect match, with only around 700 missing matched from istdaten!
#
# We could now perform the full join with the following code:

# %%
if False:
    output_table_name = f"{sharedNS}.joined_istdaten_stops_times_clean"
    
    # Drop the table if it exists
    with closing(conn.cursor()) as cur:
        cur.execute(f"DROP TABLE IF EXISTS {output_table_name}")
        print(f"Dropped existing table: {output_table_name}")
    
    # Query using the previously working matching logic
    query = f"""
    CREATE TABLE {output_table_name} AS
    WITH
    -- Step 1: Sample sbb_istdaten
    sampled_istdaten AS (
        SELECT *,
               CAST(bpuic AS VARCHAR) AS bpuic_str
        FROM {sharedNS}.sbb_istdaten
        TABLESAMPLE SYSTEM (1)
    ),
    
    -- Step 2: Extract bpuic from stop_id in sbb_stops (allow any format)
    deduped_stops AS (
        SELECT *
        FROM (
            SELECT *,
                   REGEXP_EXTRACT(stop_id, '(\\d+)') AS bpuic_extracted, 
                   ROW_NUMBER() OVER (PARTITION BY stop_id ORDER BY pub_date DESC) AS rn
            FROM {sharedNS}.sbb_stops
            WHERE stop_id IS NOT NULL
        )
        WHERE rn = 1 AND bpuic_extracted IS NOT NULL --Avoid combinatorial explosion, only choose first match
    ),
    
    -- Step 3: Extract bpuic from stop_id in sbb_stop_times
    deduped_stop_times AS (
        SELECT *
        FROM (
            SELECT *,
                   REGEXP_EXTRACT(stop_id, '(\\d+)') AS bpuic_extracted, 
                   ROW_NUMBER() OVER (PARTITION BY stop_id ORDER BY pub_date DESC) AS rn
            FROM {sharedNS}.sbb_stop_times
            WHERE stop_id IS NOT NULL
        )
        WHERE rn = 1 AND bpuic_extracted IS NOT NULL
    )
    
    -- Final join: ist → stops → stop_times
    SELECT 
        -- From sbb_istdaten
        i.operating_day,
        i.trip_id,
        i.operator_id,
        i.operator_abrv,
        i.operator_name,
        i.product_id,
        i.line_id,
        i.line_text,
        i.transport,
        i.bpuic,
        i.stop_name AS ist_stop_name,
        i.arr_time,
        i.arr_actual,
        i.arr_status,
        i.dep_time,
        i.dep_actual,
        i.dep_status,
        i.transit,
    
        -- From sbb_stops
        s.stop_id AS stops_stop_id,
        s.stop_name AS stops_stop_name,
        s.stop_lat,
        s.stop_lon,
        s.location_type,
        s.parent_station,
    
        -- From sbb_stop_times
        t.arrival_time,
        t.departure_time,
        t.stop_sequence
    
    FROM sampled_istdaten i
    JOIN deduped_stops s
      ON i.bpuic_str = s.bpuic_extracted
    JOIN deduped_stop_times t
      ON s.stop_id = t.stop_id
    """
    
    # Run the query
    with closing(conn.cursor()) as cur:
        cur.execute(query)
    
    print(f"Table created successfully: {output_table_name}")


# %%
if False:
    # Check structure of joined table
    query = f"SELECT * FROM {output_table_name}"
    pretty_table(output_table_name, query, conn,5)

# %%
if False:
    # Check row count of joined table
    query = f"SELECT COUNT(*) AS row_count FROM {output_table_name}"
    pretty_table("Row count of joined table", query, conn)


# %% [markdown]
# ### c) Type of transport - 5/10
#
# Explore the distribution of _product_id_ in _{sharedNS}.sbb_istdaten_ for the whole of 2024 and visualize it in a bar graph.
#
# - Query the istdaten table to get the total number of stop events for different types of transport in each month.
# |year|month|product|stops|
# |---|---|---|---|
# |...|...|...|...|
# - Create a facet bar chart of monthly counts, partitioned by the type of transportation. 
# - If applicable, document any patterns or abnormalities you can find.
#
# __Note__: 
# - One entry in the sbb istdaten table means one stop event, with information about arrival and departure times.
# - We recommend the facet _bar_ plot with plotly: https://plotly.com/python/facet-plots/ the monthly count of stop events per transport mode as shown below (the number of _product_id_ may differ):
#
# ```
# fig = px.bar(
#     df_ttype, x='month_year', y='stops', color='ttype',
#     facet_col='ttype', facet_col_wrap=3, 
#     facet_col_spacing=0.05, facet_row_spacing=0.2,
#     labels={'month_year':'Month', 'stops':'#stops', 'ttype':'Type'},
#     title='Monthly count of stops'
# )
# fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# fig.update_yaxes(matches=None, showticklabels=True)
# fig.update_layout(showlegend=False)
# fig.show()
# ```
#
#
# <img src="./figs/1a-example.png" alt="1a-example.png" width="400"/>

# %%
# %%time
query = f"""
SELECT 
    YEAR(operating_day) AS year,
    MONTH(operating_day) AS month,
    LOWER(product_id) AS ttype,
    COUNT(*) AS stops
FROM {sharedNS}.sbb_istdaten
WHERE YEAR(operating_day) = 2024
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
"""
df_ttype = to_pd_table("Monthly stop events by transport type", query, conn)

# %%
df_ttype["month_year"] = pd.to_datetime({
    "year": df_ttype["year"],
    "month": df_ttype["month"],
    "day": 1
}).dt.strftime("%Y-%m")
df_ttype

# %%
### TODO - display the results to match the graph above
import plotly.express as px

fig = px.bar(
    df_ttype, x='month_year', y='stops', color='ttype',
    facet_col='ttype', facet_col_wrap=3,
    facet_col_spacing=0.05, facet_row_spacing=0.2,
    labels={'month_year': 'Month', 'stops': '# Stops', 'ttype': 'Type'},
    title='Monthly count of stops by transport type (2024)'
)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(matches=None, showticklabels=True)
fig.update_layout(showlegend=False)
fig.show()


# %% [markdown]
# We notice that transport type had inconsistent capitalisation so everything has been put to lowercase. We also notice language inconsistencies with some transport modes being marked in German, which could potentially correspond to the same transports as some of the english names. We also notice different scales, with some transports going into the tens of Millions such as bus, while taxi has numbers ranging from less than 10 to 100. We also notice an odd transport type, wm-bus, which should probably be included along with bus. We also notice a seasonality pattern in the schiff (boat?) and zahnradbahn modes of transports, we also see a slight summer drop in metro use.

# %% [markdown]
# ---

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Part II. 50 Points

# %% [markdown]
# In this second Part, we will leverage the historical SBB data to model the public transport infrastructure within the Lausanne region.
#
# Our objective is to establish a comprehensive data representation of the public transport network, laying the groundwork for our final project. While we encourage the adoption of a data structure tailored to the specific requirements of your final project implementation, the steps outlined here provide a valuable foundation.
#
# In this part you will make good use of DQL statements of nested SELECT, GROUP BY, JOIN, IN, DISTINCT, and Geo Spatial UDF.

# %% [markdown]
# You must create a managed database within your designated namespace, where you'll define the tables necessary for modeling your infrastructure. By 'managed,' we mean that you should not specify a default external location for the namespace.
#
# While it's a bit of an overkill, the safest approach is to drop and recreate the namespace the first time you run this notebook.

# %% jupyter={"outputs_hidden": true}
list(sql_fetch([
    f"""DROP SCHEMA IF EXISTS {namespace} CASCADE""", # CASCADE will drop all the tables
    f"""CREATE SCHEMA IF NOT EXISTS {namespace}""",
], conn))

# %%
# solving schema issues by manually creating user schema

# def sql_fetch(queries, conn):
#     with closing(conn.cursor()) as cur:
#         for query in queries:
#             cur.execute(query)
#         return ["Success"]
# list(sql_fetch([
#     f"""DROP SCHEMA IF EXISTS {namespace} CASCADE""",  # CASCADE will drop all the tables
#     f"""CREATE SCHEMA IF NOT EXISTS {namespace}""",    # Create the schema if not already present
# ], conn))


# %%
# just checking that it worked
# I reran the original sql_fetch code after creating shayakhm schema
# schemas = pd.read_sql(f"SHOW SCHEMAS IN iceberg", conn)
# filtered_schemas = [schema for schema in schemas['Schema'] if schema.startswith('s')]
# print(filtered_schemas)

# %% [markdown]
# ### a) Find the stops in Lausanne region - 5/50
#
#

# %% [markdown]
# * Explore _{sharedNS}.geo_ and find the records containing the _wkb_geometry_ shapes of the _Lausanne_ and _Ouest lausannois_ districts.
#      * The shape is from swiss topo
# * Find all the stops in the _Lausanne_ district from _{sharedNS}.sbb_stops_, as of the first week of July 2024 (use [geo spatial](https://trino.io/docs/471/functions/geospatial.html) functions)
# * Save the results into a table _{namespace}.sbb_stops_lausanne_region_ using the CTAS (Create Table As Select) approach.
# * Validation: you should find around $400\pm 25$ stops.
# * Table _{namespace}.sbb_stops_lausanne_region_ is a subset of table _{sharedNS}.sbb_stops_:
#     * _stop_id_
#     * _stop_name_
#     * _stop_lat_
#     * _stop_lon_

# %% [markdown]
# ---
# #### Solutions

# %%
output_table_name = f"{namespace}.sbb_stops_lausanne_region"

# Drop the table if it already exists
with closing(conn.cursor()) as cur:
    cur.execute(f"DROP TABLE IF EXISTS {output_table_name}")
    print(f"Dropped table if it existed: {output_table_name}")

# Define CTAS query with distinct on bpuic
query = f"""
CREATE TABLE {output_table_name} AS
WITH lausanne_region AS (
    SELECT wkb_geometry
    FROM {sharedNS}.geo
    WHERE LOWER(name) IN ('lausanne', 'ouest lausannois')
),

geo_filtered_stops AS (
    SELECT 
        s.stop_id AS stop_id,
        s.stop_name,
        s.stop_lat,
        s.stop_lon,
        REGEXP_EXTRACT(stop_id, '(\\d+)') AS bpuic,
        ROW_NUMBER() OVER (PARTITION BY REGEXP_EXTRACT(stop_id, '(\\d+)') ORDER BY pub_date DESC) AS rn
    FROM {sharedNS}.sbb_stops s
    JOIN lausanne_region g
      ON ST_Contains(
          ST_GeomFromBinary(g.wkb_geometry),
          ST_Point(s.stop_lon, s.stop_lat)
      )
    WHERE pub_date >= DATE '2024-07-01'
      AND pub_date < DATE '2024-07-08'
      AND stop_id IS NOT NULL
      AND REGEXP_EXTRACT(stop_id, '(\\d+)') IS NOT NULL
)

SELECT stop_id,stop_name, stop_lat, stop_lon

FROM geo_filtered_stops
WHERE rn = 1
"""

# Run the CTAS query
with closing(conn.cursor()) as cur:
    cur.execute(query)
    print(f"Table created: {output_table_name}")


# %%
### TODO verify the results
query = f"""
SELECT * FROM {namespace}.sbb_stops_lausanne_region
LIMIT 5
"""
pretty_table("sbb_stops_lausanne_region", query, conn)

# %%
query = f"""
SELECT COUNT(*) AS total_stops
FROM {namespace}.sbb_stops_lausanne_region
"""
pretty_table("Number of stops in Lausanne region", query, conn)


# %% [markdown]
# We validate that the subset table has roughly the right number of rows, and has the correct structure. We add a bpuic column containing only the bpuic identifier as in sbb.ist_daten

# %% [markdown]
# ### b) Find stops with real time data in Lausanne region - 5/50

# %% [markdown]
# * Use the results of table _{username}.sbb_stops_lausanne_region_ to find all the stops for which real time data is reported in the _{sharedNS}.sbb_istdaten_ table for the full month of **July 2024**.
# * Report the results in a pandas DataFrame that you will call _stops_df_.
# * Validation: you should find between 3% and 4% of stops in the area of interest that do not have real time data.
# * Hint: it is recommended to first generate a list of _distinct_ stop identifiers extracted from istdaten data. This can be achieved through either a nested query or by creating an intermediate table (use your findings of Part I.b).

# %% [markdown]
# ---
# #### Solution
#

# %%
### TODO - Create the data frame stops_df
query = f"""
WITH ist_july AS (
    SELECT DISTINCT CAST(bpuic AS VARCHAR) AS bpuic
    FROM {sharedNS}.sbb_istdaten
    WHERE operating_day >= DATE '2024-07-01'
      AND operating_day <  DATE '2024-08-01'
),

lausanne_stops AS (
    SELECT DISTINCT stop_id, stop_name, stop_lat, stop_lon,
           REGEXP_EXTRACT(stop_id, '(\\d+)') AS bpuic
    FROM {namespace}.sbb_stops_lausanne_region
)

SELECT 
    l.stop_id,
    l.stop_name,
    l.stop_lat,
    l.stop_lon,
    CASE WHEN i.bpuic IS NOT NULL THEN TRUE ELSE FALSE END AS has_realtime
FROM lausanne_stops l
LEFT JOIN ist_july i
  ON l.bpuic = i.bpuic
"""

# %%
# %%time
stops_df= to_pd_table("Stops with/without real-time data (July 2024)", query, conn, limit = 1000)

# %%
### TODO - Verify the results
total_stops = len(stops_df)
no_rt_stops = len(stops_df[stops_df["has_realtime"] == False])
pct_missing = 100.0 * no_rt_stops / total_stops

print(f"Total stops in Lausanne region: {total_stops}")
print(f"Stops without real-time data: {no_rt_stops} ({pct_missing:.2f}%)")

# Optional preview
stops_df.head()

# %%
# Filter to stops without real-time data
no_rt_stops_df = stops_df[stops_df["has_realtime"] == False]

# Preview the first few rows
print("Preview: Stops without real-time data (July 2024)")
display(no_rt_stops_df.head())


# %% [markdown]
# Findings: there are five stops that have no real time data, they are listed above.

# %% [markdown]
# ### c) Display stops in the Lausanne Region - 3/50
#
# * Use plotly or similar plot framework to display all the stop locations in Lausanne region on a map (scatter plot or heatmap), using a different color to highlight the stops for which istdaten data is available.

# %%

### TODO - Display results of stops_df

fig = px.scatter_map(data_frame=stops_df, lat="stop_lat", lon="stop_lon", color="has_realtime", hover_name="stop_name")

fig.show()


# %% [markdown]
# Note that some stops lacking real-time data may actually serve as representations for groups of stops, like `Parent8592050` for Lausanne Gare, which denotes a cluster of stops within Lausanne train station. We ignore these.

# %% [markdown]
# ### d) Find stops that are within walking distances of each other - 10/50

# %% [markdown]
# * Use the results of table _{username}.sbb_stops_lausanne_region_ to find all the (directed) pair of stops that are within _500m_ of each other.
# * Save the results in table _{username}.sbb_stops_to_stops_lausanne_region_
# * Validation: you should find around $3500\pm 250$ directed stop paris (each way, i.e. _A_ to _B_ and _B_ to _A_).
# * Hint: Use the Geo Spatial UDF, in spherical geopgraph.
# * Aim for the table _{namespace}.sbb_stop_to_stop_lausanne_region_:
#     * _stop_id_a_: an _{sharedNS}.sbb_stops.stop_id_
#     * _stop_id_b_: an _{sharedNS}.sbb_stops.stop_id_
#     * _distance_: straight line distance in meters from _stop_id_a_ to _stop_id_b_

# %%
# %%time
### TODO - create the stop to stop table

output_table_name = f"{namespace}.sbb_stop_to_stop_lausanne_region"

# Drop the table if it already exists
with closing(conn.cursor()) as cur:
    cur.execute(f"DROP TABLE IF EXISTS {output_table_name}")
    print(f"Dropped table if it existed: {output_table_name}")

create_query = f"""
CREATE TABLE {output_table_name}(
    stop_id_a VARCHAR,
    stop_id_b VARCHAR,
    distance DOUBLE
)
"""

insert_query = f"""
INSERT INTO {output_table_name} (stop_id_a, stop_id_b, distance)
WITH pairwise_distances AS (
    SELECT
        a.stop_id AS stop_id_a,
        b.stop_id AS stop_id_b,
        6371000 * 2 * ASIN(SQRT(
            POWER(SIN(RADIANS((b.stop_lat - a.stop_lat) / 2)), 2) +
            COS(RADIANS(a.stop_lat)) * COS(RADIANS(b.stop_lat)) *
            POWER(SIN(RADIANS((b.stop_lon - a.stop_lon) / 2)), 2)
        )) AS distance
    FROM
        {sharedNS}.sbb_stops_lausanne_region a
    JOIN
        {sharedNS}.sbb_stops_lausanne_region b
    ON a.stop_id != b.stop_id
)
SELECT stop_id_a, stop_id_b, distance
FROM pairwise_distances
WHERE distance <= 500
"""

with closing(conn.cursor()) as cur:
    cur.execute(create_query)
    print(f"Table created: {output_table_name}")
    
    cur.execute(insert_query)
    print(f"Data inserted into {output_table_name}")


# %% [markdown]
# We couldn't create a new UDF in {namespace} as the permission was denied. The spherical distance between the point is calculated from the Harvesine formula:
#
# \begin{align}
# h = \sin^2 (\Delta \phi /2) + \cos(\phi_1) \cos(\phi_2) \sin^2(\Delta \lambda /2)
# \end{align}
#
# with $\phi_1$ and $\phi_2$ being the latitude in radians of points 1 and 2,
# $\lambda_1$ and $\lambda_2$ the longitudes of points 1 and 2 in radians,
# $\Delta \phi = \phi_2 - \phi_1$ and $\Delta \lambda = \lambda_2 - \lambda_1$.
#
# Retreiving the spherical distance is then possible with the formula:
# \begin{align}
# d = 2R\sin^{-1}(\sqrt h)
# \end{align}
#
# sing R as the Earth radius, 6371000 m.

# %%
### TODO - Verify the results

pd.read_sql(f"""SELECT COUNT(*) FROM {namespace}.sbb_stop_to_stop_lausanne_region""", conn)

# %% [markdown]
# ### e) Finds the _stop times_ in Lausanne region - 10/50
#
# * Find the stop times and weekdays of trips (trip_id) servicing stops found previously in the Lausanne region.
# * Use the stop times and calendar information published on the same week as the stops information used to compute the stops in the Lausanne region.
# * Save the results in the table _{username}.sbb_stop_times_lausanne_region_
# * Validation: you should find around $1M\pm 50K$ trip_id, stop_id pairs in total, out of which $450K\pm 25K$ happen on Monday.
#
# At a minimum, the table should be as follow. Use the provided information to decide the best types for the fields.
#
# * _{namespace}.sbb_stop_times_lausanne_region_ (subset of _{sharedNS}.sbb_stop_times_ and _{sharedNS}.sbb_calendar_).
#     * _trip_id_
#     * _stop_id_
#     * _departure_time_
#     * _arrival_time_
#     * _monday_ (trip happens on Monday)
#     * _tuesday_
#     * _wednesday_
#     * _thursday_
#     * _friday_
#     * _saturday_
#     * _sunday_
#  
# **Hints:**
# * Pay special attention to the value ranges of the _departure_time_ and _arrival_time_ fields in the _{sharedNS}.sbb_stop_times_ table.
# * This new table will be used in the next exercise for a routing algorithm. We recommend reviewing the upcoming questions to determine the appropriate data types and potential transformations for the _departure_time_ and _arrival_time_ fields.

# %%
### TODO - Create the table stop times as described above
# # %%time
### TODO - Create the table stop times as described above

output_table_name = f"{namespace}.sbb_stop_times_lausanne_region"

# Drop the table if it already exists
with closing(conn.cursor()) as cur:
    cur.execute(f"DROP TABLE IF EXISTS {output_table_name}")
    print(f"Dropped table if it existed: {output_table_name}")

create_query = f"""
CREATE TABLE {output_table_name}(
    trip_id VARCHAR,
    stop_id VARCHAR,
    departure_time TIME,
    arrival_time TIME,
    monday BOOLEAN,
    tuesday BOOLEAN,
    wednesday BOOLEAN,
    thursday BOOLEAN,
    friday BOOLEAN,
    saturday BOOLEAN,
    sunday BOOLEAN
)
"""


insert_query = f"""INSERT INTO {output_table_name} (trip_id, stop_id, departure_time, arrival_time, monday, tuesday, wednesday, thursday, friday, saturday, sunday)
SELECT DISTINCT
    a.trip_id,
    a.stop_id,
    CAST(
      CASE 
        WHEN CAST(split(a.departure_time, ':')[1] AS INTEGER) >= 24 THEN
          LPAD(CAST((CAST(split(a.departure_time, ':')[1] AS INTEGER) - 24) AS VARCHAR), 2, '0') || ':' ||
          split(a.departure_time, ':')[2] || ':' ||
          split(a.departure_time, ':')[3]
        ELSE a.departure_time
      END AS TIME
    ) AS departure_time,
    CAST(
      CASE 
        WHEN CAST(split(a.arrival_time, ':')[1] AS INTEGER) >= 24 THEN
          LPAD(CAST((CAST(split(a.arrival_time, ':')[1] AS INTEGER) - 24) AS VARCHAR), 2, '0') || ':' ||
          split(a.arrival_time, ':')[2] || ':' ||
          split(a.arrival_time, ':')[3]
        ELSE a.arrival_time
      END AS TIME
    ) AS arrival_time,
    b.monday,
    b.tuesday,
    b.wednesday,
    b.thursday,
    b.friday,
    b.saturday,
    b.sunday
FROM
    {sharedNS}.sbb_trips c
JOIN {sharedNS}.sbb_stop_times a ON c.trip_id = a.trip_id
JOIN {sharedNS}.sbb_calendar b ON c.service_id = b.service_id
WHERE c.pub_date BETWEEN DATE '2024-07-03' AND DATE '2024-07-07'
  AND a.pub_date BETWEEN DATE '2024-07-01' AND DATE '2024-07-07'
  AND b.pub_date BETWEEN DATE '2024-07-03' AND DATE '2024-07-07'
  AND a.stop_id IN (SELECT stop_id FROM {namespace}.sbb_stops_lausanne_region)
"""


with closing(conn.cursor()) as cur:
    cur.execute(create_query)
    print(f"Table created: {output_table_name}")
    
    cur.execute(insert_query)
    print(f"Data inserted into {output_table_name}")  

# %% [markdown]
# Some departure and arrival times were larger than 24h as they arrive on the next day so we removed 24 to the hour number so they can be cast as TIME variables.
#
# sbb_trips and sbb_calendar had two entries on the first week of July so we selected one of them to avoid duplicates.

# %%
query = f"""
SELECT * FROM {sharedNS}.sbb_stop_times
LIMIT 10
"""
pretty_table("sbb_stop_times", query, conn, 10)

# %%
### TODO - Verify the results

query = f"""
SELECT COUNT(*) AS total_pairs
FROM (
  SELECT DISTINCT trip_id, stop_id
  FROM {namespace}.sbb_stop_times_lausanne_region
) AS distinct_pairs
"""

pd.read_sql_query(query, conn)


# %%
query = f"""
SELECT COUNT(*) AS monday_pairs
FROM (
  SELECT DISTINCT trip_id, stop_id
  FROM {namespace}.sbb_stop_times_lausanne_region
  WHERE monday = TRUE
) AS monday_pairs
"""

pd.read_sql_query(query, conn)

# %% [markdown]
# ### f) Design considerations - 2/50
#
# We aim to use our previous findings to recommend an optimal public transport route between two specified locations at a given time on any day of a particular week in any region.
#
# Running queries on all data for the entire data set would be inefficient. Could you suggest an optimized table structure to improve the efficiency of queries on the {username}.sbb_stop_times_lausanne_region table?

# %% [markdown]
# ---
# #### Solutions
#
# **TODO**: Design considerations, no code required.
#
# The sbb_stop_times_lausanne_region table has the following columns:
# * _trip_id_
# * _stop_id_
# _departure_time_
# * _arrival_time_
# * _monday_
# * _tuesday_
# * _wednesday_
# * _thursday_
# * _friday_
# * _saturday_
# * _sunday_
#
# To recommend optimal public transport routes efficiently, we can consider using the following designs to minimize computational overhead with HDFS, Apache Iceberg, and Trino:
#
# <u>i) Metadata Indexing and Partitioning</u> <br>
# Apache Iceberg inherently has metadata indexing, which allows Trino to optimize queries by pruning unnecessary data before scanning the actual table.
#
# Moreover, we can Iceberg partition the table temporally such as by weekdays or by specific time ranges for departure and arrival. Thus, the large table will be divided into smaller, more manageable partitions, which will only be scanned when they match the search criteria.
#
# For example, if we partition by weekdays, we would have separate blocks for Monday, Tuesday, Wednesday, and so on. If we partition by departure or arrival time, we would create specific time period blocks, and only the relevant partitions for the queried time range would be examined.
#
# <u>ii) Compaction</u> <br>
# Indexing and Partitioning are skipping-files approaches. On the other hand, using compaction methods like Z-ordering would involve reorganizing the table to optimize query performance.  By doing so, we are physically grouping similar records based on significant search query elements like departure_time and stop_id. This table reorganization reduces disk I/O as fewer data blocks need to be read.
#
# <u>iii) Materialized Views</u> <br>
# We can analyze past data to identify the most common queries. Based on this information, we can precompute common aggregations, such as the average duration between stops, and store them in Trino’s materialized views. Additionally, we can create route-specific views for popular routes, such as the popular M1 route from Lausanne Flon to Renens.
#
# This approach helps save computational resources by avoiding the need to recalculate common search queries each time.
#

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### h) Isochrone Map - 15/50

# %% [markdown]
# Note: This question is open-ended, and credits will be allocated based on the quality of both the proposed algorithm and its implementation. You will receive credits for proposing a robust algorithm, even if you do not carry out the implementation.
#
# Moreover, it is not mandatory to utilize the large scale database for addressing this question; plain Python is sufficient. You are free to employ any Python package you deem necessary. However, ensure that you list it as a prerequisite of this notebook so that we remember to install them.

# %% [markdown]
# **Question**:
# * Given a time of day on Monday (or any other day of the week you may choose), and a starting point in Lausanne area.
# * Propose a routing algorithm (such as Bellman-Ford, Dijkstra, A-star, etc.) that leverages the previously created tables to estimate the shortest time required to reach each stop within the Lausanne region using public transport.
# * Visualize the outcomes through a heatmap (e.g., utilizing Plotly), where the color of each stop varies based on the estimated travel time from the specified starting point. See example:
#
# ![example](./figs/isochrone.png).
#
# * Hints:
#     - Focus solely on scenarios where walking between stops is not permitted. Once an algorithm is established, walking can optionally be incorporated, assuming a walking speed of 50 meters per minute. Walking being optional, bonus points (+2) will be awarded for implementing it. 
#     - If walking is not considered, a journey consists of a sequence of stop_ids, each separated by a corresponding trip_id, in chronological order. For example: stop-1, trip-1, stop-2, trip-2, ..., stop-n.
#     - Connections between consecutive stops and trips can only occur at predetermined times. Each trip-id, stop-id pair must be unique and occur at a specific time on any given day according to the timetable. If you want to catch an earlier connection, you must have taken an earlier trip; you cannot go back in time once you've arrived at a stop.
#     - Consider both a _label setting_ and a _label correcting_ method when making your design decision.

# %% [markdown]
# ---
# #### Solution

# %% [markdown]
# **TODO**: Explain your algorithm, design decisions etc. here

# %%
# Peeking at the tables that we will be using 

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
query = f"""
SELECT *
FROM {namespace}.sbb_stop_to_stop_lausanne_region
LIMIT 5
"""
pretty_table("sbb_stop_to_stop_lausanne_region", query, conn, 5)

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
query = f"""
SELECT *
FROM {namespace}.sbb_stop_times_lausanne_region
LIMIT 5
"""
pretty_table("sbb_stop_times_lausanne_region", query, conn, 5)

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
query = f"""
SELECT *
FROM {namespace}.sbb_stop_to_stop_lausanne_region
LIMIT 5
"""
pretty_table("sbb_stop_to_stop_lausanne_region", query, conn, 5)

# %% jupyter={"outputs_hidden": true}
# Install these necessary libraries please 
# !pip install networkx
# !pip install datetime
# !pip install geopy

# %%
# Importing the necessary libraries that were not imported above
import heapq
import numpy as np
import networkx as nx
import plotly.express as px
import ipywidgets as widgets
import plotly.graph_objs as go
from IPython.display import display
from geopy.distance import geodesic
from datetime import datetime, timedelta

# %%
### TODO - Data Preparation
# Using the sbb_stop_times_lausanne table, we are going to create a NetworkGx Graph.


# %%
# Prepping the data to build a graph 

# Note: takes about a minute to run 

# lausanne stop rimes 
query = f"SELECT * FROM {namespace}.sbb_stop_times_lausanne_region"
lausanne_stop_times = pd.read_sql(query, conn)
# lausanne stops 
query = f"SELECT * FROM {namespace}.sbb_stops_lausanne_region"
lausanne_stops = pd.read_sql(query, conn)

# making the times timedelta objects 
lausanne_stop_times['departure_td'] = lausanne_stop_times['departure_time'].apply(lambda t: timedelta(hours=t.hour, minutes=t.minute, seconds=t.second))
lausanne_stop_times['arrival_td'] = lausanne_stop_times['arrival_time'].apply(lambda t: timedelta(hours=t.hour, minutes=t.minute, seconds=t.second))

# decided to filter by wednesday rather than monday
lausanne_stop_times = lausanne_stop_times[lausanne_stop_times['wednesday'] == True].copy()

# sorting chronologically 
lausanne_stop_times = lausanne_stop_times.sort_values(by=['trip_id', 'departure_td']).reset_index(drop=True)

# mapping stop id to its name 
stop_names = dict(zip(lausanne_stops['stop_id'], lausanne_stops['stop_name']))





# %%
# Building the transport graph 
# Note: it takes about 2-3 minutes to run


# our empty graph 
G = nx.DiGraph()

# Graph structure:
# node = stop id and time at the stop 
# edge = route between stops if part of same trip id 
# each edge weight = time between stops 

# populating graph
# we're iterating over each trip and the stops included in each trip 
for trip_id, trip_group in lausanne_stop_times.groupby("trip_id"):
    trip_group = trip_group.sort_values("departure_td").reset_index(drop=True) #sorting chronologically again because I had issues with just doing it once above

    # defining the current stop (a) and its connecting stop (b) to build edge a-b 
    for i in range(len(trip_group) - 1):
        a = trip_group.loc[i]
        b = trip_group.loc[i + 1]
        from_node = (a['stop_id'], a['departure_td'])
        to_node = (b['stop_id'], b['arrival_td'])

        # arrival_time: scheduled (local) time of arrival at the stop
        # departure_time: scheduled (local) time of departure at the stop
        # thus the time it takes to get from one stop to another is arrival_time of b - departure_time of a
        travel_time = (b['arrival_td'] - a['departure_td']).total_seconds() / 60.0

        
        G.add_edge(from_node, to_node, weight=travel_time, trip_id=trip_id,
                   dep_time=a['departure_td'], arr_time=b['arrival_td'])





# %%
# Adding the walking consideration
# Note: takes about a minute to run

walking_speed = 50  # meters per minute speed constraint

# loading the precomputed walking distances between stops from 2d
query = f"SELECT * FROM {namespace}.sbb_stop_to_stop_lausanne_region"
walking_data = pd.read_sql(query, conn)

# filtering only to the max distance we're okay walking ? I am not sure if this part is necessary so it's commented out
# maximum_walking_distance = 500  
# walking_data = walking_data[walking_data['distance'] <= maximum_walking_distance]

# grouping already created nodes by stop_id for efficiency
nodes_by_stop = {}
for node in G.nodes:
    sid, time = node
    nodes_by_stop.setdefault(sid, []).append(node)

# populating graph with walking edges  
for _, row in walking_data.iterrows():
    stop_a = row['stop_id_a']
    stop_b = row['stop_id_b']
    distance = row['distance']
    # how much time it will take to walk if our speed is 50m/min speed
    walk_time = distance / walking_speed  

    
    # making a walking edge from a to b 
    # we made trip_id = 'WALK' so we can print out the path in the visualization
    for node_a in nodes_by_stop.get(stop_a, []):
        dep_time = node_a[1]
        arr_time = dep_time + timedelta(minutes=walk_time) #when we leave + how much time it will take to walk to destination
        node_b = (stop_b, arr_time)

        G.add_edge(node_a, node_b, weight=walk_time, trip_id="WALK", dep_time=dep_time, arr_time=arr_time)

    # walking is bidirectional, so we make a walking edge from b to a 
    for node_b in nodes_by_stop.get(stop_b, []):
        dep_time = node_b[1]
        arr_time = dep_time + timedelta(minutes=walk_time)
        node_a = (stop_a, arr_time)

        G.add_edge(node_b, node_a, weight=walk_time, trip_id="WALK",dep_time=dep_time, arr_time=arr_time)




# %%
### TODO - Routing Algorithm

# We decided to use a Dijkstra base + label-setting algorithm
# so we find the earliest arrival times from a starting stop and time to all other stops within a maximum duration 
# it accounts for both walking and public transport 
# and it explores nodes based on the current known shortest total travel time


# %%

# we input the chosen criteria in the run 
def routing_algorithm(start_stop_id, start_time_input, max_duration_minutes):

    # filtering graph nodes by start stop id and start time 
    start_time = pd.to_timedelta(start_time_input)
    candidate_nodes = [n for n in G.nodes if n[0] == start_stop_id and n[1] >= start_time]

    # if candidates are empty 
    if not candidate_nodes:
        print("No trips available from this stop at or after selected time.")
        return {}, {}

    # picking node that leaves first 
    start_node = min(candidate_nodes, key=lambda x: x[1])

    # dijkstra inspo 
    # using a priority queue where our first one is the node that leaves first
    # the (0,0,0) would be  # (total_time, walk_time, transit_time, node)
    queue = [(0, 0, 0, start_node)] 
    best_duration = {start_node: (0, 0, 0)} #shortest known time to each node
    paths = {start_node: [start_node]} #we're also gonna store our path so that we can display it in the viz below

    # keeping track of the visited stops
    # I had issues where the algo with actually go in a loop and revisit the same stops again 
    # but this solved it 
    visited = {}

    while queue:

        # we're popping the next best node 
        cum_time, cum_walk, cum_transit, curr_node = heapq.heappop(queue)

        # skip it though if there's a better way to reach it 
        if cum_time > best_duration.get(curr_node, (float('inf'),))[0]:
            continue

        # grabbing the stop id and arrival time of the node 
        curr_stop, curr_arrival = curr_node

        # if we have already visited the node before, skip ! 
        if curr_stop in visited and curr_arrival >= visited[curr_stop]:
            continue
        # otherwise add to visited list 
        visited[curr_stop] = curr_arrival

        # now let's explore all the outgoing connections 
        for neighbor in G.successors(curr_node):
            edge = G.edges[curr_node, neighbor]
            dep_time = edge['dep_time']
            arr_time = edge['arr_time']

            # I can't leave the stop before I arrive to it, so skip this type of node
            if dep_time < curr_node[1]:
                continue

            # sometimes, i'll have to wait at a stop until the next bus/train/transport leaves 
            wait_time = (dep_time - curr_node[1]).total_seconds() / 60.0
            
            travel_time = edge['weight']

            # am I walking or taking some form of transport? let's note it 
            mode = 'walk' if edge.get('trip_id') == 'WALK' else 'transit'

            # if I take this path, need to update the cumulative walking, transit, and total trip durations 
            # I decided to track each separately so I would know how long I would walk, sit, etc. 
            new_walk = cum_walk + (travel_time if mode == 'walk' else 0)
            new_transit = cum_transit + (travel_time if mode == 'transit' else 0)
            total_time = cum_time + wait_time + travel_time

            # there is a total travel time constraint from the input
            if total_time <= max_duration_minutes:

                # if this is the best path so far, update the duration, path sequence, and add to queue
                # so this is the label setting in our code
                if total_time < best_duration.get(neighbor, (float('inf'),))[0]:
                    best_duration[neighbor] = (total_time, new_walk, new_transit)
                    paths[neighbor] = paths[curr_node] + [neighbor]
                    heapq.heappush(queue, (total_time, new_walk, new_transit, neighbor))

    # storage for best results
    final_times = {}
    final_paths = {}
    stop_arrivals = {}

    # for each stop in our graph, grab the earliest arrival and the path to get to it 
    for node, (total, walk, transit) in best_duration.items():
        stop_id, arrival_td = node
        if stop_id not in stop_arrivals or arrival_td < stop_arrivals[stop_id][0]:
            stop_arrivals[stop_id] = (arrival_td, total, walk, transit)
            final_paths[stop_id] = paths[node] 

    # for the click in the visualiztion, I wanted to format it nicely 
    for stop_id, (arr_td, total, walk, transit) in stop_arrivals.items():
        final_times[stop_id] = {
            "arrival_td": arr_td,
            "duration_total": total,
            "duration_walk": walk,
            "duration_transit": transit
        }

    return final_times, final_paths



# %%
###TODO - Interactive Interface to Verify the Algorithm

# %%
# making the input widgets
stop_dict = dict(sorted(zip(lausanne_stops['stop_name'], lausanne_stops['stop_id'])))
start_stop_widget = widgets.Dropdown(options=stop_dict, description='Start Stop:')
hour_widget = widgets.BoundedIntText(value=8, min=0, max=23, description='Hour:')
minute_widget = widgets.BoundedIntText(value=0, min=0, max=59, description='Min:')
duration_widget = widgets.BoundedIntText(value=60, min=5, max=540, step=5, description='Max Time (min):')
run_button = widgets.Button(description="Run", button_style='success')
output = widgets.Output()

# function to run the viz on different inputs 
def on_run_clicked(_):

    # since we are using widgets, everything is redirected to output widget
    with output:
        output.clear_output() 
        # grabbing the inputs 
        stop_id = start_stop_widget.value
        hour = hour_widget.value
        minute = minute_widget.value
        max_time = duration_widget.value
        time_str = f"{hour:02}:{minute:02}:00"

        # running our routing algorithm and storing the best times and paths 
        result_times, result_paths = routing_algorithm(stop_id, time_str, max_time)

        # number of stops we can reach 
        if len(result_times) == 1:
            print("Unfortunately, only the start stop is reachable with those inputs. Try a different time or start location.")
        else:
            print(f"Number of reachable stops: {len(result_times)}")


        # what will be displayed when we click on a point in the map  
        results_df = pd.DataFrame([
            {
                "stop_id": sid,
                "arrival_time": str(vals["arrival_td"].components.hours).zfill(2) + ':' +
                                str(vals["arrival_td"].components.minutes).zfill(2),
                "duration_total": int(vals["duration_total"]),
                "duration_walk": int(vals["duration_walk"]),
                "duration_transit": int(vals["duration_transit"]),
                "path": result_paths[sid]
            }
            for sid, vals in result_times.items()
        ])

        # we're also gonna add the long and lat to the displayed info 
        merged = results_df.merge(lausanne_stops, on="stop_id")


        # building our interactive map 
        fig = go.FigureWidget(
            px.scatter_mapbox(
                merged,
                lat="stop_lat", lon="stop_lon",
                color="duration_total", size_max=15, zoom=12,
                # colour scheme like in the example viz
                color_continuous_scale=[
                    (0.0, "#6a0dad"), 
                    (0.5, "#ffa500"),  
                    (1.0, "#ffff00"),   
                ],
                hover_data={
                    "stop_name": True,
                    "duration_total": True,
                    "duration_walk": True,
                    "duration_transit": True,
                    "arrival_time": True,
                },
                title=f"Isochrone Map from {start_stop_widget.label} at {time_str}"
            )
        )

        fig.update_layout(mapbox_style="carto-positron", height=600)

        text_output = widgets.Output()

        # when we click on a point on the map, show the full path to that stop at the bottom 
        # this was usefu for testing and seeing whether it goes in circles or not 
        # I was also using this to fact check and compare with SBB app 
        def handle_click(trace, points, state):
            with text_output:
                text_output.clear_output()

                # grab the clicked stop's information
                if points.point_inds:
                    idx = points.point_inds[0]
                    selected_stop = merged.iloc[idx]
                    stop_id_clicked = selected_stop['stop_id']
                    path = result_paths.get(stop_id_clicked, [])
                    print(f"Path to {selected_stop['stop_name']}:\n")
        
                    for i in range(len(path) - 1):
                        from_node = path[i]
                        to_node = path[i + 1]
                        from_stop, from_time = from_node
                        to_stop, to_time = to_node
        
                        if G.has_edge(from_node, to_node):
                            edge = G.edges[from_node, to_node]
                            mode = "🚶 Walk" if edge.get("trip_id") == "WALK" else f"🚌 🚆 Transit (Trip: {trip_id})"
                            minutes = edge.get("weight", 0)
                            dep_td = edge.get("dep_time")
                            arr_td = edge.get("arr_time")
                            dep_time_str = (datetime.min + dep_td).time().strftime("%H:%M")
                            arr_time_str = (datetime.min + arr_td).time().strftime("%H:%M")

                            print(f"{stop_names.get(from_stop, from_stop)} at {dep_time_str} → "
                                  f"{stop_names.get(to_stop, to_stop)} at {arr_time_str} "
                                  f"({mode}, {minutes:.1f} min)")
                        else:
                            print(f"{stop_names.get(from_node[0], from_node[0])} → "
                                  f"{stop_names.get(to_node[0], to_node[0])} (No edge)")


        # display
        fig.data[0].on_click(handle_click)
        display(fig)
        display(text_output)

# UI running 
run_button.on_click(on_run_clicked)

display(widgets.VBox([
    start_stop_widget, hour_widget, minute_widget, duration_widget, run_button, output
]))

# %%
### TODO - Others as you see fit.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # General Considerations and Questions About the Algorithm
#
# While working on 2h, I had a few considerations and thoughts about the realism of the current implementation and question:
#
# ## 1. Zero Travel Time 
#
# To calculate the travel time between two stops a and b within the same trip, we subtract the departure time at stop a from the arrival time at stop b. However, in some cases, we observe that the travel time comes out as 0 minutes. 
#
# In the example below, during trip 727.TA.92-21-I-j24-1.12.H on a Monday, lines 2 and 3 have identical departure and arrival times, even though they correspond to two different stops. A similar issue occurs on lines 11 and 12. 
#
# This is unrealistic, as traveling between any two stops should take at least one minute. As a result, our algorithm sometimes shows transit segments with 0-minute durations. 

# %%
query = f"""
SELECT *
FROM {namespace}.sbb_stop_times_lausanne_region
WHERE trip_id = '727.TA.92-21-I-j24-1.12.H'
  AND monday = TRUE
ORDER BY departure_time
"""
pretty_table("sbb_stop_times_lausanne_region", query, conn, 13)


# %% [markdown]
# ## 2. Hold-up Times 
#
# In many entries, the arrival time at a stop is exactly the same as the departure time. However, in real-world scenarios, a bus or train usually waits at a stop for at least a few seconds or even minutes to allow passengers to board and unboard. Moreover, this "hold-up" time can vary significantly depending on the station's size and passenger flow. 
#
# ## 3. Transfer Times
#
# Following the previous two questions, an important factor to consider is the time required to switch between vehicles. Passengers often need 1 min or more to walk from one platform to another or navigate between stops, especially in larger stations. 
#
# So I think for our project, one way to improve realism is to calculate an average transfer time and add it as padding when calciulating the best possible paths. 
#
# ## 4. Empty Trips
#
# While testing the algorithm on different stops, I occasionally noticed runs where only the starting stop appeared in the results. Logically, it does not make sense since surely some transport should be available...
#
# Let’s take Bussigny as an example with a departure time of 23:00 and a maximum travel duration of 60 minutes. We get this output:
#
# ![example](bussigny_example.png).
#
# Weird. Let us check the data to see if there are truly no bus/trains departing Bussigny at that time. Note: 8501117:0:1 == Bus
#

# %%
time_filter = pd.to_timedelta("23:00:00")
bussigny_departures = lausanne_stop_times[
    (lausanne_stop_times['stop_id'] == stop_dict['Bussigny']) &
    (lausanne_stop_times['departure_td'] >= time_filter)]
# Sorting chronologically by departure time
bussigny_departures_sorted = bussigny_departures.sort_values(by='departure_td')
bussigny_departures_sorted

# %% [markdown]
# We see in the table above we have 4 trips departing Bussigny! So why aren't they showing up in our visualized map ??
#
# Let's take a closer look at each trip:

# %%
example_a= '932.TA.91-4-L-j24-1.313.R'
lausanne_stop_times[lausanne_stop_times['trip_id'] == example_a]

# %%
example_b = '914.TA.91-3-T-j24-1.687.R'
lausanne_stop_times[lausanne_stop_times['trip_id'] == example_b]

# %% [markdown]
# The trips above start AND end at Bussigny, explaining why they did not created edges in the graph to new stops. 

# %%
example_c = '297.TA.91-3-T-j24-1.874.H'
lausanne_stop_times[lausanne_stop_times['trip_id'] == example_c]

# %%
example_d = '277.TA.91-4-L-j24-1.293.H'
lausanne_stop_times[lausanne_stop_times['trip_id'] == example_d]

# %% [markdown]
# Even though it looks like something departs from Bussigny, it's actually the last stop of the trip. That’s why our visualization shows no reachable stops from Bussigny.
#
# After thorough examination, I understand why some stops like Bussigny show no connections to other stops, even if it still feels a bit counterintuitive.
#
# Side note: departure time being earlier than arrival time is also questionable. 
#
# <br> <br>
# Overall, the algorithm seems to function well, but there are a few issues to consider, as it doesn't always display the truly optimal paths to reach certain destinations.
#

# %%
