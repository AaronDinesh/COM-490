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

# %% [markdown]
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
from pyspark.storagelevel import StorageLevel
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
            .config("spark.driver.memory", "10g")\
            .config("spark.executor.cores", "2")\
            .config("spark.executor.instances", "7")\
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")\
            .config("spark.driver.maxResultSize", "10g")\
            .master('yarn')\
            .getOrCreate()


checkpoint_hdfs_path = f"{hadoopFS}/user/com-490/group/{groupName}/checkpoints"

spark.sparkContext.setCheckpointDir(checkpoint_hdfs_path)
print(f"Setting the checkpoint dir to be {checkpoint_hdfs_path}")

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

# %% vscode={"languageId": "shellscript"}
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
# stops_lausanne.printSchema()
# print("Rows fetched:", stops_lausanne.count())

# %%
# stops_lausanne
# ### Getting stops with realtime data for Lausanne region:


# %%
from pyspark.sql.functions import col, regexp_extract, lit, when
# 1) Build the ist_july DataFrame
ist_july_df = istdaten.filter((col("operating_day") >= lit("2024-07-01")) &\
          (col("operating_day") <  lit("2024-08-01")))\
      .select(regexp_extract(col("bpuic").cast("string"), "(\\d+)", 1).alias("bpuic"))\
      .distinct()

# %% vscode={"languageId": "shellscript"}
stops_lausanne = stops_lausanne.select(\
          col("stop_id"),
          col("stop_name"),
          col("stop_lat"),
          col("stop_lon"),
          regexp_extract(col("stop_id"), "(\\d+)", 1).alias("bpuic")\
      ).distinct()

# %%
#stops_lausanne.show(3)

# %%
#stops_lausanne.count()

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
NUM_PARTITIONS = 256

spark.conf.set("spark.sql.shuffle.partitions", NUM_PARTITIONS) 

pairwise_distances = stops_lausanne.repartition(NUM_PARTITIONS).alias("a").crossJoin(stops_lausanne.alias("b")) \
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


stop_to_stop_df = stop_to_stop_df.checkpoint()
stop_to_stop_df.show(5)
stop_to_stop_df.printSchema()
stop_to_stop_df.count()


# %%
from pyspark.sql.functions import to_timestamp

lausanne_stop_times = joined_df.withColumn(
    "departure_td", to_timestamp("departure_time", "HH:mm:ss")
).withColumn(
    "arrival_td", to_timestamp("arrival_time", "HH:mm:ss")
)

# %%
# Time dependent edges + transfer times (USE THIS)
# takes max 1 mins to build graph



# Nodes are unique stops
# Edges are time dependent



# public transport (PT) edge has a similar sturcture to:
# G_stops.edges[A, B] = {
#     'pt_connections': [
#         {
#             'trip_id': '1234_ABC',
#             'departure_td': timedelta(hours=8, minutes=15, seconds=0),
#             'arrival_td': timedelta(hours=8, minutes=30, seconds=0),
#             'type': 'PT'
#         },
#         ...
#     ]
# }


# walking edges on the other hand look like:
# G_stops.edges[A, B] = {
#     'walk_info': {
#         'walk_time_minutes': 2.3,
#         'distance_m': 115,
#         'type': 'WALK'
#     }
# }

# An edge can contain both 'pt_connections' and 'walk_info' if both PT and walk options exist between two stops.




import networkx as nx
from datetime import timedelta
from collections import defaultdict
import math

# data
# Collecting data from Spark 

# transfers data
transfers = spark.table("iceberg.sbb.transfers")
# Filter transfers for Lausanne region stops
transfers_lausanne = transfers.join(
    stops_lausanne.select("stop_id").alias("from_stops"),
    transfers.from_stop_id == col("from_stops.stop_id"),
    "inner"
).join(
    stops_lausanne.select("stop_id").alias("to_stops"),
    transfers.to_stop_id == col("to_stops.stop_id"),
    "inner"
).select(
    "from_stop_id",
    "to_stop_id",
    "min_transfer_time"
)


# creating transfer lookup
transfers_lookup = {}
transfer_rows = transfers_lausanne.collect()
for row in transfer_rows:
    key = (row.from_stop_id, row.to_stop_id)
    transfers_lookup[key] = row.min_transfer_time
print(f"Loaded {len(transfers_lookup)} explicit transfer definitions")

stop_coords_dict = {}
stop_coords_rows = stops_lausanne.select("stop_id", "stop_lat", "stop_lon").collect()
for row in stop_coords_rows:
    stop_coords_dict[row.stop_id] = (row.stop_lat, row.stop_lon)
print(f"Created coordinate lookup for {len(stop_coords_dict)} stops")

# stop times
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

# stops list
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



G_stops = nx.DiGraph()

# adding all unique stop_ids as nodes
stop_ids_for_nodes = set() 

for stop_id in stop_coords_dict.keys():
    G_stops.add_node(stop_id)
    stop_ids_for_nodes.add(stop_id)
print(f"Added {len(G_stops.nodes())} initial nodes from stop_coords_dict.")


# Processing Public Transport Edges
if trip_groups_graph:
    print("Building PT connections on stop-graph...")
    for trip_id, group in trip_groups_graph.items():
        sorted_group = sorted(group, key=lambda x: x["departure_td"])
        for i in range(len(sorted_group) - 1):
            stop_a_data = sorted_group[i]
            stop_b_data = sorted_group[i+1]

            from_stop_id = stop_a_data['stop_id']
            to_stop_id = stop_b_data['stop_id']

            # making sure nodes exist
            if from_stop_id not in stop_ids_for_nodes:
                G_stops.add_node(from_stop_id)
                stop_ids_for_nodes.add(from_stop_id)
            if to_stop_id not in stop_ids_for_nodes:
                G_stops.add_node(to_stop_id)
                stop_ids_for_nodes.add(to_stop_id)

            travel_time_delta = stop_b_data['arrival_td'] - stop_a_data['departure_td']
            if travel_time_delta.total_seconds() < 0:
                # print(f"Warning: Negative travel time in trip {trip_id} from {from_stop_id} to {to_stop_id}. Skipping.")
                continue

            connection_info = {
                'departure_td': stop_a_data['departure_td'],
                'arrival_td': stop_b_data['arrival_td'],
                'trip_id': trip_id,
                'type': 'PT'
            }

            if G_stops.has_edge(from_stop_id, to_stop_id):
                # If the edge already exists, append this PT trip info to the pt_connections list
                if 'pt_connections' not in G_stops.edges[from_stop_id, to_stop_id]:
                    G_stops.edges[from_stop_id, to_stop_id]['pt_connections'] = []
                G_stops.edges[from_stop_id, to_stop_id]['pt_connections'].append(connection_info)
            else:
                G_stops.add_edge(from_stop_id, to_stop_id, pt_connections=[connection_info])

    # sorting all pt_connections lists by departure time
    for u, v, data in G_stops.edges(data=True):
        if 'pt_connections' in data:
            data['pt_connections'].sort(key=lambda x: x['departure_td'])
    print("Finished building PT connections.")
else:
    print("No PT trips to process for stop-graph building.")


# processing Walking Edges
WALKING_SPEED_MPS = 50 / 60 # meters per second

if stop_to_stop_list:
    print(f"Building walking connections on stop-graph. Walk definitions: {len(stop_to_stop_list)}")
    for walk_info_row in stop_to_stop_list:
        try:
            stop_a_id = walk_info_row['stop_id_a']
            stop_b_id = walk_info_row['stop_id_b']
            distance_meters = float(walk_info_row['distance'])
        except (TypeError, KeyError, ValueError, AttributeError) as e:
            print(f"Skipping malformed walk_info_row: {walk_info_row} due to {e}")
            continue

        if distance_meters < 0: continue
        if WALKING_SPEED_MPS <= 0: continue # Avoid division by zero

        walking_time_seconds = distance_meters / WALKING_SPEED_MPS
        walking_time_minutes = walking_time_seconds / 60.0

        # makign sure nodes exist
        if stop_a_id not in stop_ids_for_nodes: G_stops.add_node(stop_a_id); stop_ids_for_nodes.add(stop_a_id)
        if stop_b_id not in stop_ids_for_nodes: G_stops.add_node(stop_b_id); stop_ids_for_nodes.add(stop_b_id)

        walk_info = {
            'walk_time_minutes': walking_time_minutes,
            'distance_m': distance_meters,
            'type': 'WALK'
        }
        if G_stops.has_edge(stop_a_id, stop_b_id):
            # if edge exists, add/update walk info (but we prefer shorter walk if multiple definitions)
            if 'walk_info' not in G_stops.edges[stop_a_id, stop_b_id] or \
               G_stops.edges[stop_a_id, stop_b_id]['walk_info']['walk_time_minutes'] > walking_time_minutes:
                G_stops.edges[stop_a_id, stop_b_id]['walk_info'] = walk_info
        else:
            G_stops.add_edge(stop_a_id, stop_b_id, walk_info=walk_info)

        # adding walk in reverse direction too 
        if G_stops.has_edge(stop_b_id, stop_a_id):
            if 'walk_info' not in G_stops.edges[stop_b_id, stop_a_id] or \
               G_stops.edges[stop_b_id, stop_a_id]['walk_info']['walk_time_minutes'] > walking_time_minutes:
                G_stops.edges[stop_b_id, stop_a_id]['walk_info'] = walk_info
        else:
            G_stops.add_edge(stop_b_id, stop_a_id, walk_info=walk_info)

    print("Finished building walking connections.")
else:
    print("Warning: 'stop_to_stop_list' is empty. No walking definitions to process for stop-graph.")

print(f"\n=== Final Stop-Graph Statistics ===")
print(f"Total Nodes (Stops) in Graph: {len(G_stops.nodes())}")
print(f"Total Edges (Stop-to-Stop relationships) in Graph: {len(G_stops.edges())}")

# counting PT connections and Walk edges for check!
pt_connection_count = 0
walk_edge_count = 0
for _, _, data in G_stops.edges(data=True):
    if 'pt_connections' in data:
        pt_connection_count += len(data['pt_connections'])
    if 'walk_info' in data:
        walk_edge_count += 1
print(f"Total individual PT connections: {pt_connection_count}")
print(f"Total walk-enabled edges: {walk_edge_count}")
print(f"\nStop-graph building complete!")

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
      .persist(StorageLevel.DISK_ONLY))

hist_ist.printSchema()

# %%
# check the time range filter
# hist_ist.select(hour(col("dep_actual")).alias("h")).distinct().orderBy("h").show()

# %%
# sanity check, 1% of data sample
# sample = hist_ist.sample(fraction=0.01, seed=42)

# print("Sample size:", sample.count())

# # compute a few approximate quantiles
# arr_q = sample.approxQuantile("arr_delay_s", [0.1, 0.5, 0.9], 0.01)
# dep_q = sample.approxQuantile("dep_delay_s", [0.1, 0.5, 0.9], 0.01)
# print(f"arr_delay_s quantiles (10%,50%,90%): {arr_q}")
# print(f"dep_delay_s quantiles (10%,50%,90%): {dep_q}")
# sample.unpersist()

# sample.select("operating_day","arr_delay_s","dep_delay_s").show(5, truncate=False)

# %%
# hourly delays percentile
hourly_stats = (
    hist_ist.sample(withReplacement=False,fraction=0.3, seed=42) 
      .withColumn("dep_hour", hour(col("dep_actual")))
      .groupBy("dep_hour")
      .agg(
        percentile_approx(col("arr_delay_s"), [0.5,0.75,0.9,0.95]).alias("arr_qs"),
        #percentile_approx(col("dep_delay_s"), [0.5,0.75,0.9,0.95]).alias("dep_qs"),
        F.count("*").alias("n_trips"))
      .orderBy("dep_hour")
      .persist(StorageLevel.DISK_ONLY))

# trigger 
#hourly_stats.head(1)

# %%
#sanity check
# are all 24 hours present?
# distinct_hours = hourly_stats.select("dep_hour").distinct().orderBy("dep_hour")
# print("Hours covered:", [r.dep_hour for r in distinct_hours.collect()])
# hourly_stats.show(24, truncate=False)

# %% [markdown]
# ## Feature engineering

# %%
# time features
hist_feat = (
    hist_ist.sample(withReplacement=False, fraction=0.5, seed=42)
        .withColumn("sched_dep_hour", hour(to_timestamp(col("dep_time"), "yyyy-MM-dd HH:mm:ss")))
        .withColumn("act_dep_hour", hour(to_timestamp(col("dep_actual"), "yyyy-MM-dd HH:mm:ss")))
        .withColumn("dow",((dayofweek(col("operating_day")) + 5) % 7) + 1) #days of week
        .withColumn("day_of_year",dayofyear(col("operating_day")))).select(
            "bpuic",
            "trip_id",
            "arr_time",
            "dep_time",
            "arr_delay_s",
            "sched_dep_hour",
            "act_dep_hour",
            "dow",
            "day_of_year")

    
hist_feat.printSchema()

hourly_stats.unpersist()
hist_ist.unpersist()
#trigger (for cache)
#hist_feat.head(1)

# %%
# #sanity check
# sample_feat = hist_feat.sample(fraction=0.01, seed=42)
# sample_feat.printSchema()

# print("\nSample rows with time features:")
# sample_feat.select("dep_time","dep_actual","sched_dep_hour","act_dep_hour","operating_day","dow").limit(5).show(truncate=False)

# hours = sorted(r.sched_dep_hour for r in sample_feat.select("sched_dep_hour").distinct().collect())
# days  = sorted(r.dow for r in sample_feat.select("dow").distinct().collect())
# print(f"\nsched_dep_hour values in sample: {hours}")
# print(f"dow values in sample: {days}")


# %%
# take interesting column
## Added a subsampling to reduce memory pressure
hist_fact = hist_feat.select(
    "bpuic",
    "trip_id",
    "arr_time",
    "dep_time",
    "arr_delay_s",
    "sched_dep_hour",
    "act_dep_hour",
    "dow",
    "day_of_year").persist(StorageLevel.DISK_ONLY)

hist_feat.unpersist()

# %%
# stops_geo: bpuic, name, coords
stops_geo = (spark.table("iceberg.sbb.stops")
         .withColumn("bpuic", regexp_extract("stop_id", "(\\d+)", 1).cast("int"))
         .select("bpuic", "stop_lat", "stop_lon"))

# trip and route
trips_df  = spark.table("iceberg.sbb.trips").select("trip_id", "route_id")
routes_df = spark.table("iceberg.sbb.routes").select("route_id", "route_type")

# %%
from pyspark.sql.functions import broadcast

# hist_route_station = (
#     hist_fact
#       .join(stops_geo, on="bpuic", how="left")
#       .join(trips_df, on="trip_id", how="left")
#       .join(routes_df, on="route_id", how="left")
#       .withColumn("scheduled_tt", unix_timestamp("arr_time") - unix_timestamp("dep_time"))
#       .withColumn("delay", col("arr_delay_s"))
#       .persist(StorageLevel.DISK_ONLY))

# hist_route_station = (hist_fact
#       .join(stops_geo, on="bpuic", how="left")
#       .join(trips_df, on="trip_id", how="left")
#       .join(routes_df, on="route_id", how="left")
#       .withColumn("scheduled_tt", unix_timestamp("arr_time") - unix_timestamp("dep_time"))
#       .withColumn("delay", col("arr_delay_s"))                       
#       .select("trip_id","bpuic","delay","route_id","route_desc","route_type",
#               "stop_lat","stop_lon","scheduled_tt","arr_delay_s",
#               "sched_dep_hour","act_dep_hour","dow","day_of_year"))


stops_sel  = stops_geo.select("bpuic", "stop_lat", "stop_lon")
trips_sel  = trips_df.select("trip_id", "route_id")
routes_sel = routes_df.select("route_id", "route_type")

hist_route_station = (
    hist_fact
      .join(trips_sel, "trip_id", "left")
      .join(stops_sel,  "bpuic", "left")
      .join(routes_sel, "route_id", "left")
      .withColumn(
          "scheduled_tt",
          unix_timestamp("arr_time") - unix_timestamp("dep_time")
      )
      .withColumnRenamed("arr_delay_s", "delay")
      .select(
          "trip_id", "bpuic", "delay",
          "route_id", "route_type",
          "stop_lat", "stop_lon",
          "scheduled_tt",
          "sched_dep_hour", "act_dep_hour", "dow", "day_of_year"
      )
)


# hist_route_station.count() # Materialize and cache
hist_route_station.createOrReplaceTempView("hist_route_station")
hist_fact.unpersist()


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
      .join(broadcast(transfer_stats), hist_route_station.bpuic == transfer_stats.from_bpuic, how="left")
      .drop("from_bpuic")
      .na.fill({"min_transfer_time": 2, "transfer_degree": 0}))

hist_route_station.unpersist()

# weather
feat_with_weather = (
    feat_with_transfer
        # .withColumn("temperature", lit(0.0))
        # .withColumn("precipitation", lit(0.0))
        # .withColumn("wind_speed", lit(0.0))
        .withColumn("hour_sin", sin(2 * lit(math.pi) * col("sched_dep_hour") / lit(24)))
        .withColumn("hour_cos", cos(2 * lit(math.pi) * col("sched_dep_hour") / lit(24)))
        .withColumn("doy_sin", sin(2 * lit(math.pi) * col("day_of_year") / lit(365)))
        .withColumn("doy_cos", cos(2 * lit(math.pi) * col("day_of_year") / lit(365))))

feat_with_transfer.unpersist()

# final features col
# final_features = (
#     feat_with_weather
#       .select(
#          "trip_id", "bpuic", "route_id", "route_desc", "route_type",
#          "scheduled_tt", "delay", "sched_dep_hour", "act_dep_hour", "dow", 
#          "day_of_year", "hour_sin", "hour_cos", "doy_sin", "doy_cos", 
#          "stop_lat", "stop_lon", "transfer_degree", "min_transfer_time",
#          "temperature", "precipitation", "wind_speed").dropDuplicates())


# final_features = (
#     feat_with_weather
#       .select(
#          "trip_id", "bpuic", "route_id", "route_desc", "route_type",
#          "scheduled_tt", "delay", "sched_dep_hour", "act_dep_hour", "dow", 
#          "day_of_year", "hour_sin", "hour_cos", "doy_sin", "doy_cos", 
#          "stop_lat", "stop_lon", "transfer_degree", "min_transfer_time",
#          "temperature", "precipitation", "wind_speed"))

final_features = (
    feat_with_weather
      .select(
         "trip_id", "bpuic", "route_id", "route_type",
         "scheduled_tt", "delay", "sched_dep_hour", "act_dep_hour", "dow", 
         "day_of_year", "hour_sin", "hour_cos", "doy_sin", "doy_cos", 
         "stop_lat", "stop_lon", "transfer_degree", "min_transfer_time"))

final_features.createOrReplaceTempView("segment_features")
feat_with_weather.unpersist()

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
)

baseline_qs.createOrReplaceTempView("baseline_quantiles")

# %%
baseline_qs.printSchema()

# %% [markdown]
# ## Hybrid Delay Model
#
# #### Students Note to TAs
# The hybrid delay model could not be trained due to memory limits on the cluster. It seems that we are able to build the
# final dataset yet during the process of training the model, the spark driver is not able to write some data into memory
# and it looses RDDs. As a result the process restarts and the cycle continues. We tried all the suggestions from the TA
# regarding using the new garbage collector and also subsampling the dataset. However, we ran into issues with even just
# 30% of the dataset. We also tried to unpersist some dataframes that were no longer needed but we still could not train
# the model. I think if we had access to more RAM we would be able to train the model and perform hyperparameter tuning.
# The code below automatically loads the best model from the HDFS store if it exists and skips the training process. If no
# model is found the code will retrain the model, perform hyperparameter tuning, and save it to the HDFS store. We really
# did try everything we could think of but still nothing seemed to work. However with more RAM our code should be able to run.
#

# %%
from hashlib import sha256
from pyspark.ml import PipelineModel

TRAINING = True


if isinstance(target_day_of_week, list):
    day_of_week = sorted(["".join(x.lower().split()) for x in target_day_of_week])
else:
    day_of_week = [target_day_of_week.lower()]


cleaned_region_names = sorted(["".join(x.lower().split()) for x in region_names])
joined_region_names = "".join(cleaned_region_names + day_of_week)
region_hash = sha256(joined_region_names.encode('utf-8')).hexdigest()
model_hdfs_path = f"{hadoopFS}/user/com-490/group/{groupName}/{region_hash}/best_model"

print(f"Attempting to load the RandomForest PipelineModel from: {model_hdfs_path}")

try:
    # Load the PipelineModel
    resid_model = PipelineModel.load(model_hdfs_path)
    print(f"PipelineModel loaded successfully from {model_hdfs_path}")
    TRAINING = False
except Exception as e:
    print(f"Could not find a PipelineModel to load for the regions {region_names} at {model_hdfs_path}")
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
    broadcast(baseline_qs.select("route_id","bpuic","sched_dep_hour","q50","q75","q90","q95")),
    on=["route_id","bpuic","sched_dep_hour"], how="left").withColumn("resid_label", col("delay") - col("q50"))


# ## This allows us to save the RDDs to HDFS. The count is just there to let SPARK know how to query this again if it ever gets lost.
# hist_ist = hist_ist.checkpoint()        # writes a safe copy to HDFS
# hist_ist.count()
# hist_feat = hist_feat.checkpoint()
# hist_feat.count()
# final_features = final_features.checkpoint()
# final_features.count()

## We unpersist some RDDs to free up some space here
# final_features.unpersist()
# hist_feat.unpersist()
# hist_ist.unpersist()

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

feature_cols = [
    "scheduled_tt","sched_dep_hour","act_dep_hour",
    "dow","day_of_year","hour_sin","hour_cos","doy_sin","doy_cos",
    "stop_lat","stop_lon","transfer_degree","min_transfer_time",
    "q50"]  # include baseline median


assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

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
# ### Hyperparameter Tuning and Validation
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

    cv_rf = CrossValidator(estimator=pipeline,
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

# # Saving the best model
# if TRAINING:
#     cleaned_region_names = sorted(["".join(x.lower().split()) for x in region_names])
#     joined_region_names = "".join(cleaned_region_names)
#     region_hash = sha256(joined_region_names.encode('utf-8')).hexdigest()

#     model_hdfs_path = f"{hadoopFS}/user/com-490/group/{GroupName}/{region_hash}/best_model"
#     print(f"Attempting to save the trained RandomForest PipelineModel to: {model_path_rf}")

#     try:
#         # To overwrite if the model path already exists
#         bestPipelineModel_rf.write().overwrite().save(model_path_rf)
#         print(f"PipelineModel saved successfully to {model_path_rf}")
#     except Exception as e:
#         print(f"Error saving model: {e}")
 

# Saving the best model
if TRAINING:
    if isinstance(target_day_of_week, list):
        day_of_week = sorted(["".join(x.lower().split()) for x in target_day_of_week])
    else:
        day_of_week = [target_day_of_week.lower()]

    cleaned_region_names = sorted(["".join(x.lower().split()) for x in region_names])
    joined_region_names = "".join(cleaned_region_names + day_of_week)
    region_hash = sha256(joined_region_names.encode('utf-8')).hexdigest()

    model_hdfs_path = f"{hadoopFS}/user/com-490/group/{groupName}/{region_hash}/best_model"
    print(f"Attempting to save the trained RandomForest PipelineModel to: {model_hdfs_path}")

    try:
        # To overwrite if the model path already exists
        resid_model.write().overwrite().save(model_hdfs_path)
        print(f"PipelineModel saved successfully to {model_hdfs_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

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

# %% [markdown]
# # IV. Route Planning Algorithm
# %%
# Optimized version with transfer logic 
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date as DateObject 
from heapq import heappush, heappop 
import math
from collections import defaultdict, deque
from IPython.display import display, clear_output
import ipywidgets as widgets
import plotly.express as px
import plotly.graph_objects as go
import bisect

class OptimizedRobustJourneyPlanner:
    def __init__(self, stop_graph, transfers_lookup, stop_coords_dict,
                 delay_model=None, max_walking_distance_total=1500,
                 walking_speed_mps=50/60):
        self.graph = stop_graph
        self.transfers_lookup = transfers_lookup
        self.stop_coords_dict = stop_coords_dict
        self.delay_model = delay_model
        self.max_walking_distance_total = max_walking_distance_total
        self.walking_speed_mps = walking_speed_mps
        
        # pre-computing neighbor data for faster access
        self._precompute_neighbor_data()
        
        # this is a cache for transfer times to avoid repeated calculations
        self._transfer_time_cache = {}

    def _precompute_neighbor_data(self):
        self.neighbor_data = {}
        for node in self.graph.nodes():
            self.neighbor_data[node] = []
            for successor in self.graph.successors(node):
                edge_data = self.graph.get_edge_data(node, successor)
                if edge_data:
                    self.neighbor_data[node].append((successor, edge_data))

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
        R = 6371000
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

    # DUMMY FOR NOW
    # Currently assuming that there is no delay 
    def _get_delay_distribution(self, trip_id, dep_stop_id, dep_time_td, day_of_week=None):
        if self.delay_model is None: return {"q50": 0, "q75": 0, "q90": 0, "q95": 0}
        return {"q50": 0, "q75": 0, "q90": 0, "q95": 0}

    # also cached so that we can be efficient with transfer time calc!
    def _calculate_transfer_time_minutes(self, from_stop_id, to_stop_id):
        cache_key = (from_stop_id, to_stop_id)
        if cache_key in self._transfer_time_cache:
            return self._transfer_time_cache[cache_key]
            
        transfer_key = (from_stop_id, to_stop_id)
        if transfer_key in self.transfers_lookup:
            result = self.transfers_lookup[transfer_key] / 60.0
            self._transfer_time_cache[cache_key] = result
            return result

        if from_stop_id == to_stop_id:
            result = 2.0
            self._transfer_time_cache[cache_key] = result
            return result

        if from_stop_id in self.stop_coords_dict and to_stop_id in self.stop_coords_dict:
            from_lat, from_lon = self.stop_coords_dict[from_stop_id]
            to_lat, to_lon = self.stop_coords_dict[to_stop_id]

            distance = haversine_distance_python(from_lat, from_lon, to_lat, to_lon)
            if distance is None:
                self._transfer_time_cache[cache_key] = None
                return None

            MAX_IMPLICIT_TRANSFER_WALK_M = 500
            if distance <= MAX_IMPLICIT_TRANSFER_WALK_M:
                additional_time = distance / 50.0
                result = 2.0 + additional_time
                self._transfer_time_cache[cache_key] = result
                return result
            else:
                self._transfer_time_cache[cache_key] = None
                return None
        
        self._transfer_time_cache[cache_key] = None
        return None

    # binary searching for a solid connection
    def _find_next_pt_connection(self, connections, earliest_departure_td):
        if not connections: return None
        dep_times = [c['departure_td'] for c in connections]
        idx = bisect.bisect_left(dep_times, earliest_departure_td)
        if idx < len(connections):
            return connections[idx]
        return None

    # double checking that our routes arent loops
    def _has_significant_physical_stop_loop(self, path_segments_list):
        if not path_segments_list or len(path_segments_list) < 2:
            return False

        # visited stops
        visited_stops = [path_segments_list[0][0]]  
        visited_stops.extend(seg[1] for seg in path_segments_list)  

        # loop detection algo 
        seen_at_position = {}
        for i, stop_id in enumerate(visited_stops):
            if stop_id in seen_at_position:
                prev_position = seen_at_position[stop_id]
                if i - prev_position > 1: 
                    for j in range(prev_position + 1, i):
                        if visited_stops[j] != stop_id:
                            return True
            seen_at_position[stop_id] = i
        return False

    def find_robust_paths(self, source_stop_id, target_stop_id,
                         departure_time_td=None,
                         latest_arrival_constraint_td=None,
                         confidence_level=0.75,
                         max_options=3):

        num_options_desired = max_options
        if departure_time_td is None and latest_arrival_constraint_td is None:
            print("Error: At least departure time or arrival constraint must be specified.")
            return []

        # early termination for long distances
        # so basically we stop exploring paths that have already become significantly longer or more costly 
        if source_stop_id == target_stop_id:
            return self._handle_same_stop_case(source_stop_id, departure_time_td, latest_arrival_constraint_td)

        # but double checking if this is a long-distance query and adjusting parameters
        is_long_distance = self._estimate_journey_distance(source_stop_id, target_stop_id)
        
        if is_long_distance:
            # reducing search space for long distances
            max_iterations = 100000  
            early_target_hits = max(3, num_options_desired)  #earlier stop
        else:
            max_iterations = 750000
            early_target_hits = num_options_desired * 10

        effective_earliest_departure_td = departure_time_td
        search_mode = "departure"
        if departure_time_td is None and latest_arrival_constraint_td is not None:
            # While exploring our data, we noticed that the farthest routes took max 2 hours within the same region.
            # Thus, we did this.
            print("Warning: Only arrival constraint specified. Using a wide departure window for forward search (2h earlier).")
            effective_earliest_departure_td = latest_arrival_constraint_td - timedelta(hours=2)
            if effective_earliest_departure_td < timedelta(0): 
                effective_earliest_departure_td = timedelta(0)
            search_mode = "arrival_constraint"
        elif departure_time_td is None:
             effective_earliest_departure_td = timedelta(0)

        confidence_map = {0.5: "q50", 0.75: "q75", 0.9: "q90", 0.95: "q95"}
        quantile_key = confidence_map.get(confidence_level, "q75")

        # Priority queue
        queue = []
        heappush(queue, (effective_earliest_departure_td.total_seconds(), 0.0, source_stop_id, []))

        # tracking best arrival time and walking distance for each stop
        visited_states = {}  # so stop_id -> (best_arrival_time_sec, min_walk_dist)
        
        completed_options_raw = []
        iterations = 0
        target_hit_count = 0

        while queue and target_hit_count < early_target_hits and iterations < max_iterations:
            iterations += 1
            
            current_arrival_sec, current_total_walk_dist_m, current_stop_id, current_path_segments = heappop(queue)
            arrival_at_current_stop_td = timedelta(seconds=current_arrival_sec)

            # pruning with state comparison
            state_key = current_stop_id
            if state_key in visited_states:
                best_time_sec, min_walk_dist = visited_states[state_key]
                # we're gonna continue only if this state is better in time or significantly better in walking
                if (current_arrival_sec > best_time_sec + 300 and  # 5 minute MAX tolerance
                    current_total_walk_dist_m >= min_walk_dist):
                    continue
                # updating if this is better
                if (current_arrival_sec < best_time_sec or 
                    (current_arrival_sec <= best_time_sec + 60 and current_total_walk_dist_m < min_walk_dist)):
                    visited_states[state_key] = (current_arrival_sec, current_total_walk_dist_m)
            else:
                visited_states[state_key] = (current_arrival_sec, current_total_walk_dist_m)

            # did we reach our target dest?
            if current_stop_id == target_stop_id:
                if not current_path_segments:
                    actual_departure_td = arrival_at_current_stop_td
                    actual_arrival_td = arrival_at_current_stop_td
                else:
                    first_seg = current_path_segments[0]
                    actual_departure_td = first_seg[2]
                    actual_arrival_td = arrival_at_current_stop_td

                # applying the input constraints (dep time, arrival time, etc)
                if departure_time_td is not None and actual_departure_td < departure_time_td:
                    continue
                if latest_arrival_constraint_td is not None and actual_arrival_td > latest_arrival_constraint_td:
                    continue
                
                total_travel_time_minutes = (actual_arrival_td - actual_departure_td).total_seconds() / 60.0
                if total_travel_time_minutes < -1e-7: 
                    continue

                if self._has_significant_physical_stop_loop(current_path_segments):
                    continue

                option_details = {
                    "departure_stop": source_stop_id, "departure_time": actual_departure_td,
                    "arrival_stop": target_stop_id, "arrival_time": actual_arrival_td,
                    "travel_time_minutes": total_travel_time_minutes,
                    "walking_distance_meters": current_total_walk_dist_m,
                    "transfers": self._count_transfers_from_segments(current_path_segments),
                    "path_segments": current_path_segments,
                    "confidence_level": self._quantile_key_to_confidence(quantile_key)
                }
                completed_options_raw.append(option_details)
                target_hit_count += 1
                continue

            # pruning for arrival constraint
            if latest_arrival_constraint_td and arrival_at_current_stop_td > latest_arrival_constraint_td + timedelta(minutes=60):
                continue

            # exploring neighbors using our pre-computed data
            if current_stop_id not in self.neighbor_data:
                continue

            for successor_stop_id, edge_data in self.neighbor_data[current_stop_id]:
                # processing PT connections
                if 'pt_connections' in edge_data:
                    min_transfer_wait_minutes = 0
                    if current_path_segments:
                        last_segment = current_path_segments[-1]
                        if last_segment[4] == 'PT':
                            min_transfer_wait_minutes = self._calculate_transfer_time_minutes(
                                last_segment[1], current_stop_id) or 2.0
                    
                    earliest_pt_boarding_td = arrival_at_current_stop_td + timedelta(minutes=min_transfer_wait_minutes)
                    connection = self._find_next_pt_connection(edge_data['pt_connections'], earliest_pt_boarding_td)

                    if connection:
                        trip_id_seg = connection['trip_id']
                        segment_scheduled_departure_td = connection['departure_td']
                        segment_scheduled_arrival_td = connection['arrival_td']

                        # just in case loop check for immediate turnaround
                        if (current_path_segments and len(current_path_segments) >= 1):
                            prev_seg = current_path_segments[-1]
                            if (prev_seg[4] == 'PT' and prev_seg[5] == trip_id_seg and 
                                prev_seg[0] == successor_stop_id):
                                continue

                        delay_on_segment_seconds = 0
                        if self.delay_model:
                             delay_dist = self._get_delay_distribution(trip_id_seg, current_stop_id, segment_scheduled_departure_td)
                             delay_on_segment_seconds = delay_dist.get(quantile_key, 0)
                        
                        actual_arrival_at_successor_td = segment_scheduled_arrival_td + timedelta(seconds=delay_on_segment_seconds)

                        if latest_arrival_constraint_td and actual_arrival_at_successor_td > latest_arrival_constraint_td:
                            continue
                        
                        new_segment = (current_stop_id, successor_stop_id, segment_scheduled_departure_td, 
                                     actual_arrival_at_successor_td, 'PT', trip_id_seg)
                        new_path_segments = current_path_segments + [new_segment]

                        # quick loop check before expensive full check
                        # our algo was kinda slow so we wanted to try and speed it up by doing small check first 
                        if len(new_path_segments) > 2 and self._has_significant_physical_stop_loop(new_path_segments):
                            continue

                        heappush(queue, (actual_arrival_at_successor_td.total_seconds(),
                                       current_total_walk_dist_m,
                                       successor_stop_id,
                                       new_path_segments))

                # processing walking connections
                if 'walk_info' in edge_data:
                    walk_data = edge_data['walk_info']
                    walk_time_minutes = walk_data['walk_time_minutes']
                    walk_distance_m = walk_data['distance_m']

                    if current_total_walk_dist_m + walk_distance_m > self.max_walking_distance_total:
                        continue

                    walk_departure_td = arrival_at_current_stop_td
                    walk_arrival_at_successor_td = walk_departure_td + timedelta(minutes=walk_time_minutes)

                    if latest_arrival_constraint_td and walk_arrival_at_successor_td > latest_arrival_constraint_td:
                        continue

                    new_segment = (current_stop_id, successor_stop_id, walk_departure_td, 
                                 walk_arrival_at_successor_td, 'WALK', walk_distance_m)
                    new_path_segments = current_path_segments + [new_segment]
                    
                    if len(new_path_segments) > 2 and self._has_significant_physical_stop_loop(new_path_segments):
                        continue

                    heappush(queue, (walk_arrival_at_successor_td.total_seconds(),
                                   current_total_walk_dist_m + walk_distance_m,
                                   successor_stop_id,
                                   new_path_segments))

        if iterations >= max_iterations:
            print(f"Warning: Dijkstra search reached max iterations ({max_iterations}). Results might be incomplete.")
        
        if not completed_options_raw: 
            return []

        return self._process_and_filter_options(completed_options_raw, num_options_desired, departure_time_td)

    # is this a long distance journey?
    def _estimate_journey_distance(self, source_stop_id, target_stop_id):
        if (source_stop_id in self.stop_coords_dict and 
            target_stop_id in self.stop_coords_dict):
            src_lat, src_lon = self.stop_coords_dict[source_stop_id]
            tgt_lat, tgt_lon = self.stop_coords_dict[target_stop_id]
            
            # Manhattan distance in degrees for dist esimate
            lat_diff = abs(float(tgt_lat) - float(src_lat))
            lon_diff = abs(float(tgt_lon) - float(src_lon))
            
            # long distance == ~50km or more so roughly 0.5 degrees 
            return (lat_diff + lon_diff) > 0.5
        return False

    # if source and target are the same, do this as a default
    def _handle_same_stop_case(self, stop_id, departure_time_td, latest_arrival_constraint_td):
        effective_time = departure_time_td or latest_arrival_constraint_td or timedelta(0)
        return [{
            "departure_stop": stop_id, "departure_time": effective_time,
            "arrival_stop": stop_id, "arrival_time": effective_time,
            "travel_time_minutes": 0.0, "walking_distance_meters": 0.0,
            "transfers": 0, "path_segments": [],
            "confidence_level": 0.75
        }]

    # filtering loops 
    def _process_and_filter_options(self, completed_options_raw, num_options_desired, departure_time_td):
        non_loopy_options = [opt for opt in completed_options_raw 
                           if not self._has_significant_physical_stop_loop(opt["path_segments"])]
        
        if not non_loopy_options and completed_options_raw:
            print("Warning: All raw options had loops. Consider refining loop detection or relaxing constraints.")
            non_loopy_options = completed_options_raw 
        if not non_loopy_options: 
            return []

        # sorting by multiple criteria efficiently
        non_loopy_options.sort(key=lambda x: (x["arrival_time"], x["departure_time"], x["transfers"]))
        
        final_selected_options = []
        selected_path_footprints = set()

        for option in non_loopy_options:
            if len(final_selected_options) >= num_options_desired:
                break

            # more diversity
            dep_bucket = int(option["departure_time"].total_seconds() / (15 * 60))
            pt_trips_in_path = [seg[5] for seg in option["path_segments"] 
                              if seg[4] == 'PT'][:2]  # First 2 PT trips
            footprint = (dep_bucket, option["transfers"], tuple(pt_trips_in_path))

            if footprint not in selected_path_footprints:
                if departure_time_td is not None and option["departure_time"] < departure_time_td:
                    continue
                final_selected_options.append(option)
                selected_path_footprints.add(footprint)
        
        # filling remaining slots if we can 
        if len(final_selected_options) < num_options_desired:
            existing_sigs = {self._get_path_signature(opt['path_segments']) 
                           for opt in final_selected_options}

            for option in non_loopy_options:
                if len(final_selected_options) >= num_options_desired:
                    break
                
                if (departure_time_td is not None and 
                    option["departure_time"] < departure_time_td):
                    continue

                current_sig = self._get_path_signature(option['path_segments'])
                if current_sig not in existing_sigs:
                    final_selected_options.append(option)
                    existing_sigs.add(current_sig)

        final_selected_options.sort(key=lambda x: (x["arrival_time"], x["departure_time"], x["transfers"]))
        return final_selected_options[:num_options_desired]

    # this is for checking if a path is unique or not
    def _get_path_signature(self, segments):
        sig = []
        for s in segments:
            item_id = s[5] if s[4] == 'PT' else int(s[5]) if isinstance(s[5], (float, int)) else str(s[5])
            sig.append((s[0], s[1], s[4], item_id))
        return tuple(sig)

    # HELPERS ! 
    def _count_transfers_from_segments(self, path_segments_list):
        if not path_segments_list: return 0
        transfers = 0
        last_pt_trip_id = None
        for seg in path_segments_list:
            seg_type = seg[4]
            if seg_type == 'PT':
                current_pt_trip_id = seg[5]
                if last_pt_trip_id is not None and current_pt_trip_id != last_pt_trip_id:
                    transfers += 1
                last_pt_trip_id = current_pt_trip_id
        return transfers

    def _quantile_key_to_confidence(self, quantile_key):
        quantile_map = {"q50": 0.5, "q75": 0.75, "q90": 0.9, "q95": 0.95}
        return quantile_map.get(quantile_key, 0.75)

    def _format_time(self, time_td):
        if not isinstance(time_td, timedelta): return "00:00:00"
        total_seconds = time_td.total_seconds(); sign = "-" if total_seconds < 0 else ""; total_seconds = abs(total_seconds)
        h = int(total_seconds // 3600); m = int((total_seconds % 3600) // 60); s = int(total_seconds % 60)
        return f"{sign}{h:02d}:{m:02d}:{s:02d}"

    def format_results(self, paths_list_of_dicts, stop_info_dict):
        formatted_paths = []
        if not stop_info_dict: print("Warning: stop_info not provided to format_results."); stop_info_dict = {}

        for i, path_result_dict in enumerate(paths_list_of_dicts):
            path_detail_segments = path_result_dict.get("path_segments", [])
            ui_segments = []

            if not path_detail_segments:
                 dep_stop_id = path_result_dict["departure_stop"]
                 arr_stop_id = path_result_dict["arrival_stop"]
                 if dep_stop_id == arr_stop_id:
                    curr_stop_name = stop_info_dict.get(dep_stop_id, {}).get("stop_name", dep_stop_id)
                    time_str = self._format_time(path_result_dict["departure_time"])
                    ui_segments.append({
                        "from_stop": curr_stop_name, "to_stop": curr_stop_name,
                        "departure": time_str, "arrival": time_str,
                        "duration_mins": 0.0, "type": "AT_STOP", "trip_id": "N/A"
                    })
            else:
                for seg_data in path_detail_segments:
                    from_stop_s, to_stop_s, dep_td_s, arr_td_s, type_s, id_s = seg_data

                    from_name = stop_info_dict.get(from_stop_s, {}).get("stop_name", from_stop_s)
                    to_name = stop_info_dict.get(to_stop_s, {}).get("stop_name", to_stop_s)
                    dep_str = self._format_time(dep_td_s)
                    arr_str = self._format_time(arr_td_s)
                    duration_s = (arr_td_s - dep_td_s).total_seconds() / 60.0
                    
                    trip_display_id = "N/A"
                    if type_s == 'PT':
                        transport_type_display = "TRANSIT"
                        trip_display_id = id_s
                    elif type_s == 'WALK':
                        transport_type_display = "WALK"
                        trip_display_id = f"{id_s:.0f}m"
                    else:
                        transport_type_display = type_s.upper()

                    ui_segments.append({
                        "from_stop": from_name, "to_stop": to_name,
                        "departure": dep_str, "arrival": arr_str,
                        "duration_mins": round(duration_s, 1),
                        "type": transport_type_display, "trip_id": trip_display_id
                    })

            raw_path_nodes_for_viz = []
            if path_detail_segments:
                first_seg_viz = path_detail_segments[0]
                raw_path_nodes_for_viz.append((first_seg_viz[0], first_seg_viz[2]))
                for seg_viz in path_detail_segments:
                    raw_path_nodes_for_viz.append((seg_viz[1], seg_viz[3]))
            elif path_result_dict["departure_stop"] == path_result_dict["arrival_stop"]:
                 raw_path_nodes_for_viz.append((path_result_dict["departure_stop"], path_result_dict["departure_time"]))

            route_summary = {
                "route_id": i + 1,
                "departure": self._format_time(path_result_dict["departure_time"]),
                "arrival": self._format_time(path_result_dict["arrival_time"]),
                "travel_time_mins": round(path_result_dict.get("travel_time_minutes", 0), 1),
                "transfers": path_result_dict.get("transfers", 0),
                "walking_distance": round(path_result_dict.get("walking_distance_meters", 0)),
                "confidence": f"{int(path_result_dict.get('confidence_level', 0.75) * 100)}%",
                "segments": ui_segments,
                "_raw_path_nodes": raw_path_nodes_for_viz
            }
            formatted_paths.append(route_summary)
        return formatted_paths

# UI STUFF
    def visualize_path(self, path_dict, stop_info):
            path_nodes_viz = path_dict.get("_raw_path_nodes")
            path_segments_detail = path_dict.get("segments") # The formatted UI segments
    
            if not path_nodes_viz or not stop_info:
                print("Warning: Cannot visualize path. Missing nodes or stop_info.")
                return go.Figure()
    
            stop_map_data_viz = []
            unique_stops_for_map_markers = {}
            for node_viz_stop_id, node_viz_time_td in path_nodes_viz:
                info_viz = stop_info.get(node_viz_stop_id)
                if info_viz and 'stop_lat' in info_viz and 'stop_lon' in info_viz:
                    if node_viz_stop_id not in unique_stops_for_map_markers:
                        stop_map_data_viz.append({
                            "stop_id": node_viz_stop_id,
                            "stop_name": info_viz.get("stop_name", node_viz_stop_id),
                            "lat": info_viz["stop_lat"],
                            "lon": info_viz["stop_lon"]
                        })
                        unique_stops_for_map_markers[node_viz_stop_id] = True
            
            if not stop_map_data_viz: print("Warning: No valid stops for visualization map markers."); return go.Figure()
            df_map = pd.DataFrame(stop_map_data_viz)
            if df_map.empty: print("Warning: Map DataFrame empty."); return go.Figure()
    
            path_lines_for_map = [] # lat/lon for lines
    
            original_segments_from_algo = path_dict.get("path_segments_original_for_viz_hack")
        
            for j in range(len(path_nodes_viz) - 1):
                curr_node_map_event = path_nodes_viz[j] # (stop_id, time_td)
                next_node_map_event = path_nodes_viz[j+1] # (stop_id, time_td)
    
                curr_sid_map = curr_node_map_event[0]
                next_sid_map = next_node_map_event[0]
    
                curr_geo = stop_info.get(curr_sid_map)
                next_geo = stop_info.get(next_sid_map)
    
                if not (curr_geo and 'stop_lat' in curr_geo and 'stop_lon' in curr_geo and \
                        next_geo and 'stop_lat' in next_geo and 'stop_lon' in next_geo):
                    continue # skipping if geo info missing
    
                # trynna figure out the segment type (Walk or Transit) for this leg
                # to do that we are looking for the corresponding path_segments_detail or original segments
                segment_type_for_line = "Transit" # default since most edges are transits
                if j < len(path_segments_detail): 
                    ui_seg_info = path_segments_detail[j]
                    # will look smth like: ui_seg_info: {"from_stop": from_name, "to_stop": to_name, ..., "type": "TRANSIT" or "WALK"}
                    if ui_seg_info["type"] == "WALK":
                        segment_type_for_line = "Walk"
                
                color = 'red' if segment_type_for_line == 'Walk' else 'blue'
                
                path_lines_for_map.append({
                    'lat': [float(curr_geo["stop_lat"]), float(next_geo["stop_lat"])],
                    'lon': [float(curr_geo["stop_lon"]), float(next_geo["stop_lon"])],
                    'segment_type': segment_type_for_line,
                    'color': color
                })
    
            # making a figure
            fig = px.scatter_mapbox(df_map, lat='lat', lon='lon', hover_name='stop_name',
                                    text=df_map['stop_name'].fillna('') if 'stop_name' in df_map.columns else None,
                                    size=[8]*len(df_map), zoom=11,
                                    center={"lat": df_map['lat'].mean(), "lon": df_map['lon'].mean()} if not df_map.empty else {"lat":46.52, "lon":6.63},
                                    mapbox_style="carto-positron")
            if 'stop_name' in df_map.columns: fig.update_traces(textposition='top right')
    
            for seg_line in path_lines_for_map:
                fig.add_trace(go.Scattermapbox(
                    lat=seg_line['lat'], lon=seg_line['lon'], mode='lines',
                    line=dict(width=4 if seg_line['segment_type']=='Transit' else 2, color=seg_line['color']),
                    name=seg_line['segment_type'], hoverinfo='skip'
                ))
            
            legend_names = set(); fig.for_each_trace(lambda trace: trace.update(showlegend=False) if (trace.name in legend_names) else legend_names.add(trace.name))
            fig.update_layout(title=f"Route: Dep @ {path_dict['departure']}, Arr @ {path_dict['arrival']}", legend_title_text='Segment Type', height=600, margin={"r":0,"t":50,"l":0,"b":0})
            return fig


# %%
# MORE UI STUFF
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
    _cached_formatted_paths = [] 
    
    route_selection_dropdown = widgets.Dropdown(description='View Route on Map:', disabled=True, layout=widgets.Layout(width='450px'))

    def on_route_selected_for_map(change): # This callback needs access to _cached_formatted_paths
        map_out.clear_output(wait=True)
        selected_route_label_or_value = change.new 
        
        if not _cached_formatted_paths: return

        # 'value' of the dropdown will be the index if options are (label, index)
        # 'label' attribute will be the string label.
        # setting .options as list of (label, index) tuples, so change.new will be the index (the value part)
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
                    # commented out cause it kept creating double maps ?
                    # with map_out: display(planner.visualize_path(_cached_formatted_paths[0], stop_info_for_planner_methods))
                else: route_selection_dropdown.disabled = True
            except Exception as e: print(f"Error during search: {e}"); import traceback; traceback.print_exc()
    
    search_btn.on_click(on_search_clicked)
    time_input_controls = widgets.VBox([widgets.HBox([use_dep_time_cb, dep_time_text]), widgets.HBox([use_arr_time_cb, arr_time_text])])
    input_ui_layout = widgets.VBox([widgets.HBox([dep_dropdown, arr_dropdown]),date_picker_ui, time_input_controls, widgets.HBox([conf_slider, max_opts_slider]),search_btn, route_selection_dropdown])
    tabs_output = widgets.Tab([results_out, map_out]); tabs_output.set_title(0, 'Options'); tabs_output.set_title(1, 'Map')
    display(widgets.VBox([input_ui_layout, tabs_output]))


# %% jupyter={"source_hidden": true}
# Putting it all together
def implement_robust_journey_planner(graph, stops_df, delay_model=None):
    planner = OptimizedRobustJourneyPlanner(
    G_stops, # new graph with updated time dependent edges 
    transfers_lookup, 
    stop_coords_dict, 
    delay_model=None, 
    )
    
    ui = create_journey_planner_ui(planner, stops_df)
    
    return planner, ui


# %% [markdown]
# ## The UI and Algorithm
#
# #### Students Note to TAs
#
# Takes about 30 seconds to initialize the UI
# On average, routing algorithm can take anywhere from 20 seconds to 2 minutes to display routes, depending on the route length and the number of routes requested.
#
# Not average, 
# While testing our routing algorithm, we found the farthest route to be from around the Lausanne, Ste-cathering area to St-Sulipce VD, Venoge Sud
# where displaying 1 route took about 1.5 mins
# However, it accurately mapped and showed the route

# %%
planner, ui = implement_robust_journey_planner(G_stops, stops_lausanne_rt)
display(ui)


# %% [markdown]
# #### Students Note to TAs
#
# Small note #1:   
# That little "None" that is printed after our UI does not directly mean that the algo didnt find routes  
# It's just an implicit return of create_journey_planner_ui function  
# In general though:  
# Our routing algorithm finds the earliest arrival path in a time-dependent graph.   
# It uses a modified Dijkstra’s algorithm that accounts for waiting times at each stop.   
# At each node, it explores only connections that depart after the current arrival time.   
# A priority queue maintains the next earliest connections to process based on arrival time.   
# The algorithm continues until it reaches the target stop or all options are exhausted.   
#
# Small note #2:  
# Difference between this algorithm and YEN   
# Us : Depart Source 07:00 -> Arrive Target 08:30 (Total Travel Time: 90 mins) --> Arrival time at destination focus   
# YEN : Depart Source 07:30 -> Arrive Target 08:40 (Total Travel Time: 70 mins) --> Total travel time focus 

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # V. Validation
# %% vscode={"languageId": "shellscript"}
spark.stop()
