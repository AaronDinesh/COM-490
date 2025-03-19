# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # DSLab Assignment 2 - Data Wrangling with Hadoop
# ---
#
# ## Hand-in Instructions
#
# - __Due: **.04.2025 23h59 CET__
# - Create a fork of this repository under your group name, if you do not yet have a group, you can fork it under your username.
# - `git push` your final verion to the master branch of your group's repository before the due date.
# - Set the group name variable below, e.g. group_name='Z9'
# - Add necessary comments and discussion to make your codes readable.
# - Let us know if you need us to install additional python packages.
#
# ## Useful references
#
# * [Trino documentation](https://trino.io/docs/471)
# * [Enclosed or Unenclosed](https://github.com/Esri/spatial-framework-for-hadoop/wiki/JSON-Formats)

groupName='#TODO

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

# <div style="font-size: 100%" class="alert alert-block alert-warning">
#     <b>Fair cluster Usage:</b>
#     <br>
#     As there are many of you working with the cluster, we encourage you to prototype your queries on small data samples before running them on whole datasets. Do not hesitate to partion your tables, and LIMIT the output of your queries to a few rows to begin with. You are also free to test your queries using alternative solutions such as <i>DuckDB</i>.
#     <br><br>
#     You may lose your session if you remain idle for too long or if you interrupt a query. If that happens you will not lose your tables, but you may need to reconnect to the warehouse.
#     <br><br>
#     <b>Try to use as much SQL as possible and avoid using pandas operations.</b>
# </div>

# +
import os
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable .*")

# +
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


# +
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
# -

# ---

# +
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

# +
import pandas as pd

pd.read_sql(f"""SHOW TABLES IN {sharedNS}""", conn)


# -

# ## Part I. 10 Points

# ### a) Declare an SQL result generator - 2/10
#
# Complete the Python generator below to execute a single query or a list of queries, returning the results row by row.
#
# The generator should implement an out-of-core approach, meaning it should limit memory usage by fetching results incrementally, rather than loading all data into memory at once.

def sql_fetch(queries, conn, batch_size=100):
    if isinstance(queries, str):
        queries = [queries]
    ### TODO


# ### b) Explore SBB data - 3/10
#
# Explore the _{sharedNS}.sbb_istdaten_, _{sharedNS}.sbb_stops_, and _{sharedNS}.sbb_stop_times_ tables.
#
# Identify the field(s) used across all three tables to represent stop locations. Analyze their value ranges, format patterns, null and invalid values, and identify any years when null or invalid values are more prevalent. Use this information to implement the necessary transformations for reliably joining the tables on these stop locations.

# +
### TODO
# -

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

# %%time
### TODO - create the data frame

# +
### TODO - display the results to match the graph above
# -

# ---

# ## Part II. 50 Points

# In this second Part, we will leverage the historical SBB data to model the public transport infrastructure within the Lausanne region.
#
# Our objective is to establish a comprehensive data representation of the public transport network, laying the groundwork for our final project. While we encourage the adoption of a data structure tailored to the specific requirements of your final project implementation, the steps outlined here provide a valuable foundation.
#
# In this part you will make good use of DQL statements of nested SELECT, GROUP BY, JOIN, IN, DISTINCT, and Geo Spatial UDF.

# You must create a managed database within your designated namespace, where you'll define the tables necessary for modeling your infrastructure. By 'managed,' we mean that you should not specify a default external location for the namespace.
#
# While it's a bit of an overkill, the safest approach is to drop and recreate the namespace the first time you run this notebook.

list(sql_fetch([
    f"""DROP SCHEMA IF EXISTS {namespace} CASCADE""", # CASCADE will drop all the tables
    f"""CREATE SCHEMA IF NOT EXISTS {namespace}""",
], conn))

# ### a) Find the stops in Lausanne region - 5/50
#
#

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

# ---
# #### Solutions

# +
### TODO create the table

# +
### TODO verify the results
# -

# ### b) Find stops with real time data in Lausanne region - 5/50

# * Use the results of table _{username}.sbb_stops_lausanne_region_ to find all the stops for which real time data is reported in the _{sharedNS}.sbb_istdaten_ table for the full month of **July 2024**.
# * Report the results in a pandas DataFrame that you will call _stops_df_.
# * Validation: you should find between 3% and 4% of stops in the area of interest that do not have real time data.
# * Hint: it is recommended to first generate a list of _distinct_ stop identifiers extracted from istdaten data. This can be achieved through either a nested query or by creating an intermediate table (use your findings of Part I.b).

# ---
# #### Solution
#

# +
### TODO - Create the data frame stops_df
# -

# %%time
stops_df= # TODO

# +
### TODO - Verify the results
# -

# Findings ...

# ### c) Display stops in the Lausanne Region - 3/50
#
# * Use plotly or similar plot framework to display all the stop locations in Lausanne region on a map (scatter plot or heatmap), using a different color to highlight the stops for which istdaten data is available.

# +
### TODO - Display results of stops_df
# -

# Note that some stops lacking real-time data may actually serve as representations for groups of stops, like `Parent8592050` for Lausanne Gare, which denotes a cluster of stops within Lausanne train station. We ignore these.

# ### d) Find stops that are within walking distances of each other - 10/50

# * Use the results of table _{username}.sbb_stops_lausanne_region_ to find all the (directed) pair of stops that are within _500m_ of each other.
# * Save the results in table _{username}.sbb_stops_to_stops_lausanne_region_
# * Validation: you should find around $3500\pm 250$ directed stop paris (each way, i.e. _A_ to _B_ and _B_ to _A_).
# * Hint: Use the Geo Spatial UDF, in spherical geopgraph.
# * Aim for the table _{namespace}.sbb_stop_to_stop_lausanne_region_:
#     * _stop_id_a_: an _{sharedNS}.sbb_stops.stop_id_
#     * _stop_id_b_: an _{sharedNS}.sbb_stops.stop_id_
#     * _distance_: straight line distance in meters from _stop_id_a_ to _stop_id_b_

# %%time
### TODO - create the stop to stop table

# +
### TODO - Verify the results
# -

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

# +
### TODO - Create the table stop times as described above

# +
### TODO - Verify the results
# -

# ### f) Design considerations - 2/50
#
# We aim to use our previous findings to recommend an optimal public transport route between two specified locations at a given time on any day of a particular week in any region.
#
# Running queries on all data for the entire data set would be inefficient. Could you suggest an optimized table structure to improve the efficiency of queries on the {username}.sbb_stop_times_lausanne_region table?

# ---
# #### Solutions
#
# **TODO**: Design considerations, no code required.

# ### h) Isochrone Map - 15/50

# Note: This question is open-ended, and credits will be allocated based on the quality of both the proposed algorithm and its implementation. You will receive credits for proposing a robust algorithm, even if you do not carry out the implementation.
#
# Moreover, it is not mandatory to utilize the large scale database for addressing this question; plain Python is sufficient. You are free to employ any Python package you deem necessary. However, ensure that you list it as a prerequisite of this notebook so that we remember to install them.

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

# ---
# #### Solution

# **TODO**: Explain your algorithm, design decisions etc. here

# +
### TODO - Data Preparation

# +
### TODO - Routing Algorithm

# +
### TODO - Interactive interface to verify the algorithm

# +
### TODO - Others as you see fit.
