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

groupName='X1'

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


# ### b) Explore SBB data - 3/10
#
# Explore the _{sharedNS}.sbb_istdaten_, _{sharedNS}.sbb_stops_, and _{sharedNS}.sbb_stop_times_ tables.
#
# Identify the field(s) used across all three tables to represent stop locations. Analyze their value ranges, format patterns, null and invalid values, and identify any years when null or invalid values are more prevalent. Use this information to implement the necessary transformations for reliably joining the tables on these stop locations.

# +
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


# -

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

# +
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


# -


query = basic_info("sbb_istdaten", ["bpuic", "stop_name"])
pretty_table('sbb_istdaten', query, conn)

query = basic_info("sbb_stops", ["stop_id", "stop_name"])
pretty_table('sbb_stops', query, conn)

query = basic_info("sbb_stop_times", ["stop_id"])
pretty_table('sbb_stop_times', query, conn)

# We can see sbb_istdaten has almost 3 billion rows, big data indeed. The bpuic has significantly less null or empty values than the stop names (97 vs ~200 million), so perhaps we should merge our tables on this value, we can see that the buic seems to start with 85 and range from 6 to 9 digits, but there could be typos somewhere. We can also comfirm from the min and max not being NULL that as mentionned in the column description, the stop_name attribute is not always purely alphabetical, some of the stop_names contain the buic (ones before 2021). There are ~25 thousand distinct buics, and ~30 thousand distict stop names, pointing to some inconsistent naming of stop names. 
#
# sbb_stops is smaller with  roughly 12 million rows, while sbb_stop_times is the biggest table with over 3 billion rows! In the sbb_stops and sbb_stop_times data, we notice zero null or empty stop_ids, although perhaps some unconventional number has been chosen to represent the NULL stop and therefore bypasses initial detection. We also notice that both datatables have the same min and max stop_id which is encouraging for joining them together, however it is not the same as sbb_istdaten. sbb_stops has over double the amount of unique stop_ids (\~93,000) compared to unique stop_names (\~42,000), which suggests that perhaps two ID conventions have been used. sbb_stop_times has less unique stop_ids with (\~57,000).

# +
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


# -

query = null_by_year("sbb_istdaten","operating_day", ["bpuic","stop_name"])
df_ist = to_pd_table("sbb_istdaten", query, conn)
df_ist

plot_null_percentages_by_year(df_ist,"sbb_istdaten")

# We can see in the graph above that no year seems to have a particularly high percentage of nulls, with 2021 being the year with the lowest percentage of nulls. As seen previously, there are a lot more null stop_names than stop_ids.
#
# Let us now begin the joining process by joining sbb_istdaten and sbb_stops by stop name to see how well they match up, we can then compare the id naming conventions and test to see if we get a better match that way.

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


# We see that overall the names seem to match somewhat well on the ist part at least, although stops has significantly more unique stop names, let's look at some individual matches to inspect stop_id:  

# +
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

# -

# Visually inspecting the matches, we can see three main types of formatting conventions for stop_ids in the sbb_stops:
# - {buic}
# - Parent{buic}
# - {buic}:\*:\*
#   
# Let us count how many rows of sbb_stops follow each format

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


# It looks like the majority of IDs do follow this format!
# Lets do a sanity check with the other two tables

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


# We see that istdaten and stop_times almost exactly fall in the given categories! Let's check how many matches we now get on stop_id for all three tables:

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


# We see that we get an almost perfect match, with only around 700 missing matched from istdaten!
#
# We could now perform the full join with the following code:

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


if False:
    # Check structure of joined table
    query = f"SELECT * FROM {output_table_name}"
    pretty_table(output_table_name, query, conn,5)

if False:
    # Check row count of joined table
    query = f"SELECT COUNT(*) AS row_count FROM {output_table_name}"
    pretty_table("Row count of joined table", query, conn)


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

df_ttype["month_year"] = pd.to_datetime({
    "year": df_ttype["year"],
    "month": df_ttype["month"],
    "day": 1
}).dt.strftime("%Y-%m")
df_ttype

# +
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

# -

# We notice that transport type had inconsistent capitalisation so everything has been put to lowercase. We also notice language inconsistencies with some transport modes being marked in German, which could potentially correspond to the same transports as some of the english names. We also notice different scales, with some transports going into the tens of Millions such as bus, while taxi has numbers ranging from less than 10 to 100. We also notice an odd transport type, wm-bus, which should probably be included along with bus. We also notice a seasonality pattern in the schiff (boat?) and zahnradbahn modes of transports, we also see a slight summer drop in metro use.

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

# +
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


# +
# just checking that it worked
# I reran the original sql_fetch code after creating shayakhm schema
# schemas = pd.read_sql(f"SHOW SCHEMAS IN iceberg", conn)
# filtered_schemas = [schema for schema in schemas['Schema'] if schema.startswith('s')]
# print(filtered_schemas)
# -

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

# -

### TODO verify the results
query = f"""
SELECT * FROM {namespace}.sbb_stops_lausanne_region
LIMIT 5
"""
pretty_table("sbb_stops_lausanne_region", query, conn)

query = f"""
SELECT COUNT(*) AS total_stops
FROM {namespace}.sbb_stops_lausanne_region
"""
pretty_table("Number of stops in Lausanne region", query, conn)


# We validate that the subset table has roughly the right number of rows, and has the correct structure. We add a bpuic column containing only the bpuic identifier as in sbb.ist_daten

# ### b) Find stops with real time data in Lausanne region - 5/50

# * Use the results of table _{username}.sbb_stops_lausanne_region_ to find all the stops for which real time data is reported in the _{sharedNS}.sbb_istdaten_ table for the full month of **July 2024**.
# * Report the results in a pandas DataFrame that you will call _stops_df_.
# * Validation: you should find between 3% and 4% of stops in the area of interest that do not have real time data.
# * Hint: it is recommended to first generate a list of _distinct_ stop identifiers extracted from istdaten data. This can be achieved through either a nested query or by creating an intermediate table (use your findings of Part I.b).

# ---
# #### Solution
#

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

# %%time
stops_df= to_pd_table("Stops with/without real-time data (July 2024)", query, conn, limit = 1000)

# +
### TODO - Verify the results
total_stops = len(stops_df)
no_rt_stops = len(stops_df[stops_df["has_realtime"] == False])
pct_missing = 100.0 * no_rt_stops / total_stops

print(f"Total stops in Lausanne region: {total_stops}")
print(f"Stops without real-time data: {no_rt_stops} ({pct_missing:.2f}%)")

# Optional preview
stops_df.head()

# +
# Filter to stops without real-time data
no_rt_stops_df = stops_df[stops_df["has_realtime"] == False]

# Preview the first few rows
print("Preview: Stops without real-time data (July 2024)")
display(no_rt_stops_df.head())

# -

# Findings: there are five stops that have no real time data, they are listed above.

# ### c) Display stops in the Lausanne Region - 3/50
#
# * Use plotly or similar plot framework to display all the stop locations in Lausanne region on a map (scatter plot or heatmap), using a different color to highlight the stops for which istdaten data is available.

# +

### TODO - Display results of stops_df

fig = px.scatter_map(data_frame=stops_df, lat="stop_lat", lon="stop_lon", color="has_realtime", hover_name="stop_name")

fig.show()

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

# +
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
# -


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

# +
### TODO - Verify the results

pd.read_sql(f"""SELECT COUNT(*) FROM {namespace}.sbb_stop_to_stop_lausanne_region""", conn)
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
# -

# Some departure and arrival times were larger than 24h as they arrive on the next day so we removed 24 to the hour number so they can be cast as TIME variables.
#
# sbb_trips and sbb_calendar had two entries on the first week of July so we selected one of them to avoid duplicates.

query = f"""
SELECT * FROM {sharedNS}.sbb_stop_times
LIMIT 10
"""
pretty_table("sbb_stop_times", query, conn, 10)

# +
### TODO - Verify the results

query = f"""
SELECT COUNT(*) AS total_pairs
FROM (
  SELECT DISTINCT trip_id, stop_id
  FROM {namespace}.sbb_stop_times_lausanne_region
) AS distinct_pairs
"""

pd.read_sql_query(query, conn)


# +
query = f"""
SELECT COUNT(*) AS monday_pairs
FROM (
  SELECT DISTINCT trip_id, stop_id
  FROM {namespace}.sbb_stop_times_lausanne_region
  WHERE monday = TRUE
) AS monday_pairs
"""

pd.read_sql_query(query, conn)
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
# Peeking at the tables that we will be using 

# + jupyter={"source_hidden": true}
query = f"""
SELECT *
FROM {namespace}.sbb_stop_to_stop_lausanne_region
LIMIT 5
"""
pretty_table("sbb_stop_to_stop_lausanne_region", query, conn, 5)

# + jupyter={"source_hidden": true}
query = f"""
SELECT *
FROM {namespace}.sbb_stop_times_lausanne_region
LIMIT 5
"""
pretty_table("sbb_stop_times_lausanne_region", query, conn, 5)

# + jupyter={"source_hidden": true}
query = f"""
SELECT *
FROM {namespace}.sbb_stop_to_stop_lausanne_region
LIMIT 5
"""
pretty_table("sbb_stop_to_stop_lausanne_region", query, conn, 5)
# -

# Install these necessary libraries please 
# !pip install networkx
# !pip install datetime
# !pip install geopy

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

# +
### TODO - Data Preparation
# Using the sbb_stop_times_lausanne table, we are going to create a NetworkGx Graph.


# +
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





# +
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





# +
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




# +
### TODO - Routing Algorithm

# We decided to use a Dijkstra base + label-setting algorithm
# so we find the earliest arrival times from a starting stop and time to all other stops within a maximum duration 
# it accounts for both walking and public transport 
# and it explores nodes based on the current known shortest total travel time


# +

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



# +
###TODO - Interactive Interface to Verify the Algorithm

# +
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

# +
### TODO - Others as you see fit.
# -

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

query = f"""
SELECT *
FROM {namespace}.sbb_stop_times_lausanne_region
WHERE trip_id = '727.TA.92-21-I-j24-1.12.H'
  AND monday = TRUE
ORDER BY departure_time
"""
pretty_table("sbb_stop_times_lausanne_region", query, conn, 13)


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

time_filter = pd.to_timedelta("23:00:00")
bussigny_departures = lausanne_stop_times[
    (lausanne_stop_times['stop_id'] == stop_dict['Bussigny']) &
    (lausanne_stop_times['departure_td'] >= time_filter)]
# Sorting chronologically by departure time
bussigny_departures_sorted = bussigny_departures.sort_values(by='departure_td')
bussigny_departures_sorted

# We see in the table above we have 4 trips departing Bussigny! So why aren't they showing up in our visualized map ??
#
# Let's take a closer look at each trip:

example_a= '932.TA.91-4-L-j24-1.313.R'
lausanne_stop_times[lausanne_stop_times['trip_id'] == example_a]

example_b = '914.TA.91-3-T-j24-1.687.R'
lausanne_stop_times[lausanne_stop_times['trip_id'] == example_b]

# The trips above start AND end at Bussigny, explaining why they did not created edges in the graph to new stops. 

example_c = '297.TA.91-3-T-j24-1.874.H'
lausanne_stop_times[lausanne_stop_times['trip_id'] == example_c]

example_d = '277.TA.91-4-L-j24-1.293.H'
lausanne_stop_times[lausanne_stop_times['trip_id'] == example_d]

# Even though it looks like something departs from Bussigny, it's actually the last stop of the trip. That’s why our visualization shows no reachable stops from Bussigny.
#
# After thorough examination, I understand why some stops like Bussigny show no connections to other stops, even if it still feels a bit counterintuitive.
#
# Side note: departure time being earlier than arrival time is also questionable. 
#
# <br> <br>
# Overall, the algorithm seems to function well, but there are a few issues to consider, as it doesn't always display the truly optimal paths to reach certain destinations.
#












