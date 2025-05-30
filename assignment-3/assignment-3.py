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
# ---
# # DSLab Homework3 - Uncovering Public Transport Conditions using SBB data
#
# In this notebook, we will use temporal information about sbb transports to discover delay trends. 
#
# ## Hand-in Instructions:
#
# - __Due: *.05.2025~ 23:59:59 CET__
# - your project must be private
# - add necessary comments and discussion to make your codes readable
# - make sure that your code is runnable

# %% [markdown]
# ---
# <div style="font-size: 100%" class="alert alert-block alert-info">
#     <b>ℹ️  Fair Cluster Usage:</b> As there are many of you working with the cluster, we encourage you to:
#     <ul>
#         <li>Whenever possible, prototype your queries on small data samples or partitions before running them on whole datasets</li>
#         <li>Save intermediate data in your HDFS home folder <b>f"/user/{username}/..."</b></li>
#         <li>Convert the data to an efficient storage format when this is an option</li>
#         <li>Use spark <em>cache()</em> and <em>persist()</em> methods wisely to reuse intermediate results</li>
#     </ul>
# </div>
#
# For instance:
#
# ```python
#     # Read a subset of the original dataset into a spark DataFrame
#     df_sample = spark.read.csv(f'/data/com-490/csv/{table}', header=True).sample(0.01)
#     
#     # Save DataFrame sample
#     df_sample.write.parquet(f'/user/{username}/assignment-3/{sample_table}.parquet', mode='overwrite')
#
#     # ...
#     df_sample = spark.read.parquet(f'/user/{username}/assignment-3/{sample_table}.parquet')
# ```
#
# Note however, that due to Spark partitioning, and parallel writing, the original order may not be preserved when saving to files.

# %% [markdown]
# ---
# ## Start a spark Session environment

# %% [markdown]
# We provide the `username` and `hadoopFS` as Python variables accessible in both environments. You can use them to enhance the portability of your code, as demonstrated in the following Spark SQL command. Additionally, it's worth noting that you can execute Iceberg SQL commands directly from Spark on the Iceberg data.

# %%
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
            .config('spark.sql.catalog.spark_catalog.warehouse', f'{hadoopFS}/user/{username}/assignment-3/warehouse')\
            .config("spark.sql.warehouse.dir", f'{hadoopFS}/user/{username}/assignment-3/spark/warehouse')\
            .config('spark.eventLog.gcMetrics.youngGenerationGarbageCollectors', 'G1 Young Generation')\
            .config("spark.executor.memory", "10g")\
            .config("spark.executor.cores", "4")\
            .config("spark.executor.instances", "4")\
            .master('yarn')\
            .getOrCreate()

# %%
spark.sparkContext

# %% [markdown]
# Be nice to others - remember to add a cell `spark.stop()` at the end of your notebook.

# %% [markdown]
# ---
# For your convenience, the Spark sessions is configured to use a default _spark_catalog_ and our _iceberg_ catalog where the SBB data is located.
#
# Execute the code below to create your schema _spark_catalog.{username}_ and set it as your default, and verify the presence of the Iceberg SBB tables.

# %%
# %%time
spark.sql(f'CREATE SCHEMA IF NOT EXISTS spark_catalog.{username}')

# %%
# %%time
spark.sql(f'USE spark_catalog.{username}')

# %%
# %%time
spark.sql(f'SHOW CATALOGS').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW SCHEMAS').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW TABLES').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW SCHEMAS IN iceberg').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW TABLES IN iceberg.sbb').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW TABLES IN iceberg.geo').show(truncate=False)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.stop_times LIMIT 1').show(truncate=False,vertical=True)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.trips LIMIT 1').show(truncate=False,vertical=True)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.routes LIMIT 1').show(truncate=False,vertical=True)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.calendar LIMIT 1').show(truncate=False,vertical=True)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.calendar_dates LIMIT 1').show(truncate=False,vertical=True)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.istdaten LIMIT 1').show(truncate=False,vertical=True)

# %% [markdown]
# 💡 Notes:
# - Do not hesitate to create temporary views out of tabular, data stored on file or from SQL queries. That will make your code reusable, and easier to read
# - You can convert spark DataFrames to pandas DataFrames inside the spark driver to process the results. But Only do this for small result sets, otherwise your spark driver will run OOM.

# %%
# %%time
"""
This reads in the weather station csv and then creates a spark dataframe. This is then chained with the 
.withColumns() method to cast the lat and lon columns to double. Finally we register it as a temporary SQL
table called 'weather_stations'. This will only last for the lifetime of the current spark session (until 
you call spark.stop()).

Note there are other columns in this CSV, but we only explictly cast lat and lon to doubles. The rest will
remain as strings.
"""

spark.read.options(header=True).csv(f'/data/com-490/csv/weather_stations').withColumns({
      'lat': F.col('lat').cast('double'),
      'lon': F.col('lon').cast('double'),
    }).createOrReplaceTempView("weather_stations")

# %%
"""
Table Schema
root
 |-- Name: string (nullable = true)
 |-- City: string (nullable = true)
 |-- Canton: string (nullable = true)
 |-- ID: string (nullable = true)
 |-- Active: string (nullable = true)
 |-- lat: double (nullable = true)
 |-- lon: double (nullable = true)
"""

spark.sql(f'SELECT * FROM weather_stations').printSchema()

# %%
# %%time
# Note, that this would also works: SHOW TABLES IN global_temp
spark.sql(f'SHOW TABLES').show(truncate=False)

# %%
spark.table("weather_stations").show()

# %%
spark.sql(f'SELECT * FROM weather_stations LIMIT 5').toPandas()

# %%
# spark.sql(f'DROP VIEW weather_stations')

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ---
# ## PART I: First Steps with Spark DataFrames using Weather Data (20 points)
#
# We copied several years of historical weather data downloaded from [Wunderground](https://www.wunderground.com).
#
# We made this weather data available because of its strong impact on our daily lives, including in areas such as transportation.
#
# Let's see if we can see any trends in this data.

# %% [markdown]
# ### I.a Restructure the weather history - 2/20
#
# Load the JSON data from HDFS _/data/com-490/json/weather_history/_ into a Spark DataFrame using the appropriate method from the SparkSession.
#
# Restructure the data so that the schema matches this output, where the field _observation_ is a **single** record of weather meaasurements at a given point in time:
# ```
# root
#  |-- metadata: struct (nullable = true)
#  |    |-- expire_time_gmt: long (nullable = true)
#  |    |-- language: string (nullable = true)
#  |    |-- location_id: string (nullable = true)
#  |    |-- status_code: long (nullable = true)
#  |    |-- transaction_id: string (nullable = true)
#  |    |-- units: string (nullable = true)
#  |    |-- version: string (nullable = true)
#  |-- site: string (nullable = true)
#  |-- year: integer (nullable = true)
#  |-- month: integer (nullable = true)
#  |-- observation: struct (nullable = true)
#  |    |-- blunt_phrase: string (nullable = true)
#  |    |-- class: string (nullable = true)
#  |    |-- clds: string (nullable = true)
#  |    |-- ...
# ```
#
# 💡 Notes:
# - The JSON data is multilines
# - Use functions learned during the exercises.
# - https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html

# %%
## TODO - imports
from pyspark.sql.functions import input_file_name, regexp_extract, year, month, col, from_unixtime, explode
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

# %%
# %%time
## TODO - read weather history data and convert to structure as described above
weather_raw = spark.read.option("multiline", "true").json("/data/com-490/json/weather_history/")

# %%
# sneak peek into the raw structure
weather_raw.printSchema()

# %%
# exploding observations
# explode "observations" will return a new row for each element in the given array 
 
weather_df = weather_raw.select(
    "metadata",
    "site",
    explode("observations").alias("observation"),
    "year",
    "month"
)

# %%
# making datetime human readable rather than unix
# extracting year/month 
weather_df = weather_df.withColumn("year", year(from_unixtime("observation.valid_time_gmt")))
weather_df = weather_df.withColumn("month",month(from_unixtime("observation.valid_time_gmt")))

# %%
# just checking if it matches the structure that was asked for
# 
# notice how it was observation->element->* but then after exploding it is observation->*
#  
weather_df.printSchema()

# %% [markdown]
# __User-defined and builtin functions__
#
# In Spark Dataframes you can create your own user defined functions for your SQL commands.
#
# So, for example, if we wanted to make a user-defined python function that returns a string value in lowercase, we could do something like this:

# %%
import pyspark.sql.functions as F


# %%
@F.udf
def lowercase(text):
    """Convert text to lowercase"""
    return text.lower()


# %% [markdown]
# The `@F.udf` is a "decorator" -- and in this case is equivalent to:
#
# ```python
# def lowercase(text):
#     return text.lower()
#     
# lowercase = F.udf(lowercase)
# ```
#
# It basically takes our function and adds to its functionality. In this case, it registers our function as a pyspark dataframe user-defined function (UDF).
#
# Using these UDFs is very straightforward and analogous to other Spark dataframe operations. For example:

# %%
# %%time
weather_raw.select(weather_raw.site,lowercase(weather_raw.site).alias('lowercase_site')).show(n=5)

# %% [markdown]
# The DataFrame API already includes many built-in functions, including the function for converting strings to lowercase.
# Other handy built-in dataframe functions include functions for transforming date and time fields.
#
# Note that the functions can be combined. Consider the following dataframe and its transformation:
#
# ```
# from pyspark.sql import Row
#
# # create a sample dataframe with one column "degrees" going from 0 to 180
# test_df = spark.createDataFrame(spark.sparkContext.range(180).map(lambda x: Row(degrees=x)), ['degrees'])
#
# # define a function "sin_rad" that first converts degrees to radians and then takes the sine using built-in functions
# sin_rad = F.sin(F.radians(test_df.degrees))
#
# # show the result
# test_df.select(sin_rad).show()
# ```
#
# Refs:
# - [Dataframe API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)
# - [GroupedData API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.html)

# %% [markdown]
# ---
# ### I.b Processing timestamps - 2/20

# %% [markdown]
# Use UDF to organize the weather data by their timestamps.
#
# Check out the [Spark python API documentation](https://spark.apache.org/docs/latest/api/python/index.html). Look for the `sql` section and find the listing of `sql.functions`. Using either Spark built-in functions or their equivalent SQL expressions, convert the GMT _observation.valid_time_gmt_ from a string format to a date format _YYYY-mm-dd HH:MM:SS_, and extract the year, month, day, hour and minute components.
#
# A sample of the output should be similar to the one shown below:
#
# ```
# +----+--------------------------+-------------------+----+-----+----------+----+------+
# |site|observation.valid_time_gmt|observation_time_tz|year|month|dayofmonth|hour|minute|
# +----+--------------------------+-------------------+----+-----+----------+----+------+
# |LSGC|1672528800                |2023-01-01 00:20:00|2023|1    |1         |0   |20    |
# |LSGC|1672530600                |2023-01-01 00:50:00|2023|1    |1         |0   |50    |
# |LSGC|1672532400                |2023-01-01 01:20:00|2023|1    |1         |1   |20    |
# |LSGC|1672534200                |2023-01-01 01:50:00|2023|1    |1         |1   |50    |
# |LSGC|1672536000                |2023-01-01 02:20:00|2023|1    |1         |2   |20    |
# +----+--------------------------+-------------------+----+-----+----------+----+------+
# ```
#
# ⚠️ When working with dates and timestamps, always be mindful of timezones and Daylight Saving Times (DST). Verify that the time information is consistent for the first few hours of January 1st and DST changes. Note that the weather's year, month, and day fields are based on the local timezone, i.e. _'Europe/Zurich'_. Timestamps represent the number of seconds since _1970-01-01 00:00:00 UTC_. However, Spark may interpret timezones differently depending on the function used and the local timezone of the spark cluster, which can lead to inconsistencies, i.e. you may instead end up with the wrong values, like:
#
# ```
# +----+--------------------------+-------------------+----+-----+----------+----+------+
# |site|observation.valid_time_gmt|observation_time_tz|year|month|dayofmonth|hour|minute|
# +----+--------------------------+-------------------+----+-----+----------+----+------+
# |LSGC|1672528800                |2022-12-31 23:20:00|2023|1    |31        |23  |20    |
# ```

# %%
## TODO - import
from pyspark.sql import functions as F

# %%
# %%time
## TODO - write code to convert to structure as described above
# 'observation.valid_time_gmt'to human-readable datetime format
weather_df = weather_df.withColumn("observation_time_tz", F.from_unixtime(F.col("observation.valid_time_gmt"), "yyyy-MM-dd HH:mm:ss"))


# %%
# taking year, month, day, hour, and minute from the datetime column
# the .withColumn() method is used to add a new column to the DataFrame
weather_df = weather_df.withColumn("year", F.year("observation_time_tz"))
weather_df = weather_df.withColumn("month", F.month("observation_time_tz"))
weather_df = weather_df.withColumn("dayofmonth", F.dayofmonth("observation_time_tz"))
weather_df = weather_df.withColumn("hour", F.hour("observation_time_tz"))
weather_df = weather_df.withColumn("minute", F.minute("observation_time_tz"))

# %%
# checking our general result
weather_df.select("site", "observation.valid_time_gmt", "observation_time_tz", "year", "month", "dayofmonth", "hour", "minute").show(5, truncate=False)

# %%
# the above seems to match well 
# for sanity, let's also check the january 1st change

janFirst_df = weather_df.filter(
    (
        (F.date_format("observation_time_tz", "MM-dd") == "12-31") &
        (F.hour("observation_time_tz") == 23) &
        (F.minute("observation_time_tz") >= 50)
    ) |
    (
        (F.date_format("observation_time_tz", "MM-dd") == "01-01") &
        (F.hour("observation_time_tz") == 0) &
        (F.minute("observation_time_tz") <= 20)
    )
).orderBy("observation_time_tz")

janFirst_df.select(
    "site", 
    "observation.valid_time_gmt", 
    "observation_time_tz",
    "month",
    "dayofmonth",
    "hour",
    "minute"
).show(20, truncate=False)


# %%
# januray 1st change output makes sense 
# now let's check DST changes (March 26th 2023 and October 29th 2023)
                                                                              
# since the jump is 2 am to 3am  
# but many transportations don't run between 3-4 am
dst_spring = weather_df.filter(
    (F.col("site") == "LSGC") &
    (F.col("observation_time_tz") >= '2023-03-26 00:00:00') & 
    (F.col("observation_time_tz") < '2023-03-26 06:00:00')
)

# since the jump is from 2 am to 1am
dst_autumn = weather_df.filter(
    (F.col("site") == "LSGC") &
    (F.col("observation_time_tz") >= '2023-10-29 01:00:00') &
    (F.col("observation_time_tz") <= '2023-10-29 04:00:00')
)

# sneak peek to check 
dst_spring.select(
    "site", "observation_time_tz", "year", "month", "dayofmonth", "hour", "minute"
).orderBy("observation_time_tz").show(50, truncate=False)

dst_autumn.select(
    "site", "observation_time_tz", "year", "month", "dayofmonth", "hour", "minute"
).orderBy("observation_time_tz").show(50, truncate=False)


# %% [markdown]
# <font color="#7777ff">
# Luckily, the transitions seem to be pretty accurate with the observation_time_tz entry. 

# %% [markdown]
# ### I.c Transform the data - 4/20
#
# Modify the DataFrame to add the weather measurements column and save the transformation into a _weather_df_ table: 
#
# The Spark Dataframe weather_df must includes the columns _month_, _dayofmonth_, _hour_ and _minutes_, calculated from _observation.valid_time_gmt_ as before, and:
#
# - It contains all (and only) the data from a full year of data, that is if there is only data in the second part of _2022_ then you shouldn't consider any data from _2022_. However, few missing values and gaps in the data are acceptable.
# - It contains a subset of weather information columns from the original data as show in the example below
# - A row should be similar to:
#
# ```
#  site                       | LSGC                
#  observation.valid_time_gmt | 1672528800          
#  observation_time_tz        | 2023-01-01 00:20:00 
#  valid_time_gmt             | 1672528800          
#  clds                       | CLR                 
#  day_ind                    | N                   
#  dewPt                      | 6                   
#  feels_like                 | 13                  
#  gust                       | 48                  
#  heat_index                 | 13                  
#  obs_name                   | La Chaux-De-Fonds   
#  precip_hrly                | NULL                
#  precip_total               | NULL                
#  pressure                   | 904.36              
#  rh                         | 63                  
#  temp                       | 13                  
#  uv_desc                    | Low                 
#  uv_index                   | 0                   
#  vis                        | 9.0                 
#  wc                         | 13                  
#  wdir                       | 240                 
#  wdir_cardinal              | WSW                 
#  wspd                       | 30                  
#  wx_phrase                  | Fair                
#  year                       | 2023                
#  month                      | 1                   
#  dayofmonth                 | 1                   
#  hour                       | 0                   
#  minute                     | 20                  
# ```
#
# __Note:__ 
# - [pyspark.sql.DataFrame](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)

# %%
# what years are full

"""
We first group all the entries by year. Then we count the distint pairs of month and days of that month.
We give the result from this counting operation the name 'num_days'. Then we aggregate the data to count over
the the distinct (month, dayofmonth) pairs and then order it in chronological order of the year and then print the result.
"""


from pyspark.sql.functions import countDistinct
weather_df.groupBy("year").agg(
    countDistinct("month", "dayofmonth").alias("num_days")
).orderBy("year").show()

# %% [markdown]
# <span style="color:red">
# Okay so we see that 2022, 2023, and 2024 are full years! Also, 2024 is a leap year :)
# </span>

# %%
# %%time
# TODO
# we're filtering out the incomplete years
weather_df = weather_df.filter(F.col("year").isin([2022, 2023, 2024]))

# %%
# selecting the columns mentioned in the question 
# we're removing the observation.___ predicate for simplicity 
selected_columns = [
    "site",
    F.col("observation.valid_time_gmt").alias("observation.valid_time_gmt"),
    "observation_time_tz",
    F.col("observation.valid_time_gmt").alias("valid_time_gmt"),
    F.col("observation.clds").alias("clds"),
    F.col("observation.day_ind").alias("day_ind"),
    F.col("observation.dewPt").alias("dewPt"),
    F.col("observation.feels_like").alias("feels_like"),
    F.col("observation.gust").alias("gust"),
    F.col("observation.heat_index").alias("heat_index"),
    F.col("observation.obs_name").alias("obs_name"),
    F.col("observation.precip_hrly").alias("precip_hrly"),
    F.col("observation.precip_total").alias("precip_total"),
    F.col("observation.pressure").alias("pressure"),
    F.col("observation.rh").alias("rh"),
    F.col("observation.temp").alias("temp"),
    F.col("observation.uv_desc").alias("uv_desc"),
    F.col("observation.uv_index").alias("uv_index"),
    F.col("observation.vis").alias("vis"),
    F.col("observation.wc").alias("wc"),
    F.col("observation.wdir").alias("wdir"),
    F.col("observation.wdir_cardinal").alias("wdir_cardinal"),
    F.col("observation.wspd").alias("wspd"),
    F.col("observation.wx_phrase").alias("wx_phrase"),
    "year",
    "month",
    "dayofmonth",
    "hour",
    "minute"
]

# %%
# updating the final weather data frame 
weather_df = weather_df.select(*selected_columns)


# %%
# checking the columns 
weather_df.columns

# %%
# checking our result
weather_df.show(1, truncate=False)


# %% [markdown]
# ### I.d Top average monthly precipitation per site - 4/20
#
# We will now use the Spark DataFrame group by aggregations to compute monthly aggregations.
#
# The _Spark.DataFrame.groupBy_ does not return another DataFrame, but a _GroupedData_ object instead. This object extends the DataFrame with methods that allow you to do various transformations and aggregations on the data in each group of rows. 
#
# Conceptually the procedure is a lot like this:
# ![groupby](./figs/sgCn1.jpg)
#
#
# The column set used for the _groupBy_ is the _key_ - and it can be a list of column keys, such as _groupby('key1','key2',...)_ - all rows in a _GroupedData_ have the same key, and various aggregation functions can be applied on them to generate a transformed DataFrame. In the above example, the aggregation function is a simple `sum`.

# %% [markdown]
# **Question:**
#
# Apply a group by on the _weather_df_ created earlier to compute the monthly precipitation on (site,month of year). Find the sites and months that have the highest total precipitation: sort the site and month in decreasing order of monthly precipitation and show the 10 top ones.
#
# Name the spark DataFrame _avg_monthly_precip_df_.
#
# The schema of the table is, at a minimum:
# ```
# root
#  |-- site: string (nullable = true)
#  |-- month: integer (nullable = true)
#  |-- avg_total_precip: double (nullable = true)
# ```
#
# Note:
# * A site may report multiple hourly precipitation measurements (precip_hrly) within a single hour. To prevent adding up hourly measurement for the same hour, you should compute an aggregated values observed at each site within the same hour.
# * Some weather stations do not report the hourly  precipitation, they will be shown as _(null)_

# %%
# TODO - imports
from pyspark.sql.functions import round
# for viz (inspo from A2 2h)
import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

# %%
# TODO 
# only keeping non null
weather_df_cleaned = weather_df.filter(weather_df["precip_hrly"].isNotNull())

# aggregating by site and month, then summing hourly precip
avg_monthly_precip_df = weather_df_cleaned.groupBy("site", "month").agg(
    F.sum("precip_hrly").alias("avg_total_precip")
).orderBy("avg_total_precip", ascending=False)

# showing top 10 rows with highest total precipitation
avg_monthly_precip_df.show(10)

# %%
# rounding to 3 decimal places
avg_monthly_precip_df = avg_monthly_precip_df.withColumn("avg_total_precip", round("avg_total_precip", 3))

# showing top 10 rows with highest total precipitation (rounded)
avg_monthly_precip_df.show(10)

# %%
# checking schema structure so that it matches the question
avg_monthly_precip_df.printSchema()

# %% [markdown]
# Convert the _avg_monthly_precip_df_ Spark DataFrames to a Pandas DataFrame and **visualize the results** in the notebook.
#
# We are not looking for perfection, we just want to verify that your results are generally accurate. However, feel free to unleash your creativity and come up with a visualization that you find insightful.
#
# 💡 Do not hesitate to take advantage of the _weather_station_ table if you would like to include geospation information in your analysis. The data is available in _/data/com-490/csv/weather_stations_, see also examples at the beginning of this notebook. Additional details about the stations can be found [here](https://metar-taf.com/?c=464582.65369.10).

# %%
# make it pandas 
avg_monthly_precip_df_pandas = avg_monthly_precip_df.toPandas()


# %%
# interactive plot that lets the user filter by site 
# I believe this plot represents the average precipitation well and is easy to understand

def plot_precipitation(selected_site):
    # filtering by site 
    filtered_data = avg_monthly_precip_df_pandas[avg_monthly_precip_df_pandas['site'] == selected_site]

    # making the figure
    fig = px.bar(filtered_data, 
                 x="month", 
                 y="avg_total_precip", 
                 color="site", 
                 labels={"avg_total_precip": "Average Precipitation (mm)", "month": "Month"},
                 title=f"Average Monthly Precipitation for {selected_site}",
                 category_orders={"month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
    
    fig.show()

# creating the dropdown option
site_dropdown = widgets.Dropdown(
    options=avg_monthly_precip_df_pandas['site'].unique(),
    description='Site:',
    disabled=False
)

# putting it all together 
widgets.interactive(plot_precipitation, selected_site=site_dropdown)

# %% [markdown]
# ### I.e Spark Windows  - 4/20
#
# In the previous question, we calculated the total average monthly precipitation for each site.
#
# Now, let's shift focus: suppose we want to determine, for each day, which site reported the highest temperature.
#
# This is a more complex task—it can't be done with simple aggregations alone. Instead, it requires _windowing_ our data, a powerful technique that allows us to perform calculations across sets of rows that are related to the current row, without collapsing the data as a regular group-by would.
#
#
# We recommend reading this [window functions article](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html), the [spark.sql.Window](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html)  and optionally the [Spark SQL](https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-window.html) documentation to get acquainted with the idea. You can think of a window function as a fine-grained and more flexible _groupBy_. 
#
# To use window functions in Spark, we need to define two key aspects:
# 1. Window Specifications: This includes defining the columns for partitioning, the order in which the rows should be arranged, and the grouping criteria for the window.
# 2. Aggregation Logic: This specifies the aggregation or computation (such as max, avg, etc.) to be performed on each windowed group
#
# Define a window function, _hourly_window_, that partitions the data by the columns (_year, month, dayofmonth, hour_). Within each partition, order the rows by hourly temperature in descending order. Then, apply the _rank_ function over this window to assign rankings to the sites based on their temperatures (_temp_). Finally, filter the results to keep only the top _N_ ranked sites.
#
# Despite the complexity of the operation, it can be accomplished efficiently in just a few lines of code!

# %%
from pyspark.sql import Window

# %% [markdown]
# First, define a 'tumbling' (fixed-size, non-overlapping) _pyspark.sql.window.WindowSpec_ to specify the partitioning and ordering of the window. This window definition partitions the data by the columns (_year, month, dayofmonth, hour_) and orders the rows (i.e., the site measurements) within each partition by temperature (_temp_), ordered in descending order. As outlined in the previous section, the pattern should follow:
#
# ```
# Window.partitionBy(...).orderBy(...) ...
# ```

# %%
# TODO - create the window specifications
hourly_window = Window.partitionBy("year", "month", "dayofmonth", "hour").orderBy(F.col("temp").desc())

# %% [markdown]
# Next, define the computation for the _hourly_window_. This is a window aggregation of type _pyspark.sql.column.Column_, which allows you to perform calculations (such as ranking or aggregation) within the defined window.
#
# Use this _hourly_window_ to calculate the hourly ranking of temperatures. Use the helpful built-in F.rank() _spark.sql.function_, and call its _over_ method to apply it over the _hourly_window_, and name the resulting column (alias) _rank_.

# %%
# TODO - create the hourly ranking logics that will be applied on hourly window
hourly_rank = F.rank().over(hourly_window).alias("rank")

# %% [markdown]
# **Checkpoint:** the resulting object is analogous to the SQL expression `RANK() OVER (PARTITION BY year, month, dayofmonth, hour ORDER BY temp DESC NULLS LAST ...) AS rank`. This _window function_ assigns a rank to each record within the partitions and based on the ordering criteria of _hourly_window_.

# %%
print(hourly_rank)

# %% [markdown]
# Finally, apply the _hourly_rank_ window computation to the _weather_df_ DataFrame computed earlier.
#
# Filter the results to show all and only the sites with the 5 highest temperature per hour (if multiple sites have the same temperature, they count as one), then order the hourly measurements in chronological order, showing the top ranked sites in their ranking order.
#
# **Checkpoint:** The output should ressemble:
#
# ```
# +----+----+-----+----------+----+----+----+
# |site|year|month|dayofmonth|hour|temp|rank|
# +----+----+-----+----------+----+----+----+
# +----+----+-----+----------+----+----+----+
# |site|year|month|dayofmonth|hour|temp|rank|
# +----+----+-----+----------+----+----+----+
# |LSZE|2023|1    |1         |0   |16  |1   |
# |LSZT|2023|1    |1         |0   |14  |2   |
# |LSPH|2023|1    |1         |0   |14  |2   |
# |....|....|.    |.         |.   |..  |.   |
# +----+----+-----+----------+----+----+----+
# ```

# %%
# TODO -- apply the window logic to create the additional column rank, and display the results as shown above
weather_with_rank = weather_df.withColumn("rank", hourly_rank)

# top 5 ranked sites per hour
top_sites_per_hour = weather_with_rank.filter(weather_with_rank.rank <= 5)

# now chronologically (by year, month, day, hour)
top_sites_per_hour_sorted = top_sites_per_hour.orderBy("year", "month", "dayofmonth", "hour", "rank")

# sneak peek result
top_sites_per_hour_sorted.select("site", "year", "month", "dayofmonth", "hour", "temp", "rank").show(5,truncate=False)

# %% [markdown]
# <font color="#7777ff">
# So we notice how the biggest difference between our result and the exemplary one is that our data includes the year
# 2022, while the exemplary data shows 2023. However, according to Part 1C of the question, the years 2022, 2023, and 2024
# are all considered full years. Therefore, I believe our result is still valid, as the question does not specifically
# exclude the year 2022.

# %% [markdown]
# ### I.f Sliding Spark Windows - 4/20

# %% [markdown]
# In the previous question, we computed the rank over a tumbling window, where the windows are of fixed size (hourly) and do not overlap.
#
# With window functions, you can also compute aggregate functions over a _sliding window_, where the window moves across the data, potentially overlapping with previous intervals.
#
# **Question:** For each site, calculate the hourly average temperature computed over the past 3 hours of data.
#
# The process follows a similar pattern to the previous steps, with a few distinctions:
#
# * Rows are processed independently for each site.
# * The window slides over the timestamps (_valid_time_gmt_, in seconds) in chronological order, spanning intervals going back to 2 hours and 59 minutes (10740 seconds) **before** the current row's timestamp up to the current row's timestamp."
#
# 💡 Hints:
# * [spark.sql.Window(Spec)](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html)
# * Times are in minutes intervals

# %% [markdown]
# First, as before, define a _pyspark.sql.window.WindowSpec_ to specify the window partition, the row ordering inside the partition, and its _range_.

# %%
# TODO
sliding_3hour_window = Window.partitionBy("site").orderBy(F.col("valid_time_gmt").asc()).rangeBetween(-10740, 0)

# %% [markdown]
# Next, define a computation on the window: calculate the average ranking of temperatures. Use the helpful built-in F.avg() spark.sql.function, apply it _over_ the window, and name the result (_alias_) _avg_temp_.

# %%
# TODO
sliding_3hour_avg = F.avg("temp").over(sliding_3hour_window).alias("avg_temp")

# %% [markdown]
# **Checkpoint:** the resulting object is analogous to the SQL expression `avg(temp) OVER (PARTITION BY site ORDER BY valid_time_gmt ASC NULLS FIRST RANGE BETWEEN ... AND CURRENT ROW) AS avg_temp`

# %%
print(sliding_3hour_avg)

# %% [markdown]
# Finally, apply _sliding_3hour_avg_ to the _weather_df_ DataFrame computed earlier, and order chronologically. _Then_ filter the output to show the outpout of sites 'LSTO', 'LSZH and 'LSGL'.
#
# **Checkpoint:** The output should ressemble:
#
# ```
# +--------------+----+-----+----------+----+----+----+------------------+
# |valid_time_gmt|year|month|dayofmonth|hour|site|temp|          avg_temp|
# +--------------+----+-----+----------+----+----+----+------------------+
# |    1640991600|2022|    1|         1|   0|LSGL|   7|               7.0|
# |    1640991600|2022|    1|         1|   0|LSTO|  10|              10.0|
# |    1640992800|2022|    1|         1|   0|LSZH|   2|               2.0|
# |    1640994600|2022|    1|         1|   0|LSZH|   3|               2.5|
# |    1640995200|2022|    1|         1|   1|LSGL|   6|               6.5|
# |    1640995200|2022|    1|         1|   1|LSTO|  10|              10.0|
# |    1640996400|2022|    1|         1|   1|LSZH|   3|2.6666666666666665|
# |    1640998200|2022|    1|         1|   1|LSZH|   3|              2.75|
# (...)
# ```

# %%
# check point 
weather_df_with_avg = weather_df.select(
    "valid_time_gmt", "year", "month", "dayofmonth", "hour", "site", "temp",
    sliding_3hour_avg
)
# filtering
selected_sites = ["LSTO", "LSZH", "LSGL"]
weather_filtered = weather_df_with_avg.filter(F.col("site").isin(selected_sites))

#chronological order
weather_filtered.orderBy("valid_time_gmt").show(10)

# %% [markdown]
# <font color="#7777ff">
# Seems like a passed checkpoint!

# %%
# TODO -- apply the sliding_3hour_avg logic, showing ony the results for the site in LSGG","LSGP","LSGL","LSTR","LSGB"
weather_df_with_avg = weather_df.select(
    "valid_time_gmt", "year", "month", "dayofmonth", "hour", "site", "temp",
    sliding_3hour_avg
)

# filtered according to the To Do comment
# we're doing the same thing we did in the previous code block
# both just had different sites 
selected_sites = ["LSGG", "LSGP", "LSGL", "LSTR", "LSGB"]
weather_filtered = weather_df_with_avg.filter(F.col("site").isin(selected_sites))

# chronological order
weather_filtered.orderBy("valid_time_gmt").show(10)

# %% [markdown]
# Adapt the _hourly_rank_ and combine it with the window to show the weather stations with the 5 top temperatures averaged over the 3h sliding window.

# %%
# %%time
# TODO: idk what to do here if it was already defined?
sliding_hourly_rank=Window.partitionBy('year','month','dayofmonth','hour').orderBy(F.desc('avg_temp'))

# this code was given with the assignment, but the double select looked weird to me
# weather_df\
#     .select('valid_time_gmt', 'year', 'month', 'dayofmonth','hour', 'site', 'temp', sliding_3hour_avg) \
#     .select('valid_time_gmt', 'year', 'month', 'dayofmonth','hour', 'site', col('avg_temp').alias('temp'),hourly_rank) \
#     .filter('rank <= 5')\
#     .sort('valid_time_gmt','rank')\
#     .show()

# adjusting the code slightly
weather_df \
    .withColumn("avg_temp", sliding_3hour_avg) \
    .withColumn("rank", F.rank().over(sliding_hourly_rank)) \
    .filter(F.col("rank") <= 5) \
    .select(
        "valid_time_gmt",
        "year",
        "month",
        "dayofmonth",
        "hour",
        "site",
        F.col("avg_temp").alias("temp"),
        "rank"
    ) \
    .orderBy("valid_time_gmt", "rank") \
    .show()

# %% [markdown]
# <font color="#7777ff">
#
# In my understanding, when a site has rank 1, it means it had the highest 3-hour average temperature among all sites for that specific hour.
#
# Interestingly though, I do not see rank 3 anywhere. Apparently, there’s no rank 3 because rank() skips ranks after ties. So if two sites tie at rank 2 (which often happens in our table), the next is rank 4.

# %% [markdown]
# ---
# ## PART II: SBB Network - Vehicle Journey Trajectories (20 points)

# %% [markdown]
# ### II.a Filter trips from SBB Istdaten - 4/20

# %% [markdown]
# In this part, you will reconstruct public transport journey trajectories from the available transport data, as illustrated below. The example displays the historical data extracted from istdaten for a single trip ID, collected over the course of a year. The horizontal axis represents the sequence of stops along the trip, while the vertical axis shows the timeline of arrivals and departures of each trip. This type of chart offers valuable insights into where delays typically occur along the journey.
#
# ![./figs/journeys.png](./figs/journeys.png)
#
# There are several ways to compute this table in Spark, each with its own trade-offs. In the next question, you'll explore one such method using window and table pivot functions.
#
# ⚠️ The question in this section can be computationally demanding if you are not careful, therefore:
#
# - It is advisable to begin by experimenting with smaller datasets first. Starting with smaller datasets enables faster iterations and helps to understand the computational requirements before moving on to larger datasets.
# - It is advisable to use the DataFrame _cache()_ for the most expensive computation. _Istdaten_trips_df_ is a good candidate for that, because it can take several tens of seconds to generate this relatively small table.
#

# %% [markdown]
# ---
# First, create the DataFrame _istdaten_trips_df_ from the _iceberg.sbb.istdaten_ table.
#
# The table must:
# - Only include the ids of _distinct_ trip that appear on at least 200 different days in _isdaten_ in _2024_
# - Only trips from the Transport Lausanne (TL) operator.
# - Only trip ids that serve stops in the Lausanne region.
#     - Use the data available in _/data/com-490/labs/assignment-3/sbb_stops_lausanne_region.parquet_)
#     - Or use Trino to create your own list of stops in the greater Lausanne region (Lausanne and Ouest Lausannois).
#
# 💡 Note:
# - You may assume that the SBB timetables (_stops_, _stop_times_ etc), are valid for the full year in which they are published.
# - Filtering the trips based on both the TL operator and the presence of at least one stop in the only region served by this operator might seem redundant in this case. However, in a more general context, this approach allows us to reuse the same query for nation wide operators.
#

# %%
# %%time
lausanne_stops_df = spark.sql('SELECT DISTINCT * FROM parquet.`/data/com-490/labs/assignment-3/sbb_stops_lausanne_region.parquet`')
lausanne_stops_df.createOrReplaceTempView('lausanne_stops_df')

# %%
lausanne_stops_df.printSchema()

# %%
lausanne_stops_df.show(5)

# %%
# %%time
#read and filter the Istdaten table for TL trips in 2024
tl_2024_df = (spark.table("iceberg.sbb.istdaten")
      .filter(
         (F.col("operating_day") >= "2024-01-01") &
         (F.col("operating_day") <= "2024-12-31") &
         (F.col("operator_abrv") == "TL")))
tl_2024_df.createOrReplaceTempView("tl_2024_df")
tl_2024_df.show(3, truncate=False)

# %%
# %%time
#join to Lausanne region stops to keep only those events
#lausanne_stops_df and tl_2024_df use different column names for the same values (stop_id and bpuic)
#so the join condition must be specified explicitly
tl_lausanne_df = (tl_2024_df.join(lausanne_stops_df, tl_2024_df.bpuic == lausanne_stops_df.stop_id, how="inner"))
tl_lausanne_df.createOrReplaceTempView("tl_lausanne_df")
tl_lausanne_df.show(3, truncate=False)

# %%
# %%time
#extract distinct (trip_id, operating_day) pairs
distinct_trip_days_df = (
    tl_lausanne_df
      .select("trip_id", "operating_day")
      .distinct())
distinct_trip_days_df.show(5)

# %%
# %%time
#count days per trip and filter for >=200 days
trip_day_counts_df = (
    distinct_trip_days_df
      .groupBy("trip_id")
      .agg(F.countDistinct("operating_day").alias("num_days"))
      .filter(F.col("num_days") >= 200))
trip_day_counts_df.show(5)

# %%
# %%time
##TODO -- create the table containing a list of trip_id as specified above
istdaten_trips_df = (
    trip_day_counts_df
      .select("trip_id")
      .cache())

# %%
# %%time
istdaten_trips_df.createOrReplaceTempView('istdaten_trip_ids')

# %%
# %%time
istdaten_trips_df.count()

# %%
# %%time

"""
istdaten_trips_df contains all the trip_ids that have at least 200 days of data in 2024
"""
istdaten_trips_df.show(10, truncate=False)

# %% [markdown]
# ### II.b Transform the data - 8/20
#
# Next, use the _istdaten_trips_df_ table computed earlier to create a Spark Dataframe _istdaten_df_ that contains a subset of _sbb.istdaten_ containing only trips that are listed into _istdaten_trips_df_.
#
# The table must:
# - Include the _istdaten_ details for the full year 2024 of all the trips that appear in _istdaten_trips_df_.
# - Not include _failed_ or _unplanned_ trips.
# - Include all stops in the Lausanne area and stops that are not listed in the Lausanne area, but are connected via at least one trip to stops in the Lausanne area.
#
# The table should be similar to the one shown below when properly ordered (showing only one trip ID on a given operating day, it can have additional columns if you want):
#
# ```
# +-------------+-----------------------------+-------+-------------------+-------------------+-------------------+-------------------+
# |operating_day|trip_id                      |bpuic  |arr_time           |arr_actual         |dep_time           |dep_actual         |
# +-------------+-----------------------------+-------+-------------------+-------------------+-------------------+-------------------+
# |2024-01-03   |85:151:TL013-4506262507243798|8579253|NULL               |NULL               |2024-01-03 11:51:00|2024-01-03 11:51:30|
# |2024-01-03   |85:151:TL013-4506262507243798|8579254|2024-01-03 11:52:00|2024-01-03 11:52:14|2024-01-03 11:52:00|2024-01-03 11:52:50|
# |2024-01-03   |85:151:TL013-4506262507243798|8591991|2024-01-03 11:53:00|2024-01-03 11:54:19|2024-01-03 11:53:00|2024-01-03 11:54:45|
# |2024-01-03   |85:151:TL013-4506262507243798|8592074|2024-01-03 11:56:00|2024-01-03 11:56:04|2024-01-03 11:56:00|2024-01-03 11:56:10|
# |2024-01-03   |85:151:TL013-4506262507243798|8592009|2024-01-03 11:57:00|2024-01-03 11:56:57|2024-01-03 11:57:00|2024-01-03 11:57:25|
# |2024-01-03   |85:151:TL013-4506262507243798|8592083|2024-01-03 11:58:00|2024-01-03 11:57:50|2024-01-03 11:58:00|2024-01-03 11:58:17|
# |2024-01-03   |85:151:TL013-4506262507243798|8592045|2024-01-03 11:59:00|2024-01-03 11:58:42|2024-01-03 11:59:00|2024-01-03 11:58:42|
# |2024-01-03   |85:151:TL013-4506262507243798|8592129|2024-01-03 12:00:00|2024-01-03 11:59:04|NULL               |NULL               |
# +-------------+-----------------------------+-------+-------------------+-------------------+-------------------+-------------------+
# ```

# %%
# %%time
## TODO - create the table as specified above
#filter: filter to TL in 2024 and only planned, non failed runs
#join: keep only the trip_ids selected in 2a
#select: select the stop time columns
istdaten_df = (
    spark.table("iceberg.sbb.istdaten")
      .filter(
        (F.col("operating_day") >= "2024-01-01") &
        (F.col("operating_day") <= "2024-12-31") &
        (F.col("operator_abrv") == "TL") &
        (~F.col("failed")) &
        (~F.col("unplanned")))
      .join(istdaten_trips_df, on="trip_id", how="inner")
      .select(
        "operating_day",
        "trip_id",
        "bpuic",
        "arr_time",
        "arr_actual",
        "dep_time",
        "dep_actual")
      .cache())

# %%
# %%time
istdaten_df.createOrReplaceTempView("istdaten_df")

# %%
# %%time
istdaten_df.count()

# %% [markdown]
# Validate for one trip ID:

# %%
# %%time
istdaten_df.filter(
    """
        trip_id = '85:151:TL013-4506262507243798' AND operating_day='2024-01-03'
    """).select(
        'operating_day',
        'trip_id',
        # -- Optional columns
        #'operator_abrv',
        #'line_text',
        #'line_id',
        #'product_id',
        #'stop_name',
        'bpuic',
        'arr_time',
        'arr_actual',
        'dep_time',
        'dep_actual'
    ).orderBy([
        'arr_time',
    ]).show(20,truncate=False)

# %%
istdaten_df.printSchema()

# %% [markdown]
# ### II.c Compute Journey Trajectories - 8/20
#
# Create a windows operator as seen before to work on _operating_day, trip_id_, _ordered_ by _arr_time_ (expected arrival times, and actual arrival times to break ties if expected arrival times are equal). Use the window to create the Spark DataFrame _trip_sequences_df_. In each window, compute:
# - _start_time_: the **first non-null** (ignore nulls) expected _dep_time_ in the window, with respect to the window's ordering.
# - _sequence_: the order of the _bpuic_ in the trip journey, according to the windows' ordering.
# - _arr_time_rel_: the interval _(arr_time - start_time)_, or NULL if _arr_time_ is NULL
# - _dep_time_rel_: the interval _(dep_time - start_time)_, or NULL if _dep_time_ is NULL
# - _arr_actual_rel_: the interval _(arr_actual - start_time)_, or NULL if _arr_actual_ is NULL
# - _dep_actual_rel_: the interval _(dep_actual - start_time)_, or NULL if _dep_actual_ is NULL
#
# The results for a given _operating_day, trip_id_ should look like this, feel free to add additional columns, such as _line_text_ etc.:
# ```
# +-------------+--------------------+-------+--------+-------------------+--------------------+--------------------+--------------------+--------------------+
# |operating_day|             trip_id|  bpuic|sequence|         start_time|        arr_time_rel|      arr_actual_rel|        dep_time_rel|      dep_actual_rel|
# +-------------+--------------------+-------+--------+-------------------+--------------------+--------------------+--------------------+--------------------+
# |   2024-01-03|85:151:TL013-4506...|8579253|       1|2024-01-03 11:51:00|                NULL|                NULL|INTERVAL '0 00:00...|INTERVAL '0 00:00...|
# |   2024-01-03|85:151:TL013-4506...|8579254|       2|2024-01-03 11:51:00|INTERVAL '0 00:01...|INTERVAL '0 00:01...|INTERVAL '0 00:01...|INTERVAL '0 00:01...|
# |   2024-01-03|85:151:TL013-4506...|8591991|       3|2024-01-03 11:51:00|INTERVAL '0 00:02...|INTERVAL '0 00:03...|INTERVAL '0 00:02...|INTERVAL '0 00:03...|
# |   2024-01-03|85:151:TL013-4506...|8592074|       4|2024-01-03 11:51:00|INTERVAL '0 00:05...|INTERVAL '0 00:05...|INTERVAL '0 00:05...|INTERVAL '0 00:05...|
# |   2024-01-03|85:151:TL013-4506...|8592009|       5|2024-01-03 11:51:00|INTERVAL '0 00:06...|INTERVAL '0 00:05...|INTERVAL '0 00:06...|INTERVAL '0 00:06...|
# |   2024-01-03|85:151:TL013-4506...|8592083|       6|2024-01-03 11:51:00|INTERVAL '0 00:07...|INTERVAL '0 00:06...|INTERVAL '0 00:07...|INTERVAL '0 00:07...|
# |   2024-01-03|85:151:TL013-4506...|8592045|       7|2024-01-03 11:51:00|INTERVAL '0 00:08...|INTERVAL '0 00:07...|INTERVAL '0 00:08...|INTERVAL '0 00:07...|
# |   2024-01-03|85:151:TL013-4506...|8592129|       8|2024-01-03 11:51:00|INTERVAL '0 00:09...|INTERVAL '0 00:08...|                NULL|                NULL|
# +-------------+--------------------+-------+--------+-------------------+--------------------+--------------------+--------------------+--------------------+
# ```
#
# And the schema (minimum column set):
# ```
# root
#  |-- operating_day: date (nullable = true)
#  |-- trip_id: string (nullable = true)
#  |-- bpuic: integer (nullable = true)
#  |-- sequence: integer (nullable = false)
#  |-- start_time: timestamp_ntz (nullable = true)
#  |-- arr_time_rel: interval day to second (nullable = true)
#  |-- arr_actual_rel: interval day to second (nullable = true)
#  |-- dep_time_rel: interval day to second (nullable = true)
#  |-- dep_actual_rel: interval day to second (nullable = true)
# ```
#
# 💡 Hints:
# - The times are of type _timestamp_ntz_. You can easily compute a time interval with expressions like _F.col('t2')-F.col('t1')_ or _F.expr('t2 - t1')_.
# - Use Windows aggregation logics AUDF (as previously seen) to compute the start time and the sequence number over the _operating_day, trip_id_ windows.
# - Use _F.row_number()_ to get the row number in a window (according to the window's ordering).
# - In Spark ordering, NULL timestamps come first.

# %%
## TODO - create the window specifications
# per‐trip ordering window: partition by day+trip, order by expected arrival, then actual to break ties
seq_window = (
    Window
      .partitionBy("operating_day", "trip_id")
      .orderBy(
          F.col("arr_time").asc_nulls_first(),
          F.col("arr_actual").asc_nulls_first()))
#aggregation window over the entire trip to grab the very first departure as start_time
agg_window = seq_window.rowsBetween(
    Window.unboundedPreceding,
    Window.unboundedFollowing)

# %%
## TODO - create the logics you want to apply on the window
# start_time: first non-null dep_time in the trip (ignore any NULLs)
start_time_col = F.first("dep_time", ignorenulls=True).over(agg_window)
# sequence: row number within the trip (1,2,3,…)
sequence_col  = F.row_number().over(seq_window)

# %%
## TODO - create the table as described above
#select: bring along the raw timestamps
#withColumn: add windowed columns and compute intervals relative to start_time
trip_sequences_df = (
    istdaten_df
      .select("operating_day","trip_id","bpuic", "arr_time","arr_actual","dep_time","dep_actual")
      .withColumn("start_time", start_time_col)
      .withColumn("sequence", sequence_col)
      .withColumn("arr_time_rel", F.when(F.col("arr_time").isNotNull(),
                                F.col("arr_time") - F.col("start_time")))
      .withColumn("dep_time_rel", F.when(F.col("dep_time").isNotNull(),
                                F.col("dep_time") - F.col("start_time")))
      .withColumn("arr_actual_rel", F.when(F.col("arr_actual").isNotNull(),
                                F.col("arr_actual") - F.col("start_time")))
      .withColumn("dep_actual_rel", F.when(F.col("dep_actual").isNotNull(),
                                F.col("dep_actual") - F.col("start_time")))
      .cache())

# %%
trip_sequences_df.printSchema()

# %%
# %%time
trip_sequences_df.count()

# %%
# %%time
trip_sequences_df.filter(
    """trip_id LIKE '85:151:TL013-4506262507243798'"""
).select(
    'operating_day',
    'trip_id',
    'bpuic',
    'sequence',
    'start_time',
    'arr_time_rel',
    'arr_actual_rel',
    'dep_time_rel',
    'dep_actual_rel'
).show(8,truncate=True)

# %%
trip_sequences_df.createOrReplaceTempView('trip_sequences_df')

# %% [markdown]
# ---
# Use the _trip_sequence_df_ dataframe to trace the journey of each trip, displaying the travel time from the first stop to all subsequent stops sequence along the route. The _x-axis_ of the graph represents the stops in the journey, while the _y-axis_ represents the travel time from the first stop.
#
# Note that the dataframe contains many invalid traces that should be ignored
#
# You can verify the presence of invalid traces by running queries on the dataframe, such as:
#
# ```
# SELECT trip_id,
#        bpuic,
#        collect_set(sequence)
#        FROM trip_sequences_df
#              GROUP BY trip_id,bpuic
# ```
# Or
# ```
# SELECT trip_id,
#        num_stops,
#        COUNT(*) as freq
#        FROM (
#              SELECT operating_day,
#                     trip_id,
#                     MAX(sequence) as num_stops
#                  FROM trip_sequences_df
#                  GROUP BY operating_day,trip_id
#        ) GROUP BY trip_id,num_stops ORDER BY trip_id
# ```
#
# These queries reveal inconsistencies in stop ordering, or inconsistent number of stops per trip across the year. Together, these queries (and similar ones) are useful for identifying day-to-day variations or anomalies in the stop sequences of scheduled trips.
#
# We did this analysis for you and we suggest that you to focus your analysis on valid and reliable trip patterns, such as _trip_id='85:151:TL031-4506262507505612'_ for the rest of this study.

# %% [markdown]
# Consider only _trip_id='85:151:TL031-4506262507505612'_.
#
# Create the table _trips_pivoted_df_, based on the following conditions:
# - It only include the trip with the specified trip_id.
# - It only include trips on operating days where the first actual departure time (at sequence id 1) is within 5 minutes (before or after) of the expected departure time.
# - The table should consists of the following columns:
#     - _bpuic_: a stop on the _trip_id_ line
#     - _sequence_: The sequence number of the bpuic, _1_ is the first stop.
#     - _evt_type_: indicates if the row corresponds to an arrival or a departure time of the trip at the given stop.
#     - _{trip_id}_: Contains the expected departure and arrival times (in chronological order) of the the selected _trip_id_.
#     - _{trip_id}-{operating_day}_: Contains the actual departure and arrival times (in chronological order) of the selected _trip_id_, on the operating_day.
#
# The table schema should look like:
#
# ```
# root
#  |-- bpuic: integer (nullable = true)
#  |-- sequence: integer (nullable = false)
#  |-- evt_type: string (nullable = false)
#  |-- 85:151:TL031-4506262507505612: long (nullable = true)              # Column of expected times
#  |-- 85:151:TL031-4506262507505612_2024-01-03: long (nullable = true)   # Actual times on 2024-01-03
#  |-- 85:151:TL031-4506262507505612_2024-01-04: long (nullable = true)   # Actual times on 2024-01-04
#  |-- 85:151:TL031-4506262507505612_2024-01-05: long (nullable = true)   # ...
#  |-- 85:151:TL031-4506262507505612_2024-01-08: long (nullable = true)
#  |-- ...
# ```
#
# And below is sample of how the table should look like. Column (4) are the expected times, columns (5) and above are 
# actual times observed on the given days.
#
# ```
#    (1)      (2)       (3)         (4)                 (5)                        (...)
# +-------+--------+--------+----------------+---------------------------+---------------------------+
# |  bpuic|sequence|evt_type|85:151:TL031-...|85:151:TL031-..._2024-01-03|85:151:TL031-..._2024-01-04|
# +-------+--------+--------+----------------+---------------------------+---------------------------+
# |8588983|       1|     arr|            NULL|                       NULL|                       NULL|
# |8588983|       1|     dep|               0|                         12|                          6|
# |8593869|       2|     arr|              60|                        149|                        129|
# |8593869|       2|     dep|              60|                        158|                        129|
# |8591933|       3|     arr|             180|                        217|                        180|
# |8591933|       3|     dep|             180|                        238|                        220|
# |8593868|       4|     arr|             240|                        253|                        220|
# |8593868|       4|     dep|             240|                        256|                        220|
# |8507227|       5|     arr|             360|                        321|                        280|
# |8507227|       5|     dep|             360|                        336|                        280|
# |8593867|       6|     arr|             420|                        357|                        326|
# |8593867|       6|     dep|             420|                        376|                        343|
# |8594986|       7|     arr|             480|                        433|                        403|
# |8594986|       7|     dep|             480|                        457|                        423|
# +-------+--------+--------+----------------+---------------------------+---------------------------+
# ```
#
# 💡 Hints:
# - It will be easier for you to convert the time intervals to seconds, e.g. using the UDF _F.col('interval').cast("long")_, or the Spark DataFrame _filter("CAST(interval AS long)")_

# %% [markdown]
# There are many ways to accomplish this task. The steps outlined below are just one possible approach. Feel free to experiment and try out your own method.
#
# **First**, compute the DataFrame _trip_filter_list_df_ that contains ony the _operating days_ on which the actual departure time of the considered _trip_id_ at _sequence=1_ is no more than 5mins (300 seconds) before or after the expected departure time.

# %%
# %%time
## TODO - create the trip_filter_list_df as indicated above
# identify the trip to analyze
target_trip = "85:151:TL031-4506262507505612"
# find operating_days where sequence=1 actual dep is within +-300 s of start_time:
trip_filter_list_df = (
    trip_sequences_df
      .filter(F.col("trip_id") == target_trip)
      .filter(F.col("sequence") == 1)
      .filter(F.abs(F.col("dep_actual_rel").cast("long")) <= 300)
      .select("operating_day")
      .distinct()
      .cache())

# %%
trip_filter_list_df.createOrReplaceTempView("trip_filter_list_df")

# %%
# %%time
trip_filter_list_df.show()

# %% [markdown]
# **Second**, create _trip_filter_sequence_df_, a subset of _trip_sequence_df_ computed earlier that contains only the trips of interest happening on the days computed in _trip_filter_list_df_

# %%
# %%time
## TODO - create the subset of trip_filter_sequence_df as indicated above
#subset the full sequences to only that trip and those valid days
trip_filter_sequences_df = (
    trip_sequences_df
      .filter(F.col("trip_id") == target_trip)
      .join(trip_filter_list_df, on="operating_day", how="inner")
      .cache())

# %%
trip_filter_sequences_df.createOrReplaceTempView("trip_filter_sequences_df")

# %%
# %%time
trip_filter_sequences_df.show(3)

# %% [markdown]
# Next, create two DataFrames, _planned_df_ and _actual_df_.
#
# For _planned_df_, The schema should include the following columns:
# - _trip_id_: The trip identifier, e.g. _85:151:TL031-4506262507505612_
# - _bpuic_: A stop ID (this column is informative only, for verification purpose).
# - _sequence_: The sequence number of the stop within the specified _trip_id_.
# - _evt_type_: Use the function _F.explode()_ in a _withColumn_ operation to duplicate each row into two: one with _evt_type_ set to "arr" and the other with _evt_type_ set to "dep".
# - _evt_time_:
#     - _F.col('arr_time_rel')_ **when** _evt_type = "arr"_
#     - _F.col('dep_time_rel')_ **when** _evt_type = "dep"_
#
# For _actual_df_:
#     This DataFrame will have the same schema as _planned_df_, but the values for _evt_time_ will be based on _arr_actual_rel_ and _dep_actual_rel_, instead of the planned _arr_time_rel_ and _dep_time_rel_. The values in the column _trip_id_ should be changed to the append the _operating_day_ to the _trip_id_, e.g. 85:151:TL031-4506262507505612_2024-01-03, 85:151:TL031-4506262507505612_2024-01-04, ...
#
# 💡 Hints:
# - We recommend that you convert _evt_time_ to seconds, e.g. using _F.col("evt_time").cast(long)_

# %%
## TODO - any imports here
from pyspark.sql.functions import array, explode, struct, lit, concat_ws, date_format, first

# %%
# %%time
planned_evt_cols = F.array(
    F.struct(F.lit("arr").alias("evt_type"), F.col("arr_time_rel").cast("long").alias("evt_time")),
    F.struct(F.lit("dep").alias("evt_type"), F.col("dep_time_rel").cast("long").alias("evt_time")))
actual_evt_cols = F.array(
    F.struct(F.lit("arr").alias("evt_type"), F.col("arr_actual_rel").cast("long").alias("evt_time")),
    F.struct(F.lit("dep").alias("evt_type"), F.col("dep_actual_rel").cast("long").alias("evt_time")))

# %%
# %%time
## TODO - create the planned_df table
planned_df = (
    trip_filter_sequences_df
      .select("bpuic","sequence", F.explode(planned_evt_cols).alias("evt"))
      .select("bpuic","sequence","evt.evt_type","evt.evt_time")
      .withColumn("trip_id", F.lit(target_trip)))

# %%
# %%time
## TODO - create the actual_df table
actual_df = (
    trip_filter_sequences_df
      .withColumn("trip_id", F.concat_ws("_", F.lit(target_trip), F.date_format("operating_day","yyyy-MM-dd")))
      .select("bpuic","sequence", F.explode(actual_evt_cols).alias("evt"), "trip_id")
      .select("bpuic","sequence", F.col("evt.evt_type").alias("evt_type"), F.col("evt.evt_time").alias("evt_time"),"trip_id"))

# %%
# %%time
# sanity check
planned_df.printSchema()
actual_df.printSchema()
planned_df.show(5, truncate=False)
actual_df.show(5,  truncate=False)

# %% [markdown]
# Finally, create the the union of the _actual_df_ and the _planned_df_ DataFrames (append the rows) and execute a _pivot_ operation on the union.
#
# Pivoting in Spark is a technique used to transform data from a long format, where rows in a single column may combine values from multiple entities (i.e. different operating_day) into a wide format, where each unique value of a column in the long format becomes its own column in the wide format. Essentially, it "spreads" out the data from one column across multiple columns based on a grouping rules. Note that pivot is a _GroupedData_ operation, it requires a _groupBy_.
#
# For example:
# - We are pivoting on the _trip_id_ column, which means we want each unique _trip_id_ to become its own column in the resulting dataframe.
# - For each group, defined by _bpuic_, _evt_type_ (arrival or departure) and _sequence_, we want to select the first _evt_time_ for each unique combination of _bpuic_, _evt_type_ and _sequence_ and copy it to the corresponding _trip_id_ column.
#
# So, pivoting reorganizes the data, turning a single column's unique values into new columns, making it easier to analyze and compare data across different entities (like trips in this case).
#
# See:
# * [Spark pivot (SQL)](https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-pivot.html)
# * [Spark pivot (python)](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.pivot.html#pyspark.sql.GroupedData.pivot)

# %%
# %%time
## TODO - execute the pivot as explained above
union_df = planned_df.unionByName(actual_df)
trips_pivoted_df = (
    union_df
      .groupBy("bpuic","sequence","evt_type")
      .pivot("trip_id")
      .agg(first("evt_time"))
      .orderBy("sequence","evt_type"))

# %%
trips_pivoted_df.createOrReplaceTempView("trips_pivoted_df")

# %%
trips_pivoted_df.printSchema()

# %%
trips_pivoted_df.select('bpuic','sequence','evt_type', "85:151:TL031-4506262507505612", "85:151:TL031-4506262507505612_2024-01-03", "85:151:TL031-4506262507505612_2024-01-04").show()

# %%
planned_df.filter("sequence=1").groupBy("evt_type").agg(F.col("evt_type"),F.min("evt_time"),F.max("evt_time")).show(truncate=False)

# %%
actual_df.filter("sequence=1").groupBy("evt_type").agg(F.col("evt_type"),F.min("evt_time"),F.max("evt_time")).show(truncate=False)

# %%
trips_pivoted_df.toPandas().drop(columns=['bpuic','evt_type']).plot(x='sequence',legend=False, )

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ---
# ## PART III: SBB Delay Model building (20 points)
#
# In the final segment of this assignment, your task is to tackle the prediction of SBB delays within the Lausanne region.
#
# To maintain simplicity, we've narrowed down the scope to building and validating a model capable of predicting delays exceeding 5 minutes. The model will classify delays as 0 if they're less than 5 minutes, and 1 otherwise. That said, you're encouraged to explore regression models if you'd like to predict delays as continuous values for more granular insights.
#
# This problem offers ample room for creativity, allowing for multiple valid solutions. We provide a structured sequence of steps to guide you through the process, but beyond that, you'll navigate independently. By this stage, you should be adept in utilizing the Spark API, enabling you to explore the Spark documentation and gather all necessary information.
#
# Feel free to employ innovative approaches and leverage methods and data acquired in earlier sections of the assignment. This open-ended problem encourages exploration and experimentation.

# %% [markdown]
# ### III.a Feature Engineering - 8/20
#
#
# Construct a feature vector for training and testing your model.
#
# Best practices include:
#
# * Data Source Selection and Exploration:
#   - Do not hesitate to reuse the data from Lausanne created in assignment 2. Query the data directly from files into Spark DataFrames.
#   - Explore the data to understand its structure, identifying relevant features and potential issues such as missing or null values.
#
# * Data Sanitization:
#   - Clean up null values and handle any inconsistencies or outliers in the data.
#
# * Historical Delay Computation:
#   - Utilize the SBB historical istdaten to compute historical delays, incorporating this information into your feature vector.
#   - Experiment with different ways to represent historical delays, such as aggregating delays over different time periods or considering average delays for specific routes or stations.
#
# * Incorporating Additional Data Sources:
#   - Integrate other relevant data sources, **at a minimum, integrate weather data history** from the previous questions into your feature vector.
#   - Explore how these additional features contribute to the predictive power of your model and how they interact with the primary dataset.
#
# * Feature Vector Construction using Spark MLlib:
#   - Utilize [`Spark MLlib`](https://spark.apache.org/docs/latest/ml-features.html). methods to construct the feature vector for your model.
#   - Consider techniques such as feature scaling, transformation, and selection to enhance the predictive performance of your model.
#

# %%
"""
This code cell creates a new spark df of all the trips by TL in the lausanne region. The code also performs some basic
data cleaning and filtering to ensure that only planned, successful trips with non-null arrival times and departure
times are inlcuded.
"""

raw_tl_df = (spark.table("iceberg.sbb.istdaten")).filter(F.col("operator_abrv") == "TL")
raw_tl_df.createOrReplaceTempView("raw_tl_df")
print("Showing raw_tl_df and its schema...")
raw_tl_df.show(5, truncate=False)
raw_tl_df.printSchema()

print("Show lausanne_stops_df and its schema...")
lausanne_stops_df.show(5, truncate=False)
lausanne_stops_df.printSchema()

raw_trip_data_df = raw_tl_df.alias("raw_tl_df").join(
    lausanne_stops_df.alias("lausanne_stops_df"),     
    raw_tl_df.bpuic == lausanne_stops_df.stop_id, 
    how="inner").select(
        "raw_tl_df.operating_day",
        "raw_tl_df.trip_id",
        "raw_tl_df.operator_id",
        "raw_tl_df.operator_abrv",
        "raw_tl_df.operator_name",
        "raw_tl_df.product_id",
        "raw_tl_df.line_id",
        "raw_tl_df.line_text",
        "raw_tl_df.circuit_id",
        "raw_tl_df.transport",
        "raw_tl_df.unplanned",
        "raw_tl_df.failed",
        "raw_tl_df.bpuic",
        "raw_tl_df.stop_name",
        "raw_tl_df.arr_time",
        "raw_tl_df.arr_actual",
        "raw_tl_df.arr_status",
        "raw_tl_df.dep_time",
        "raw_tl_df.dep_actual",
        "raw_tl_df.dep_status",
        "raw_tl_df.transit",
        "lausanne_stops_df.stop_lat",
        "lausanne_stops_df.stop_lon"
    ).withColumn("dep_time_unix", F.unix_timestamp("dep_time")).cache()

print(f"Raw trip data count before basic filtering: {raw_trip_data_df.count()}")
raw_trip_data_df = raw_trip_data_df.filter(
    (F.col("unplanned") == False) & (F.col("failed") == False) &
    (F.col("arr_time").isNotNull()) & (F.col("arr_actual").isNotNull()) & # Need these for delay calculation
    (F.col("dep_time").isNotNull()) # Need this for weather joining
)
print(f"Raw trip data count after basic filtering: {raw_trip_data_df.count()}")

raw_trip_data_df.createOrReplaceTempView("raw_trip_data_df")
raw_trip_data_df.show(5, truncate=False)
raw_trip_data_df.printSchema()

# %%
"""
Now that we have all the trips in the lausanne area with more than 200 rows of data, we need to do a join between
raw_trip_data_df and weather_df based on the stop latitude, stop longitude, and the departure time. This will then
provide all the stop data as well as the weather data. Weather_df has a sampling rate of 30mins at each location.
"""
from pyspark.sql.functions import radians, sin, cos, atan2, sqrt, lit, col #Used in calculating haversine distance
from collections import OrderedDict #for deduping columns
from pyspark.sql.window import Window
from pyspark.sql.functions import abs, unix_timestamp, row_number #Used in calculating the closest weather station timestamp



EARTH_RADIUS = 6371.0

print("Finding nearest weather stations...")
weather_stations_active = spark.table("weather_stations").filter(
    (F.col("Active") == "True") & (F.col("lat").isNotNull()) & (F.col("lon").isNotNull())
).select(F.col("Name").alias("weather_station_name"), F.col("lat").alias("station_lat"), F.col("lon").alias("station_lon"))

stop_weatherStation_pairs_df = raw_trip_data_df.select("bpuic", "stop_lat", "stop_lon").distinct().crossJoin(
    weather_stations_active
)

stop_weatherStation_distance_df = stop_weatherStation_pairs_df.withColumn(
    "delta_lat", radians(col("stop_lat") - col("station_lat"))
).withColumn(
    "delta_lon", radians(col("stop_lon") - col("station_lon"))
).withColumn(
    "a", sin(col("delta_lat")/2)**2 + cos(radians(col("stop_lat"))) * cos(radians(col("station_lat"))) * sin(col("delta_lon")/2)**2
).withColumn(
    "c", 2 * atan2(sqrt(col("a")), sqrt(1-col("a")))
).withColumn(
    "distance", lit(EARTH_RADIUS) * col("c")
)

nearest_station_window = Window.partitionBy("bpuic").orderBy(F.asc("distance"))
nearest_weatherStation_df = stop_weatherStation_distance_df.withColumn(
    "rank", F.row_number().over(nearest_station_window)
).filter(F.col("rank") == 1).select("bpuic", "weather_station_name") # Use weather_station_name

# Join the nearest station name
raw_trip_data_with_station = raw_trip_data_df.join(
    nearest_weatherStation_df,
    on="bpuic",
    how="left" # Use left join in case a stop somehow doesn't match
)

# Prepare weather_df: select necessary columns and include precipitation
weather_data_to_join = weather_df.select(
    F.col("site").alias("weather_station_name"),
    F.col("valid_time_gmt"), # Unix timestamp
    "temp", "dewPt", "feels_like", "gust", "pressure", "rh", "vis", "wspd",
    "precip_hrly", "precip_total", # Added precipitation
    "clds", "wx_phrase", "day_ind" # Weather features
).filter(F.col("valid_time_gmt").isNotNull()) # Ensure weather timestamp exists

# Join based on station name and a broad time window (+/- 1 hour)
print("Joining trip data with potential weather matches...")
potential_weather_matches_df = raw_trip_data_with_station.join(
    weather_data_to_join,
    on="weather_station_name", # Match column names
    how="left" # Keep trip even if weather is missing temporarily
).where(
    abs(F.col("dep_time_unix") - F.col("valid_time_gmt")) <= 3600
)

# Calculate the exact time difference
potential_weather_matches_df = potential_weather_matches_df.withColumn(
    "time_diff_weather_dep", abs(F.col("dep_time_unix") - F.col("valid_time_gmt"))
)

# Select the single closest weather reading in time for each unique stop event
print("Selecting the temporally closest weather reading...")
closest_weather_window = Window.partitionBy(
    "operating_day", "trip_id", "bpuic", "dep_time" # Unique key for the event
).orderBy(F.asc("time_diff_weather_dep")) #THIS TAKES A LONG WHILE TO FINISH ~30 mins

ranked_weather_matches_df = potential_weather_matches_df.withColumn(
    "weather_rank", F.row_number().over(closest_weather_window)
)

# Filter to keep only the closest weather reading (rank=1) and drop helper columns
final_trip_weather_df = ranked_weather_matches_df.filter(F.col("weather_rank") == 1)\
    .drop("weather_rank", "time_diff_weather_dep", "distance", "delta_lat", "delta_lon", "a", "c").cache() # Clean up intermediate cols

print(f"Trip data count after joining closest weather: {final_trip_weather_df.count()}") 
final_trip_weather_df.printSchema()


# %%
"""
Here we calculate the labels for the ML pipeline later on. We decided to look at arrival delay rather than departure
delay. We also perform some minor feature expansions to seperate the time into hour, day, month, and year. 
"""


print("Calculating target label (arrival_delay > 5 min)...")
# Create a new column called arrival_delay_sec which is the difference between the planned arrival time and the actual
# arrival time. This is the delay in seconds
final_trip_weather_df = final_trip_weather_df.withColumn(
    "arrival_delay_sec",
    (F.unix_timestamp(F.col("arr_actual")) - F.unix_timestamp(F.col("arr_time")))
)

#Create the label based on whether the delay is greater than 5 minutes (300 secs).
delay_threshold = 300
final_trip_weather_df = final_trip_weather_df.withColumn(
    "label",
    F.when(F.col("arrival_delay_sec") > delay_threshold, 1).otherwise(0)
)


#Show the label distribution to see how balanced or unbalanced the dataset is
print("Label distribution:")
label_counts_df = final_trip_weather_df.groupBy("label").count()
label_counts = label_counts_df.collect()
label_dict = {row["label"]: row["count"] for row in label_counts}
print(label_dict)

total = sum(label_dict.values())
weights = {label: total / (len(label_dict) * count) for label, count in label_dict.items()}
# Adding weights for weighted logisitc regression

# Initialize expression to None
weight_expr = None

# Build the weight expression using all label values
for label_value, weight_value in weights.items():
    condition = (F.col("label") == label_value)
    if weight_expr is None:
        weight_expr = F.when(condition, F.lit(weight_value))
    else:
        weight_expr = weight_expr.when(condition, F.lit(weight_value))

final_trip_weather_df = final_trip_weather_df.withColumn("weight", weight_expr)


#Perform some minor feature expansions to seperate the time into hour, day, month, and year
# The idea behind this is that we might be able to find some yearly, monthly ... ect time patterns that would be hard to
# detect if we just had the timestamp. Theoretically now we can have an individual weight for each of these features or
# perform other sinuodial transforms to better encode these patterns.
final_trip_weather_df = final_trip_weather_df.withColumn("dep_year", F.year("dep_time")) # Added year for context
final_trip_weather_df = final_trip_weather_df.withColumn("dep_month", F.month("dep_time"))
final_trip_weather_df = final_trip_weather_df.withColumn("dep_dayofweek", F.dayofweek("dep_time")) # 1=Sun, 7=Sat
final_trip_weather_df = final_trip_weather_df.withColumn("dep_hour", F.hour("dep_time"))


# print("Calculating historical delay features...")

# # Window specification for stop-based history: Partition by stop, order by planned departure time
# # Look back 1 hour (3600 seconds) EXCLUDING the current row's time.
# stop_hist_window = Window.partitionBy("bpuic") \
#     .orderBy(F.col("dep_time_unix").cast("long")) \
#     .rangeBetween(-3600, -1) # Look back 1 hour, up to 1 second before current event

# # Window specification for line-based history: Partition by line, order by planned departure time
# line_hist_window = Window.partitionBy("line_text") \
#     .orderBy(F.col("dep_time_unix").cast("long")) \
#     .rangeBetween(-3600, -1) # Look back 1 hour, up to 1 second before current event

# # Calculate average arrival delay over the windows
# # The Avg function will return null if the window is empty. However we can handle this with the imputer later on when we
# # have to fix all the null values. 
# final_trip_weather_df = final_trip_weather_df.withColumn(
#     "avg_delay_stop_last_hour", F.avg("arrival_delay_sec").over(stop_hist_window)
# )
# final_trip_weather_df = final_trip_weather_df.withColumn(
#     "avg_delay_line_last_hour", F.avg("arrival_delay_sec").over(line_hist_window)
# )

# print("Historical delay features calculated.")

# Drop original timestamp columns not needed as features anymore
#final_trip_weather_df = final_trip_weather_df.drop("arr_time", "arr_actual", "dep_time")


# %%
"""
In this code block we define the features that we are going to use. We split them into numerical and categorical
features. It should be noted that things like the year, day, month, and hour are categorical features since they do not
have a numerical relationship. I.e. like 12pm is not 12 times more likely to create a delay than 1pm. Or like day 7 (a
sunday) is not 7 times more likely to create a delay than day 1 (a monday). So this is why we have to treat them as categorical
features rather than numeric.
"""


print("Defining feature columns...")
# Categorical Features: (Same as before) (Removed BPUIC, WeatherStationName) 
categorical_cols = [
    "product_id", "line_text", "transport", "day_ind", "dep_year", "dep_month", 
    "dep_dayofweek", "dep_hour"
]

# Numerical Features: (Now including historical delay features)
numerical_cols = [
    "stop_lat", "stop_lon",
    "temp", "dewPt", "feels_like", "pressure", "rh", "wspd",
    "precip_hrly", "precip_total",
    # "avg_delay_stop_last_hour", # Added historical feature
    # "avg_delay_line_last_hour"  # Added historical feature
]

#Explictly cast bpuic to string if it is an int
if 'bpuic' in final_trip_weather_df.columns and dict(final_trip_weather_df.dtypes)['bpuic'] == 'int':
     final_trip_weather_df = final_trip_weather_df.withColumn("bpuic", F.col("bpuic").cast(StringType()))

#Select all the columns we need from the final_trip_weather_df
existing_categorical = [c for c in categorical_cols if c in final_trip_weather_df.columns]
existing_numerical = [c for c in numerical_cols if c in final_trip_weather_df.columns]
final_cols_to_select = existing_categorical + existing_numerical + ["label", "weight"]

model_input_df = final_trip_weather_df.select(final_cols_to_select)

#This is just a sanity check to make sure that the columns we selected are actually in the new model_input_df.
print("Schema of model data:")
model_input_df.printSchema()
model_input_df.show(5, truncate=False)

# %%
"""
Here we drop any columns that have more than 50% null values. Since these are mostly null it doesnt make sense to impute
values to them.
"""

total_count = model_input_df.count()

for c in existing_categorical:
    null_count = model_input_df.select(F.count(F.when(F.col(c).isNull(), 1)).alias("nulls")).collect()[0]["nulls"]
    print(f"{c} has {null_count}/{total_count} nulls")
    if null_count/total_count >= 0.5:
        print(f"{c} has more than 50% null values. Dropping col...")
        existing_categorical.remove(c)

imputable_numerical_cols = []

for c in existing_numerical:
    null_count = model_input_df.select(F.count(F.when(F.col(c).isNull(), 1)).alias("nulls")).collect()[0]["nulls"]
    print(f"{c} has {null_count}/{total_count} nulls")
    if null_count/total_count >= 0.5:
        print(f"{c} has more than 50% null values. Dropping col...")
        existing_numerical.remove(c)
    elif null_count/total_count > 0.0:
        imputable_numerical_cols.append(c)


# %%
from pyspark.ml.feature import Imputer

"""
Here we will handle the case of missing values. We will make use of an imputer to fill in these values (where
appropriate) rather than just getting rid of them.
"""

if len(imputable_numerical_cols) > 0:
    print(f"Imputing values for {imputable_numerical_cols}.")

for col_name in existing_numerical: # Use list of columns that actually exist
     if col_name in model_input_df.columns:
         if dict(model_input_df.dtypes)[col_name] != 'double':
             print(f"Casting {col_name} to double.")
             model_input_df = model_input_df.withColumn(col_name, F.col(col_name).cast(DoubleType()))


# Imputer for numerical columns (will handle NaNs from weather AND historical calculations)
if imputable_numerical_cols:
    imputer = Imputer(
        inputCols=imputable_numerical_cols,
        outputCols=[f"{col}_imputed" for col in imputable_numerical_cols],
        strategy="mean" # Or median. Mean is often fine for delays unless very skewed.
    )
    print("Fitting imputer...")
    imputer_model = imputer.fit(model_input_df)
    print("Transforming data with imputer...")
    model_input_df = imputer_model.transform(model_input_df)

    # Drop original numerical columns, keep imputed ones
    cols_to_drop = imputable_numerical_cols
    model_input_df = model_input_df.drop(*cols_to_drop)

    # Rename imputed columns back to original names
    imputed_numerical_cols_final = []
    for col in imputable_numerical_cols:
        imputed_col_name = f"{col}_imputed"
        model_input_df = model_input_df.withColumnRenamed(imputed_col_name, col)
        imputed_numerical_cols_final.append(col)
else:
    imputed_numerical_cols_final = []


print(f"Data count after numerical imputation: {model_input_df.count()}")

# Cache the final features DataFrame ready for pipeline
final_feature_columns = existing_categorical + imputed_numerical_cols_final
features_final_df = model_input_df.select(final_feature_columns + ["label", "weight"]).cache()

print("Final features DataFrame prepared and cached.")
print("Schema ready for ML Pipeline:")
features_final_df.printSchema()
features_final_df.show(5, truncate=False) # Check imputed historical values

# %%
features_final_df.show(5, truncate=False)

# %% [markdown]
# ### III.b Model building - 6/20
#
# Utilizing the features generated in section III.a), your objective is to construct a model capable of predicting delays within the Lausanne region.
#
# To accomplish this task effectively:
#
# * Feature Integration:
#         - Incorporate the features created in section III.a) into your modeling pipeline.
#
# * Model Selection and Training:
#         - Explore various machine learning algorithms available in Spark MLlib to identify the most suitable model for predicting delays.
#         - Train the selected model using the feature vectors constructed from the provided data.

# %%
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

"""
Now that we have imputed all the missing values, we can begin with the model building. We will need to One-Hot Encode
the categorical features. 
"""


# For safety, let's redefine them based on features_final_df schema, assuming 'label' is the last col
all_cols_in_features_df = features_final_df.columns
label_col_name = "label"
feature_cols_in_features_df = [c for c in all_cols_in_features_df if c != label_col_name]

# Re-derive final_categorical_cols and final_numerical_cols based on their original definitions
# and what actually exists in features_final_df.
# This assumes 'existing_categorical' and 'imputed_numerical_cols_final' are correctly defined from cell [110] / [111]
# For robustness, explicitly use the lists that were used to create features_final_df
# If running cells independently, these might need to be re-established or passed.
# Assuming existing_categorical and imputed_numerical_cols_final are correctly defined and available:
final_categorical_cols = [c for c in existing_categorical if c in features_final_df.columns]
final_numerical_cols = [c for c in imputed_numerical_cols_final if c in features_final_df.columns]

# This will create the string index to be used in the OHE stage
index_stages = [
    StringIndexer(inputCol=col_name, outputCol=col_name + "_index", handleInvalid="skip")
    for col_name in final_categorical_cols
]

ohe_stages = [
    OneHotEncoder(inputCol=col_name + "_index", outputCol=col_name + "_vec")
    for col_name in final_categorical_cols
]


# Assembler for combining features (used by RF and GBT directly)
assembler_input_cols_lr = [c + "_vec" for c in final_categorical_cols] + final_numerical_cols
vector_assembler_lr = VectorAssembler(inputCols=assembler_input_cols_lr, outputCol="features", handleInvalid="skip")

# Base transformation stages (excluding scaling) - these are common
lr_transformer_stages = index_stages + ohe_stages + [vector_assembler_lr]


# Scaler (for Logistic Regression) - applied AFTER vector assembly
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)



assembler_input_cols_rf_gbt = [c + "_index" for c in final_categorical_cols] + final_numerical_cols
vector_assembler_rf_gbt = VectorAssembler(inputCols=assembler_input_cols_rf_gbt, outputCol="features", handleInvalid="skip")
common_transformer_stages_rf_gbt = index_stages + [vector_assembler_rf_gbt]



# %% [markdown]
# ### Logistic Regression

# %%
lr_stages = lr_transformer_stages + [scaler] + [
    LogisticRegression(
        featuresCol="scaledFeatures", # Use scaled features
        labelCol="label",
        maxIter=10, # Keep low for initial run, tune later
        regParam=0.01,
        elasticNetParam=0.0, # L2 regularization
        weightCol="weight"
    )
]
lr_pipeline = Pipeline(stages=lr_stages)


# %% [markdown]
# ### Random Forest

# %%
rf_stages = common_transformer_stages_rf_gbt + [
    RandomForestClassifier(
        featuresCol="features", # Use non-scaled features
        labelCol="label",
        numTrees=100,       # Default: 20, can increase
        maxDepth=5,         # Default: 5, good starting point
        seed=42,            # For reproducibility
        maxBins=64,
        weightCol="weight" 
    )
]
rf_pipeline = Pipeline(stages=rf_stages)

# %% [markdown]
# ### Gradient Boosted Trees

# %%
# Pipeline 3: Gradient Boosted Trees (with Original Features from VectorAssembler)
gbt_stages = common_transformer_stages_rf_gbt

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    maxDepth=5,
    maxBins=64,
    weightCol="weight",
    seed=42
)

gbt_stages = common_transformer_stages_rf_gbt + [gbt]

gbt_pipeline = Pipeline(stages=gbt_stages)

# %% [markdown]
# ### Training Pipeline
#

# %%
print("Splitting data into training and test sets (80/20)...")
# Ensure features_final_df is cached (done at the end of your previous cell)
(train_data, test_data) = features_final_df.randomSplit([0.8, 0.2], seed=42)

train_data.cache()
test_data.cache()
print(f"Training data count: {train_data.count()}")
print(f"Test data count: {test_data.count()}")


print("\n--- Training Models ---")
models_to_train_and_evaluate = {
    "Logistic Regression": lr_pipeline,
    "Random Forest": rf_pipeline,
    "Gradient Boosted Trees": gbt_pipeline
}

trained_models = {} # To store the fitted PipelineModels
for model_name, pipeline_definition in models_to_train_and_evaluate.items():
    print(f"Training {model_name}...")
    try:
        fitted_model = pipeline_definition.fit(train_data)
        trained_models[model_name] = fitted_model
        print(f"{model_name} training complete.")
    except Exception as e:
        print(f"ERROR training {model_name}: {e}")
        trained_models[model_name] = None # Mark as failed

print("\nAll model training attempted.")


# %% [markdown]
# ### III.c Model evaluation - 6/20
#
# * Evaluate the performance of your model
#     * Usie appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.
#     * Utilize techniques such as cross-validation to ensure robustness and generalizability of your model.
#
# * Interpretation and Iteration:
#     * Interpret the results of your model to gain insights into the factors influencing delays within the Lausanne region.
#     * Iterate III.a)on your model by fine-tuning hyperparameters, exploring additional feature engineering techniques, or experimenting with different algorithms to improve predictive performance.
#

# %% [markdown]
# ### Evaluating Models
#
# ```
#
# --- Evaluating Models ---
#
# --- Evaluating Logistic Regression ---
# Generated predictions for Logistic Regression.
#                                                                                 
# AUC = 0.7451
# Accuracy = 0.6498
# F1 Score (Weighted) = 0.7412
# Precision (for Label 1) = 0.1151
# Recall (for Label 1) = 0.7198
# F1 Score (for Label 1) = 0.7412
# Precision (for Label 0) = 0.9729
# Recall (for Label 0) = 0.6454
# F1 Score (for Label 0) = 0.7412
#
# --- Evaluating Random Forest ---
# Generated predictions for Random Forest.
#                                                                                 
# AUC = 0.6875
# Accuracy = 0.9398
# F1 Score (Weighted) = 0.9106
# Precision (for Label 1) = 0.0000
# Recall (for Label 1) = 0.0000
# F1 Score (for Label 1) = 0.9106
# Precision (for Label 0) = 0.9398
# Recall (for Label 0) = 1.0000
# F1 Score (for Label 0) = 0.9106
#
# Feature Importances (Random Forest):
#   line_text_index: 0.3548
#   dep_hour_index: 0.2841
#   transport_index: 0.1172
#   product_id_index: 0.1014
#   dep_month_index: 0.0565
#   dep_year_index: 0.0350
#   pressure: 0.0196
#   precip_hrly: 0.0087
#   rh: 0.0081
#   feels_like: 0.0076
#   day_ind_index: 0.0032
#   temp: 0.0025
#   wspd: 0.0013
#   dep_dayofweek_index: 0.0000
#   dewPt: 0.0000
#
# --- Evaluating Gradient Boosted Trees ---
# Generated predictions for Gradient Boosted Trees.
# [Stage 3716:========================================>          (159 + 29) / 200]
# AUC = 0.7580
# Accuracy = 0.9398
# F1 Score (Weighted) = 0.9107
# Precision (for Label 1) = 0.6343
# Recall (for Label 1) = 0.0005
# F1 Score (for Label 1) = 0.9107
# Precision (for Label 0) = 0.9398
# Recall (for Label 0) = 1.0000
# F1 Score (for Label 0) = 0.9107
#
# Feature Importances (Gradient Boosted Trees):
#   line_text_index: 0.3645
#   dep_hour_index: 0.2772
#   dep_dayofweek_index: 0.1210
#   dep_month_index: 0.1165
#   dep_year_index: 0.0906
#   precip_hrly: 0.0061
#   temp: 0.0057
#   pressure: 0.0054
#   rh: 0.0047
#   day_ind_index: 0.0034
#   dewPt: 0.0025
#   feels_like: 0.0024
#   product_id_index: 0.0000
#   transport_index: 0.0000
#   wspd: 0.0000
# ```
#
# Here are the rough model evaluations and feature importance. This may slightly change between runs due to the random initializations.
#

# %%
print("\n--- Evaluating Models ---")

evaluation_results_summary = {}

# For AUC, rawPredictionCol is typically used (often contains scores before thresholding)
# Most Spark classifiers output rawPrediction, probability, and prediction
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# For other metrics, predictionCol (final 0/1 class) is used
evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

for model_name, fitted_pipeline_model in trained_models.items():
    if fitted_pipeline_model is None:
        print(f"Skipping evaluation for {model_name} (failed training).")
        continue

    print(f"\n--- Evaluating {model_name} ---")

    # 1. Make Predictions
    try:
        predictions = fitted_pipeline_model.transform(test_data)
        predictions.cache() # Cache predictions for multiple evaluations on this model's output
        print(f"Generated predictions for {model_name}.")
        # predictions.select("label", "rawPrediction", "probability", "prediction").show(5, truncate=False) # Optional check
    except Exception as e:
        print(f"ERROR generating predictions for {model_name}: {e}")
        if 'predictions' in locals() and predictions.is_cached: predictions.unpersist()
        continue


    # 2. Calculate Metrics
    try:
        auc = -1.0 # Default if rawPrediction is not available
        # Some models might not produce 'rawPrediction' if not applicable (though common for binary classifiers)
        # Or if featureCol for model was set incorrectly.
        if "rawPrediction" in predictions.columns:
            auc = evaluator_auc.evaluate(predictions)
        else:
            print(f"Warning: 'rawPrediction' column not found for {model_name}. AUC cannot be calculated this way.")

        accuracy = evaluator_multi.setMetricName("accuracy").evaluate(predictions)
        f1_weighted = evaluator_multi.setMetricName("f1").evaluate(predictions) # Default F1 is weighted
        precision_1 = evaluator_multi.setMetricName("precisionByLabel").setMetricLabel(1).evaluate(predictions)
        recall_1 = evaluator_multi.setMetricName("recallByLabel").setMetricLabel(1).evaluate(predictions)
        f1_1 = evaluator_multi.setMetricName("f1").setMetricLabel(1).evaluate(predictions) # F1 for label 1
        precision_0 = evaluator_multi.setMetricName("precisionByLabel").setMetricLabel(0).evaluate(predictions)
        recall_0 = evaluator_multi.setMetricName("recallByLabel").setMetricLabel(0).evaluate(predictions)
        f1_0 = evaluator_multi.setMetricName("f1").setMetricLabel(0).evaluate(predictions) # F1 for label 0


        evaluation_results_summary[model_name] = {
            "AUC": auc,
            "Accuracy": accuracy,
            "F1 (Weighted)": f1_weighted,
            "Precision (Label 1)": precision_1,
            "Recall (Label 1)": recall_1,
            "F1 (Label 1)": f1_1,
            "Precision (Label 0)": precision_0,
            "Recall (Label 0)": recall_0,
            "F1 (Label 0)": f1_0
        }

        print(f"AUC = {auc:.4f}")
        print(f"Accuracy = {accuracy:.4f}")
        print(f"F1 Score (Weighted) = {f1_weighted:.4f}")
        print(f"Precision (for Label 1) = {precision_1:.4f}")
        print(f"Recall (for Label 1) = {recall_1:.4f}")
        print(f"F1 Score (for Label 1) = {f1_1:.4f}")
        print(f"Precision (for Label 0) = {precision_0:.4f}")
        print(f"Recall (for Label 0) = {recall_0:.4f}")
        print(f"F1 Score (for Label 0) = {f1_0:.4f}")


        # Feature Importance (for tree-based models)
        if model_name in ["Random Forest", "Gradient Boosted Trees"]:
             print(f"\nFeature Importances ({model_name}):")
             try:
                 model_stage_in_pipeline = fitted_pipeline_model.stages[-1] # The classifier is the last stage
                 if hasattr(model_stage_in_pipeline, "featureImportances"):
                     importances = model_stage_in_pipeline.featureImportances
                     va_stage = next(s for s in fitted_pipeline_model.stages if isinstance(s, VectorAssembler))
                     
                     # Attempt to get feature names from VectorAssembler metadata
                     feature_names_from_va = []
                     try:
                         attrs = sorted(
                             (attr["idx"], attr["name"]) for attr_type in va_stage.getOutputMetadata()["ml_attr"]["attrs"]
                             for attr in va_stage.getOutputMetadata()["ml_attr"]["attrs"][attr_type]
                         )
                         feature_names_from_va = [name for idx, name in attrs]
                     except Exception: # Fallback if metadata is not as expected
                         feature_names_from_va = va_stage.getInputCols()

                     if len(feature_names_from_va) == len(importances):
                        feature_importance_list = sorted(list(zip(feature_names_from_va, importances)), key=lambda x: x[1], reverse=True)
                        for f_name, f_importance in feature_importance_list[:15]: # Show top 15
                            print(f"  {f_name}: {f_importance:.4f}")
                     else:
                         print(f"  Could not match feature names to importances vector. Assembler names: {len(feature_names_from_va)}, Importances: {len(importances)}")
                 else:
                     print(f"  Feature importances attribute not found for {model_name} stage.")
             except Exception as e_fi_outer:
                 print(f"  Error accessing/processing feature importances: {e_fi_outer}")

    except Exception as e:
        print(f"ERROR evaluating {model_name}: {e}")
    finally:
        if 'predictions' in locals() and predictions.is_cached: predictions.unpersist()

# %%
print("\n--- Model Comparison Summary ---")
# Header
header = f"{'Model':<25} | {'AUC':<10} | {'Accuracy':<10} | {'F1 (W)':<10} | {'P (L1)':<10} | {'R (L1)':<10} | {'F1 (L1)':<10} | {'P (L0)':<10} | {'R (L0)':<10} | {'F1 (L0)':<10}"
print(header)
print("-" * len(header))

best_model_name_overall = None
# Choose a primary metric for "best" model selection, e.g., F1 for Label 1 or AUC
primary_metric_for_selection = "F1 (Label 1)"
best_primary_metric_value = -1

for model_name_iter, metrics in evaluation_results_summary.items():
    auc_val = metrics.get('AUC', -1)
    acc_val = metrics.get('Accuracy', -1)
    f1_w_val = metrics.get('F1 (Weighted)', -1)
    p_l1_val = metrics.get('Precision (Label 1)', -1)
    r_l1_val = metrics.get('Recall (Label 1)', -1)
    f1_l1_val = metrics.get('F1 (Label 1)', -1)
    p_l0_val = metrics.get('Precision (Label 0)', -1)
    r_l0_val = metrics.get('Recall (Label 0)', -1)
    f1_l0_val = metrics.get('F1 (Label 0)', -1)
    print(f"{model_name_iter:<25} | {auc_val:<10.4f} | {acc_val:<10.4f} | {f1_w_val:<10.4f} | {p_l1_val:<10.4f} | {r_l1_val:<10.4f} | {f1_l1_val:<10.4f} | {p_l0_val:<10.4f} | {r_l0_val:<10.4f} | {f1_l0_val:<10.4f}")

    current_primary_metric = metrics.get(primary_metric_for_selection, -1)
    if current_primary_metric > best_primary_metric_value:
        best_primary_metric_value = current_primary_metric
        best_model_name_overall = model_name_iter

print("-" * len(header))
if best_model_name_overall:
    print(f"\nSelected best model based on highest {primary_metric_for_selection}: {best_model_name_overall} ({best_primary_metric_value:.4f})")
    # Example: You could now save this best model
    # best_pipeline_model_to_save = trained_models[best_model_name_overall]
    # best_pipeline_model_to_save.write().overwrite().save(f"/user/{username}/assignment-3/best_delay_model_pipeline")
else:
    print("\nNo models evaluated successfully or primary metric not found.")

# %% [markdown]
# <font color="#7777ff">
#
# ### Ensembling Model Prediction
#
# ```
# Metrics for Ensemble Model
# AUC = 0.7617
# Accuracy = 0.6623
# F1 Score (Weighted) = 0.7507
# Precision (for Label 1) = 0.1202
# Recall (for Label 1) = 0.7296
# F1 Score (for Label 1) = 0.7507
# Precision (for Label 0) = 0.9743
# Recall (for Label 0) = 0.6580
# F1 Score (for Label 0) = 0.7507
# ```
#
# We noticed that ensembling the models barely improved performance during. The exact values may slightly fluctuate due to
# the randomized initializations of the models, however large fluctuations were not observed during multiple runs. So our
# final choose model will be the Gradient Boosted Trees due to its high F1 score and improved precision and recall metrics for both labels
# over Logistic Regression and Random Forests

# %%
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

def extract_positive_class(v):
    return float(v[1]) if v is not None else None

extract_prob_udf = udf(extract_positive_class, DoubleType())

model1 = trained_models["Logistic Regression"]
model2 = trained_models["Random Forest"]
model3 = trained_models["Gradient Boosted Trees"]

# Create a unique ID for each row to allow for joining later
test_data_with_id = test_data.withColumn("row_id", monotonically_increasing_id())

# Make predictions from all the models based on the test data
pred1 = model1.transform(test_data_with_id).select("row_id", "label", "probability").withColumnRenamed("probability", "prob1")
pred2 = model2.transform(test_data_with_id).select("row_id", "probability").withColumnRenamed("probability", "prob2")
pred3 = model3.transform(test_data_with_id).select("row_id", "probability").withColumnRenamed("probability", "prob3")


ensemble_df = pred1.join(pred2, on="row_id").join(pred3, on="row_id")

ensemble_df = ensemble_df \
    .withColumn("prob1_class1", extract_prob_udf("prob1")) \
    .withColumn("prob2_class1", extract_prob_udf("prob2")) \
    .withColumn("prob3_class1", extract_prob_udf("prob3")) \
    .withColumn("avg_prob", (F.col("prob1_class1") + F.col("prob2_class1") + F.col("prob3_class1")) / 3) \
    .withColumn("final_prediction", (F.col("avg_prob") > 0.5).cast("double"))

ensemble_df.show(5, truncate=False)
ensemble_df.printSchema()

evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="avg_prob", metricName="areaUnderROC")

evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="final_prediction")

auc = evaluator_auc.evaluate(ensemble_df)

accuracy = evaluator_multi.setMetricName("accuracy").evaluate(ensemble_df)
f1_weighted = evaluator_multi.setMetricName("f1").evaluate(ensemble_df) # Default F1 is weighted
precision_1 = evaluator_multi.setMetricName("precisionByLabel").setMetricLabel(1).evaluate(ensemble_df)
recall_1 = evaluator_multi.setMetricName("recallByLabel").setMetricLabel(1).evaluate(ensemble_df)
f1_1 = evaluator_multi.setMetricName("f1").setMetricLabel(1).evaluate(ensemble_df) # F1 for label 1
precision_0 = evaluator_multi.setMetricName("precisionByLabel").setMetricLabel(0).evaluate(ensemble_df)
recall_0 = evaluator_multi.setMetricName("recallByLabel").setMetricLabel(0).evaluate(ensemble_df)
f1_0 = evaluator_multi.setMetricName("f1").setMetricLabel(0).evaluate(ensemble_df) # F1 for label 0


print("\nMetrics for Ensemble Model")
print(f"AUC = {auc:.4f}")
print(f"Accuracy = {accuracy:.4f}")
print(f"F1 Score (Weighted) = {f1_weighted:.4f}")
print(f"Precision (for Label 1) = {precision_1:.4f}")
print(f"Recall (for Label 1) = {recall_1:.4f}")
print(f"F1 Score (for Label 1) = {f1_1:.4f}")
print(f"Precision (for Label 0) = {precision_0:.4f}")
print(f"Recall (for Label 0) = {recall_0:.4f}")
print(f"F1 Score (for Label 0) = {f1_0:.4f}")




# %% [markdown]
# <font color="#7777ff">
#
# ## Now we perform Cross Validation to check for overfitting
#
# Performing the Cross Validation with hyperparameter tuning takes a while to run. The best model parameters will be posted
# here for convenience

# %%
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

paramGrid = ParamGridBuilder() \
    .addGrid(gbt.getParam("maxDepth"), [3, 5, 7]) \
    .addGrid(gbt.getParam("maxIter"), [10, 20, 50]) \
    .addGrid(gbt.getParam("stepSize"), [0.1, 0.2]) \
    .build()


cv = CrossValidator(
    estimator=gbt_pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5,           # 5-fold CV
    parallelism=5,
    seed=42
)

# Fit on training data
print("Fitting cross-validated GBT model on training data...")
cv_model = cv.fit(train_data)

# Best model
best_model = cv_model.bestModel

# Evaluate on train_data
print("\nEvaluating best model on full training set...")
train_predictions = best_model.transform(train_data)
train_auc = evaluator.evaluate(train_predictions)
print(f"Training AUC: {train_auc:.4f}")

# Cross-validation metrics (average AUC per param set)
print("\nCross-validation average AUC scores for all parameter settings:")
for params, metric in zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics):
    param_str = {p.name: v for p, v in params.items()}
    print(f"Params: {param_str} => CV AUC: {metric:.4f}")

best_cv_auc = max(cv_model.avgMetrics)
print(f"\nBest average CV AUC: {best_cv_auc:.4f}")

# Evaluate on test_data
print("\nEvaluating best model on held-out test set...")
test_predictions = best_model.transform(test_data)
test_auc = evaluator.evaluate(test_predictions)
print(f"Test AUC: {test_auc:.4f}")


print("Printing best model summary")
best_gbt_model = best_model.stages[-1]
print(f"Best maxDepth: {best_gbt_model.getMaxDepth()}")
print(f"Best maxIter: {best_gbt_model.getMaxIter()}")
print(f"Best stepSize: {best_gbt_model.getStepSize()}")

# %%
#Evaluating the Best model


evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

accuracy = evaluator_multi.setMetricName("accuracy").evaluate(predictions)
f1_weighted = evaluator_multi.setMetricName("f1").evaluate(predictions) # Default F1 is weighted
precision_1 = evaluator_multi.setMetricName("precisionByLabel").setMetricLabel(1).evaluate(predictions)
recall_1 = evaluator_multi.setMetricName("recallByLabel").setMetricLabel(1).evaluate(predictions)
f1_1 = evaluator_multi.setMetricName("f1").setMetricLabel(1).evaluate(predictions) # F1 for label 1
precision_0 = evaluator_multi.setMetricName("precisionByLabel").setMetricLabel(0).evaluate(predictions)
recall_0 = evaluator_multi.setMetricName("recallByLabel").setMetricLabel(0).evaluate(predictions)
f1_0 = evaluator_multi.setMetricName("f1").setMetricLabel(0).evaluate(predictions) # F1 for label 0

model_name_iter = Optimized_GBT


header = f"{'Model':<25} | {'AUC':<10} | {'Accuracy':<10} | {'F1 (W)':<10} | {'P (L1)':<10} | {'R (L1)':<10} | {'F1 (L1)':<10} | {'P (L0)':<10} | {'R (L0)':<10} | {'F1 (L0)':<10}"
print(header)
print("-" * len(header))
print(f"{model_name_iter:<25} | {auc_val:<10.4f} | {acc_val:<10.4f} | {f1_w_val:<10.4f} | {p_l1_val:<10.4f} | {r_l1_val:<10.4f} | {f1_l1_val:<10.4f} | {p_l0_val:<10.4f} | {r_l0_val:<10.4f} | {f1_l0_val:<10.4f}")
print("-" * len(header))

# %% [markdown]
# <font color="#7777ff">
#
# ## Improvements for the model
# ### Weighted Regression
# We noticed there was a large class imbalance where the majority of the trips had a delay of less than 5 minutes. So
# initially the models would mostly predict 0s for all the inputs regardless of the features. So we made use of the
# weighted regression features to numerically increase the importance of the minority class. This works by penalizing
# the model more when it incorrectly predicts a trip. This massively boosted the Precision, Recall and F1 metrics of all
# the models. From our testing we found that Weighted Gradient Boosted Trees (WGBT) narrowly beat out the Weighted Random Forest and
# Weighted Logistic Regression models. Consequently, we choose The WGBT model as our final model.
#
# ### Hyperparameter tuning using Cross Validation
# Once we choose the best model, we tuned this further using Cross Validation. In Cross Validation, we split the training
# datatset in $K$ separate folds. $K-1$ folds will be used during training, while the $k^{\text{th}}$ fold will be used
# for testing. This allows us to perform hyperparameter tuning while ensuring there is no data leakage (i.e. an instance
# of the model will
# never be tested on data that it was trained on). The testing set from the original will now be used as a validation set
# to ensure there is no overfitting. During our testing we found that this Cross Validation Step takes A very long time to
# run. We noticed that it often took multiple hours to run, even after increasing the parallelism. Also, we noticed some
# errors would often appear in the output box, where the spark context would complain about RDDs that do not exist
# E.g. Error: `` Caused by: org.apache.spark.SparkException: Block rdd_1433_11 does not exist at
# org.apache.spark.errors.SparkCoreErrors ...``. Once this happened we often had to restart the whole process, which meant
# we didn't get much time to play around with the hyperparameters tuning. But the above 

# %% [markdown]
# <font color="#7777ff">
#
# ## Model Interpretation and Improvements for the Project
#
# When analysing the feature importance in the tree based methods we notice that the main feature used by both models is
# the `line_text` feature ("line_text_index" in the Feature Importance cell above). This probably means that our model
# discovered a trend in our data where certain lines are more likely to be delayed than others. Perhaps this could
# correlate to busses that are frequently caught in rush hour traffic in the morning and in the evenings. This idea is
# further reinforced when considering that the next most important feature is the `departure time` ("dep_time_index").
# Meaning that if a mode of transport were to leave during the morning rush hour traffic time or during the evening rush
# hour time, it would be more likely to be delayed than if it were to depart at another time. We also see that our model
# pays significant attention to the `depature day of week` variable ("dep_dayofweek_index") which means that our model has
# found a string dependency between trip delays and also which day of the week the trip was meant to happen. This again
# makes sense since a trip is more likely to be delayed during weekdays rather than the weekend (due to the increased
# volume of road traffic during weekdays), so the day of the week should play a role in determining whether a trip
# is delayed. The WGBT model also found a dependence between the month the journey was supposed to occur and whether it
# was going to be delayed. This connection was a bit harder to explain since there is no obvious connection between the
# departure month and the delay. We think that this may be down to large scale events that happen during certain month
# (such as Balélec) of it could be down to monthly trends such as a specific tourist season that increases the number of
# rented cars and therefore traffic on the road. We also note that the weather features computed do not play a significant
# part in deciding whether a delay will occur or not. We think that this might be due to the fact that the Lausanne area
# might not experience extreme weather patters that would significantly delay some trips over 5 minutes. However, it is
# important to note that the weather feature importance is not zero, for example features such as the temperature and
# precipitation are still used when making the final decision, indicating that the weather features play a minor but not
# insignificant role in decision making. While not included here, we think that perhaps adding some historical delay
# statistics might help to improve the model even further. This idea will be explored in the final project.

# %% [markdown]
# # That's all, folks!
#
# Be nice to other, do not forget to close your spark session.

# %%
spark.stop()
