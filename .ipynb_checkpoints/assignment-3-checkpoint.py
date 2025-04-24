# -*- coding: utf-8 -*-
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
groupName = 'Z9'

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
            .config("spark.executor.memory", "6g")\
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
spark.read.options(header=True).csv(f'/data/com-490/csv/weather_stations').withColumns({
      'lat': F.col('lat').cast('double'),
      'lon': F.col('lon').cast('double'),
    }).createOrReplaceTempView("weather_stations")

# %%
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

# %% [markdown]
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

# %% jupyter={"outputs_hidden": true}
# %%time
json_df.select(json_df.site,lowercase(json_df.site).alias('lowercase_site')).show(n=5)

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
from pyspark.sql.functions import countDistinct
weather_df.groupBy("year").agg(
    countDistinct("month", "dayofmonth").alias("num_days")
).orderBy("year").show()

# %% [markdown]
# Okay so we see that 2022, 2023, and 2024 are full years! Also, 2024 is a leap year :)

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
# So we notice how the biggest difference between our result and the exemplary one is that our data includes the year 2022, while the exemplary data shows 2023. Howver, according to Part 1C of the question, the years 2022, 2023, and 2024 are all considered full years. Therefore, I believe our result is still valid, as the question does not specifically exclude the year 2022.

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
# seems like a passed checkpoint!

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

# weather_df\
#     .select('valid_time_gmt', 'year', 'month', 'dayofmonth','hour', 'site', 'temp', sliding_3hour_avg) \
#     .select('valid_time_gmt', 'year', 'month', 'dayofmonth','hour', 'site', col('avg_temp').alias('temp'),hourly_rank) \
#     .filter('rank <= 5')\
#     .sort('valid_time_gmt','rank')\
#     .show()

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
# In my understanding, when a site has rank 1, it means it had the highest 3-hour average temperature among all sites for that specific hour.
#
# Interestingly though, I do not see rank 3 anywhere. Apparently, there’s no rank 3 because rank() skips ranks after ties. So if two sites tie at rank 2 (which happens often in our table), the next is rank 4.

# %% [markdown]
# ---
# ## PART II: SBB Network - Vehicle Journey Trajectories (20 points)

# %% [markdown]
# ### II.a Filter trips from SBB Istdaten - 4/20
#

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
# - Onyl trip ids that serve stops in the Lausanne region.
#     - Use the data available in _/data/com-490/labs/assignment-3/sbb_stops_lausanne_region.parquet_)
#     - Or use Trino to create your own list of stops in the greater Lausanne region (Lausanne and Ouest Lausannois).
#
# 💡 Note:
# - You may assume that the SBB timetables (_stops_, _stop_times_ etc), are valid for the full year in which they are published.
# - Filtering the trips based on both the TL operator and the presence of at least one stop in the only region served by this operator might seem redundant in this case. However, in a more general context, this approach allows us to reuse the same query for nation wide operators.
#

# %%
# %time
## Or use your own
lausanne_stops_df = spark.sql('SELECT DISTINCT * FROM parquet.`/data/com-490/labs/assignment-3/sbb_stops_lausanne_region.parquet`')
lausanne_stops_df.createOrReplaceTempView('lausanne_stops_df')

# %%
lausanne_stops_df.printSchema()

# %%
lausanne_stops_df.show(1)

# %%

# %%
# %%time
##TODO -- create the table containing a list of trip_id as specified above
istdaten_trips_df =

# %%
# %%time
istdaten_trips_df.createOrReplaceTempView('istdaten_trip_ids')

# %%
# %%time
istdaten_trips_df.count()

# %%
# %%time
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
istdaten_df = 

# %%

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

# %%
## TODO - create the logics you want to apply on the window

# %%
## TODO - create the table as described above
trip_sequences_df = 

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

# %%

# %%

# %%

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
## TODO - create the trip_filter_list_df as indicated above
trip_filter_list_df = 

# %%
# %%time
trip_filter_list_df.show()

# %% [markdown]
# **Second**, create _trip_filter_sequence_df_, a subset of _trip_sequence_df_ computed earlier that contains only the trips of interest happening on the days computed in _trip_filter_list_df_

# %%
## TODO - create the subset of trip_filter_sequence_df as indicated above
trip_filter_sequences_df = 

# %%

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

# %%
## TODO - create the planned_df table
planned_df = 

# %%
## TODO - create the actual_df table
actual_df = 

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
## TODO - execute the pivot as explained above
trips_pivoted_df = 

# %%
trips_pivoted_df.printSchema()

# %%
trips_pivoted_df.select('bpuic','sequence','evt_type', "85:151:TL031-4506262507505612", "85:151:TL031-4506262507505612_2024-01-03", "85:151:TL031-4506262507505612_2024-01-04").show()

# %%
planned_df.filter("sequence=1").groupBy("evt_type").agg(F.col("evt_type"),F.min("evt_time"),F.max("evt_time")).show(truncate=False)

# %%
actual_df.filter("sequence=1").groupBy("evt_type").agg(F.col("evt_type"),F.min("evt_time"),F.max("evt_time")).show(truncate=False)

# %%
trips_pivoted_df.toPandas().drop(columns=['bpuic','evt_type']).plot(x='sequence',legend=False)

# %% [markdown]
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
# TODO ...

# %%

# %%

# %%

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
# TODO ...

# %%

# %%

# %%

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

# %%

# %%

# %%

# %%

# %% [markdown]
# # That's all, folks!
#
# Be nice to other, do not forget to close your spark session.

# %%
spark.stop()

# %%
