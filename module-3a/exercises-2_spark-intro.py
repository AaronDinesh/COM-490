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
# # Basic Spark 
#
# In the lecture we discussed -- now we'll try to actually use the framework for some basic operations. 
#
# In particular, this notebook will walk you through some of the basic [Spark RDD methods](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.html). As you'll see, there is a lot more to it than `map` and `reduce`.
#
# We will explore the concept of "lineage" in Spark RDDs and construct some simple key-value pair RDDs to write our first Spark applications.
#
# If you need a reminder of some of the python concepts discussed earlier, you can make use of the [python refresher notebook](exercises-1_python-refresher.py).

# %% [markdown]
# ----
# ## Spark connections
#
# For this week's exercises, we will be using the local installation of Spark. Some of the data is in the HDFS server.

# %% [markdown] slideshow={"slide_type": "slide"}
# ----
# ## Starting up the Spark runtime: initializing a `SparkContext` 
#
# The `SparkContext` provides you with the means of communicating with a Spark cluster. The Spark cluster in turn is controlled by a master which orchestrates pieces of work between the various executors. Every interaction with the Spark runtime happens through the `SparkContext` in one way or another. Creating a `SparkContext` is therefore the very first step that needs to happen before we do anything else.
#
# First, we need to create a SparkSession with the right configuration. Once the Spark Session is created, ou can connect to the driver's UI by getting its address from the configuration.

# %%
import os, sys
import getpass
from pyspark.sql import SparkSession

# %%
import pwd
from random import randrange

os.environ['SPARK_HOME']='/opt/spark-3.5.2-bin-hadoop3/'

sparkSession = SparkSession\
            .builder\
            .appName(pwd.getpwuid(os.getuid()).pw_name)\
            .master('yarn')\
            .config('spark.ui.port', randrange(4040, 4440, 5)) \
            .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path)) \
            .getOrCreate()

print('Follow this link to application UI:', sparkSession.conf.get("spark.driver.appUIAddress"))

# %%
type(sparkSession)

# %%
sc = sparkSession.sparkContext

# %% [markdown] slideshow={"slide_type": "slide"}
# Hurrah! We have a Spark Context! Now lets get some data into the Spark universe.

# %% [markdown]
# Each Spark application runs its own dedicated Web UI -- right-click (or command-click on Mac) the `Spark UI` link two cells above to get to the UI.
#
# You will find lot of nice information about the state of your Spark job, including stats on execution time of individual tasks, available memory on all of the workers, links to worker logs, etc. You will probably begin to appreciate some of this information when things start to go wrong...
#
# **Note**: this UI service is configured to support the http protocol only, however your browser will likely try to upgrade to https. If you get a connection error, try to open the URL in a private window using the http protocol.

# %% [markdown]
# ----
# ## Creating an RDD
#
# The basic object you will be working with is the Spark data abstraction called a Resilient Distributed Dataset (RDD). This class provides you with methods to execute work on your data using the Spark cluster. The simplest way of creating an RDD is by using the `parallelize` method to distribute an array of data among the executors:

# %%
data = range(100)
data_rdd = sc.parallelize(data)
print('Number of elements: ', data_rdd.count())
print('Sum and mean: ', data_rdd.sum(), data_rdd.mean())


# %% [markdown]
# ----
# ## Map/Reduce 
#
# Lets bring some of the simple python-only examples from the [python refresher notebook](exercises-1_python-refresher.py) into the Spark framework. The first map function we made was simply doubling the input array, so lets do this here. 
#
# Write the function `double_the_number` and then use this function with the `map` method of `data_rdd` to yield `double_rdd`:

# %%
def double_the_number(x) : 
    return x*2


# %%
help(data_rdd.map)

# %%
double_rdd = data_rdd.map(double_the_number)

# %% [markdown]
# Not much happened here - or at least, no tasks were launched (you can check the console and the Web UI). Spark simply recorded that the `data_rdd` maps into `double_rdd` via the `map` method using the `double_the_number` function. You can see some of this information by inspecting the RDD debug string: 

# %%
print(double_rdd.toDebugString().decode())

# %%
# comparing the first few elements of the original and mapped RDDs using take
print(data_rdd.take(10))
print(double_rdd.take(10))

# %% [markdown]
# Now if you go over to check on the stages in the Spark UI you'll see that jobs were run to grab data from the RDD. In this case, a single task was run since all the numbers needed reside in one partition. Here we used `take` to extract a few RDD elements, a very very very convenient method for checking the data inside the RDD and debugging your map/reduce operations. 
#
# Often, you will want to make sure that the function you define executes properly on the whole RDD. The most common way of forcing Spark to execute the mapping on all elements of the RDD is to invoke the `count` method: 

# %%
double_rdd.count()

# %% [markdown]
# If you now go back to the stages page, you'll see that four tasks were run for this stage. 

# %% [markdown]
# In our initial example of using `map` in pure python code, we also used an inline lambda function. For such a simple construct like doubling the entire array, the lambda function is much neater than a separate function declaration. This works exactly the same way here.

# %% [markdown]
# Map the `data_rdd` to `double_lambda_rdd` by using a lambda function to multiply each element by 2: 

# %%
double_lambda_rdd = data_rdd.map(lambda n : n*2)
print(double_lambda_rdd.take(10))

# %% [markdown]
# Finally, do a simple `reduce` step, adding up all the elements of `double_lambda_rdd`:

# %%
from operator import add
double_lambda_rdd.reduce(add)

# %% [markdown]
# (Spark RDDs actually have a `sum` method which accomplishes essentially the same thing)

# %%
double_lambda_rdd.sum()

# %% [markdown]
# ----
# ## Filtering
#
# A critical step in many analysis tasks is to filter down the input data. In Spark, this is another *transformation*, i.e. it takes an RDD and maps it to a new RDD via a filter function. The filter function needs to evaluate each element of the RDD to either `True` or `False`. 
#
# Use `filter` with a lambda function to select all values less than 10: 

# %%
filtered_rdd = data_rdd.filter(lambda n : n < 10)
filtered_rdd.count()

# %% [markdown]
# Of course we can now apply the `map` and double the `filtered_rdd` just as before: 

# %%
filtered_rdd.map(lambda n : n * 2).take(10)

# %% [markdown]
# Note that each RDD transformation returns a new RDD instance to the caller -- for example:

# %%
data_rdd.filter(lambda x: x % 2)

# %% [markdown]
# You can therefore string together many transformations without creating a separate instance variable for each step. Our `filter` + `map` step can therefore be combined into one. Note that if we surround the operations with "( )" we can make the code more readable by placing each transformation on a separate line: 

# %%
composite = (data_rdd.filter(lambda x: x % 2)
                     .map(lambda x: x*2))

# %% [markdown]
# Again, if you now look at the Spark UI you'll see that nothing actually happened -- no job was trigerred. The `composite` RDD simply encodes the information needed to create it. 
#
# If an action is executed that only requires a part of the RDD, only those parts will be computed. If we cache the RDD and only calculate a few of the elements, this will be made clear:

# %%
composite.cache()
composite.take(10)

# %% [markdown]
# If you look at the **Storage** tab in the Spark UI you'll see that just a quarter of the RDD is cached. Now if we trigger the full calculation, this will increase to 100%:

# %%
composite.count()

# %% [markdown]
# ----
# ## Key, value pair RDDs
#
# `key`,`value` pair data is the "bread and butter" of map/reduce programming. Think of the `value` part as the meat of your data and the `key` part as some crucial metadata. For example, you might have time-series data for CO$_2$ concentration by geographic location: the `key` might be the coordinates or a time window, and `value` the CO$_2$ data itself. 
#
# If your data can be expressed in this way, then the map/reduce computation model can be very convenient for pre-processing, cleaning, selecting, filtering, and finally analyzing your data. 
#
# Spark offers a `keyBy` method that you can use to produce a key from your data. In practice this might not be useful often but we'll do it here just to make an example: 

# %%
# key the RDD by x modulo 5
keyed_rdd = data_rdd.keyBy(lambda x: x%5)
keyed_rdd.take(20)

# %% [markdown]
# This created keys with values 0-4 for each element of the RDD. We can now use the multitude of `key` transformations and actions that the Spark API offers. For example, we can revisit `reduce`, but this time do it by `key`: 

# %% [markdown]
# ----
# ## `reduceByKey`

# %%
# use the add operator in the `reduceByKey` method
red_by_key = keyed_rdd.reduceByKey(add)
red_by_key.collect()

# %% [markdown]
# Unlike the global `reduce`, the `reduceByKey` is a *transformation* --> it returns another RDD. Often, when we reduce by key, the dataset size is reduced enough that it is safe to pull it completely out of Spark and into the Spark driver. A useful way of doing this is to automatically convert it to python dictionary for subsequent processing with the `collectAsMap` method:

# %%
red_dict = red_by_key.collectAsMap()
red_dict

# %%
# access by key
red_dict[0]

# %% [markdown]
# ----
# ## `groupByKey`
#
# If you want to collect the elements belonging to a key into a list in order to process them further, you can do this with `groupByKey`. Note that if you want to group the elements only to do a subsequent reduction, you are far better off using `reduceByKey`, because it does the reduction locally on each partition first before communicating the results to the other nodes. By contrast, `groupByKey` reshuffles the entire dataset because it has to group *all* the values for each key from all of the partitions. 

# %%
keyed_rdd.groupByKey().collect()

# %% [markdown]
# Note the ominous-looking `pyspark.resultiterable.Resultiterable`: this is exactly what it says, an iterable. You can turn it into a list or go through it in a loop. For example:

# %%
key, iterable = keyed_rdd.groupByKey().first()
list(iterable)

# %% [markdown]
# ----
# ## `sortBy`
#
# Use the `sortBy` method of `red_by_key` to return a list sorted by the sums in descending order and print it out. 

# %%
sorted_red = red_by_key.sortBy(sum,ascending=False).collect()
sorted_red

# %%
assert(sorted_red == [(4, 1030), (3, 1010), (2, 990), (1, 970), (0, 950)])

# %% [markdown]
# This concludes the brief tour of the Spark runtime -- we can now shut down the `SparkContex` by calling `sc.stop()`. This removes your job from the Spark cluster and cleans up the memory and temporary files on disk. 

# %%
sc.stop()

# %%

# %%
