# -*- coding: utf-8 -*-
# # First steps with HDFS
# ----


# ### About this exercise
#
# In the following exercises you will learn to obtain data from a remote site, and import it in our Hadoop's cluster HDFS.
#
# This is an important step, because HDFS is the default storage for your Hadoop applications and a first step toward your data lake.
#
# They are many ways you can interact with the HDFS file systems, including using the python client for HDFS. For the following exercises you will be using the `hdfs dfs` command line (or `hadoop fs`), which comes by default with the Hadoop distribution. The command line is a client, which sends command from your notebook running in a docker container to a remote HDFS service running on the Hadoop big data cluster.
#
# In the following exercise, we will use the term _jupyter environment_ to refer to the machine on which this notebook is running, and _HDFS_ to refer to the HDFS storage on the remote Hadoop cluster.
#
# This notebook must be run with the bash kernel.
#
# > _Important Note:_ for the sake of simplicity, the cluster is configured for basic security only. The settings in your notebook environment are such that they should prevent you from accidentally causing any damage to our distributed data storage. However, they can easily be bypassed - **do not attempt it**, there is nothing to be proven, and you will have to face the consequences when things go avry.

# ### Sanity check
#
# Before going further, make sure that the following command returns your EPFL (gaspar) username, and talk to us if this is not the case.

echo ${USER}

# ----
# ### First steps with the HDFS command line
#
# Execute the following `hdfs dfs` command.
#
# The output is the list of HDFS file system actions available via the hdfs command line. Notice how most of the commands behave like the familiar Linux file system commands.
#
# You can find more details about that on the [Hadoop file system shell](https://hadoop.apache.org/docs/r3.1.1/hadoop-project-dist/hadoop-common/FileSystemShell.html) manual.

hdfs dfs -usage

# -------------------
# ### Uploading data to HDFS


# **Q1:** As a first exercise, you will explore and add content to the cluster's HDFS file system using the `hdfs dfs` command.

# * We have created a home folder on HDFS for each of you, can you find yours?
#   - If not, you must talk to us and we will create one for you. You will not be able to do the exercises without it.
# * Create the sub-folders `week3/sbb/istdaten/` in your HDFS home folder.
# * Download and unzip the file `module-2a.zip` from the URL https://drive.switch.ch/index.php/s/BwScS245DivdabW/download to your _jupyter_ home folder.
#     * This file contains SBB istdaten (realtime) data published by [opentransportdata.ch](https://opentransportdata.swiss/en/dataset/istdaten) and converted by us.
# * Copy the deflated files to `week3/sbb/istdaten/...` on HDFS, _under the appropriate Hive partition_, using year, month, day (pay attention to the file names)
# * Change the owner rights of your HDFS `week3` sub-folder and its content hdfs to `-rwxrwxr--`.

# Hints:
#
# 1. You have two home folders, the home folder of this jupyter notebook environment, and the HDFS home folder. 
# 2. We use the linux [curl](https://www.man7.org/linux/man-pages/man1/curl.1.html) or [wget](https://www.man7.org/linux/man-pages/man1/wget.1.html) command line in order to copy the data to your jupyter home folder.
# 3. Use the `hdfs dfs` commands (`-ls`, `-mkdir`, `-moveFromLocal`, `-chmod`)
#    - Do not hesitate to use the dfs help, e.g. `hdfs dfs -help ls`, or look up the [online manual](https://hadoop.apache.org/docs/r3.1.1/hadoop-project-dist/hadoop-common/FileSystemShell.html).
# 4. The `${USER}` environment variable is set to your gaspar name. We recommend you to take advantage of this variable in your code when you initialize the paths under your HDFS home directory instead of hardcoding your gaspar ID. This will make your code more portable, and easier for others to try it, e.g. `hdfs dfs -ls /somepath/${USER}/subpath`.
# 5. Alternatively, `hdfs dfs -ls` without a path, or `./`, will list the content of your home folder on HDFS.
# 6. HDFS does not like spaces in filenames.

pwd

wget -c -O ./module-2a.zip https://drive.switch.ch/index.php/s/BwScS245DivdabW/download

unzip -n module-2a.zip

# TODO: find and show content of your HDFS home folder
hdfs dfs -ls

hdfs dfs -mkdir week3
hdfs dfs -mkdir week3/sbb
hdfs dfs -mkdir week3/sbb/istdaten

hdfs dfs -moveFromLocal ./2025-03-02_istdaten.parquet week3 
hdfs dfs -moveFromLocal ./2025-03-03_istdaten.parquet week3

hdfs dfs -ls week3

hdfs dfs -chmod 0700 .




# ----
# ### Exploring HDFS
#
# We have already copied some data on HDFS under the sub-folders `/data/com-490/csv`.
#
# **Q2:** Do you understand how the folders are structured?

hdfs dfs -ls /data/com-490/








# **Q3:** Use the `hdfs dfs -du` command to print a human readable summary of the total HDFS size footprint of the SBB istdaten data under `/data/com-490/csv/sbb/istdaten/` and the same data stored in parquet format under `/data/com-490/iceberg/sbb/istdaten/`
#
# Note:
# - Exactly the same information is encoded in each folder.
# - The two values show the size of the information without data replication and the actual size on disk with replication (x 3)

hdfs dfs -du -h /data/com-490/csv/sbb/istdaten/

hdfs dfs -du -h /data/com-490/iceberg/sbb/istdaten/



# ----------
# **That's all folks!**
#
# You have now mastered the very basics of HDFS. There is more, and we encourage you to read the litterature about HDFS. In particular the alternatives to HDFS, and the different optimization tricks such as optimum HDFS block sizes. Also familiarize yourself with the tradeoffs of data storage formats (ORC, parquet). You should also learn about the dos and don'ts of HDFS, such as optimum compression schemes. Remember don't gzip your files before importing them, because it is not splittable into blocks, bz2 is okay though however see the next exercises. What you save on storage size and network performance with compression, must be paid with CPU cost, and it is not always a winner.

# ----
# ### Cleaning up
#
# Uncomment the hdfs command below and run the cell if you wish to clean up your HDFS home folder.
#
# But not until you have completed all the exercises from the other notebook.

hdfs dfs -rm -r -skipTrash /user/${USER-error}/week3
rm module-2a.zip
rm *.parquet


