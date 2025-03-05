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

# # 

# ---
# # SQL Queries on HDFS
#
# You can also query data stored on HDFS with Python (and other languages).
#
# In these exercise we use DuckDB to query the data.

# ---
# #### Querying HDFS with DuckDB - Single file
#
# DuckDb with the help of PyArrow/Fsspec can be used to query or update files on many storage systems, including HDFS.
#
# Import DuckDb, and make sure that the environment variable HADOOP_FS is set. This variable points to the URL of the HDFS Namenode server.

# +
import duckdb
import fsspec
from fsspec import filesystem
import os
import pyarrow as pa
import pyarrow.fs
import pyarrow.dataset as ds
import pyarrow.compute as pc

from pyarrow.dataset import CsvFileFormat
from pyarrow.csv import ParseOptions
from fsspec.implementations.arrow import ArrowFSWrapper
# -

hadoop_fs = os.environ.get('HADOOP_FS')
username = os.environ.get('USER')
print(f'HDFS namenode URL: {hadoop_fs}')
print(f'you are {username}')

hdfs_fs = pa.fs.HadoopFileSystem.from_uri(hadoop_fs)
hdfs_table=f'/user/{username}/week3/sbb/istdaten/'
print(hdfs_fs.get_file_info(hdfs_table))
hdfs_wrapper = ArrowFSWrapper(hdfs_fs)
hdfs_wrapper.ls(hdfs_table)

# First we will query one data file, in order to do so we need to tell DuckDB to understand the HDFS (`hdfs:`) protocol.
#
# This is done using the fsspec File System API abstraction.

duckdb.register_filesystem(filesystem('hdfs'))

# Now we can explore the various ways we can query the data with DuckDB.

# First, query a single file:

duck_handle = duckdb.read_parquet(f'{hadoop_fs}{hdfs_table}/year=2025/month=03/day=02/2025-03-02_istdaten.parquet')

print(duck_handle.query('df_table', 'DESCRIBE TABLE df_table').to_df())

print(duck_handle.query('df_table', 'SELECT * FROM df_table LIMIT 5').to_df())

print(duck_handle.query('df_table', 'SELECT COUNT(*) FROM df_table').to_df())
print(duck_handle.aggregate("COUNT(*)")) # same as above

# This is just one way, see the DuckDB documentation for more (e.g. `duckdb.sql('SELECT * FROM read_parquet(...) LIMIT 1')`)

# ---
# #### Querying HDFS Partitions with DuckDB
#
# We've explored how to partition data on HDFS, for example, using the Hive partitioning scheme.
#
# Tools like DuckDB are capable of reading data from these partitions. We DuckDB this is achieved by leveraging PyArrow Datasets (see doc).
#
# Execute the command below, what do you observe? Isn't the schema different?

sbb_istdaten = ds.dataset(hdfs_table, format="parquet", partitioning="hive", filesystem=hdfs_fs)
print(sbb_istdaten.schema.to_string(show_field_metadata=False))

with duckdb.connect(database=":memory:", read_only=False) as conn:
    conn.register('sbb_istdaten', sbb_istdaten) # This is a convenience, DuckDB will recognize a variable name
    print(conn.execute('DESCRIBE TABLE sbb_istdaten').df())
    print(conn.execute('SELECT year,month,day,COUNT(*) FROM sbb_istdaten GROUP BY year,month,day').df())

duckdb.sql('SELECT COUNT(*) FROM sbb_istdaten')

# ---
# #### Push-down queries 
#
# It is possible using the PyArrow.dataset.Dataset.filter method to filter the data at the source and minimize data transfer.
#
# For instance using a Push-Down predicate to filter the year, month and day at the source:

sbb_istdaten.scanner(filter=((pc.field('year')==2025)&(pc.field('month')==3)&(pc.field('day')==3))).to_table()



# Frameworks like DuckDB can take advantage of this to 

with duckdb.connect(database=":memory:", read_only=False) as conn:
    conn.register('sbb_istdaten', sbb_istdaten)
    print(conn.execute('SELECT COUNT(*) FROM sbb_istdaten WHERE year=2025 AND month=3 AND day=3').df())

# ----
# #### Exercises
#
# We copied the same SBB istdaten data under `/data/com-490/labs/week3`, but in CSV format (delimited by `;`)
#
# Use the knowledge you've gained above, and refer to the PyArrow and DuckDB documentation for querying this data as you did before. Do not hesitate to try different SQL queries.
#
# Note that when parsing SQL, methods like PyArrow or DuckDB try to guess the CSV format, and they sometime get it wrong. You may have to specify a format yourself, such as shown below:
#
#
# ```
# corrected_schema = pa.schema(
#     [
#         ('BETRIEBSTAG', pa.string()),
#         ('FAHRT_BEZEICHNER', pa.string()),
#         ('BETREIBER_ID', pa.string()),
#         ('BETREIBER_ABK', pa.string()),
#         ('BETREIBER_NAME', pa.string()),
#         ('PRODUKT_ID', pa.string()),
#         ('LINIEN_ID', pa.string()),
#         ('LINIEN_TEXT', pa.string()),
#         ('UMLAUF_ID', pa.string()),
#         ('VERKEHRSMITTEL_TEXT', pa.string()),
#         ('ZUSATZFAHRT_TF', pa.bool_()),
#         ('FAELLT_AUS_TF', pa.bool_()),
#         ('BPUIC', pa.int64()),
#         ('HALTESTELLEN_NAME', pa.string()),
#         ('ANKUNFTSZEIT', pa.string()),
#         ('AN_PROGNOSE', pa.string()),
#         ('AN_PROGNOSE_STATUS', pa.string()),
#         ('ABFAHRTSZEIT', pa.string()),
#         ('AB_PROGNOSE', pa.string()),
#         ('AB_PROGNOSE_STATUS', pa.string()),
#         ('DURCHFAHRT_TF', pa.bool_()),
#         ('year', pa.int32()),
#         ('month', pa.int32()),
#         ('day', pa.int32()),
#     ]
# )
#
# istdaten_ds = ds.dataset(istdaten_path, schema=corrected_schema, format=CsvFileFormat(parse_options=ParseOptions(delimiter=';')), partitioning="hive", filesystem=hdfs_fs)
# ```
#
# Or in DuckDB as we did for a different table:
#
# ```
# duckdb.sql(
# f"""
# SELECT * FROM read_csv('{os.environ.get('HADOOP_FS')}/{hdfs_path}',
#     columns={{
#         'stop_id':        'VARCHAR',
#         'stop_name':      'VARCHAR',
#         'stop_lat':       'DOUBLE',
#         'stop_lon':       'DOUBLE',
#         'location_type':  'BIGINT',
#         'parent_station': 'VARCHAR'
#     }}) LIMIT 10
# """
# )
# ```




