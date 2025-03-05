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

# ---
# # HDFS Programmatically
#
# You can also access HDFS programmatically with Python (and other languages).
#
# We show here two methods: using the high-level API pandas, and the lower-level API with pyarrow, on which pandas and many other SDKs (AWS, Azure, etc.) are based for not only HDFS but other file systems, such as S3.
#
# In the past, we were using hdfs3 and other APIs, which closely match the HDFS command line (CLI) functions and offer an intermediate between straightforward file read/write access of pandas, and file management functions (create directory, move or remove files) of pyarrow but without the complexity. However, those APIs are no longer being maintained.

# ---
# #### 1. HDFS with pandas
#
# Pandas can be used to read or write files on many storage systems, including HDFS. However, it is not meant to be used to move files around or delete files.
#
# Import pandas, and make sure that the environment variable HADOOP_FS is set. This variable points to the URL of the HDFS Namenode server.

# +
import pandas as pd
import os

hadoop_fs = os.environ.get('HADOOP_FS')
username = os.environ.get('USER')
print(f'HDFS namenode URL: {hadoop_fs}')
print(f'you are {username}')
# -

# When reading a DataFrame with Pandas, you can assign it to a variable like df and inspect its contents. However, keep in mind that Pandas operates on the DataFrame in-memory, so if the dataset is too large, you may run into memory issues or run out of RAM.
#
# You can check that with `print(f'Memory usage: {df.memory_usage(deep=True).sum()} bytes')`.
#
# Note also that you must specify the URL of the HDFS nodename (hadoop_df) when reading from HDFS.

# +
sbb_istdaten=f'{hadoop_fs}/user/{username}/week3/sbb/istdaten/year=2025/month=03/day=02/2025-03-02_istdaten.parquet'
print(f'Read from {sbb_istdaten}')
pd.read_parquet(sbb_istdaten)

#df=pd.read_parquet(sbb_istdaten)
#print(f'Memory usage: {df.memory_usage(deep=True).sum()} bytes')
# -

# ---
# ### 2. HDFS with pyarrow, low level API
#
# The PyArrow API provides a generic abstraction designed to simplify data exchange and interaction between different applications, enabling seamless integration with various data storage formats and systems.
#
# However, due to its cross-platform design, its capabilities are somewhat limited, as they represent the intersection of features supported across the different platforms and systems it aims to support.

# For instance, to upload a file you could have used:

# +
from pyarrow.fs import HadoopFileSystem
from pyarrow.fs import LocalFileSystem
from pyarrow.fs import FileType

hdfs = HadoopFileSystem.from_uri(hadoop_fs)
localfs = LocalFileSystem()

from_local='2025-03-02_istdaten.parquet'
to_hdfs=f'/user/{username}/week3/sbb/istdaten/year=2025/month=03/day=02/2025-03-02_istdaten.parquet'

print(hdfs.get_file_info(to_hdfs).type)

if not hdfs.get_file_info(to_hdfs).type==FileType.NotFound:
    print(f'Delete file {to_hdfs} on HDFS')
    hdfs.delete_file(to_hdfs)

if hdfs.get_file_info(to_hdfs).type==FileType.NotFound:
    print(f'Uploading local file {from_local} to {to_hdfs} on HDFS')
    buffer_size=int(1024*1024)
    with localfs.open_input_stream(from_local, compression='detect', buffer_size=buffer_size) as f_in:
        with hdfs.open_output_stream(to_hdfs, compression='detect', buffer_size=buffer_size) as f_out:
                while True:
                        buf = f_in.read(buffer_size)
                        if buf:
                                f_out.write(buf)
                        else:
                                break
# -

# Note that you can list the content of a folder, see [pyarrow.fs.FileInfo](https://arrow.apache.org/docs/python/generated/pyarrow.fs.FileInfo.html)

from pyarrow import fs
for file in hdfs.get_file_info(fs.FileSelector(f'/user/{username}/week3', recursive=True)):
    print(f'File -- path:{file.path}, type:{file.type}, mod_time:{file.mtime_ns}, is_file:{file.is_file}, size:{file.size}')

# PyArrow integrates well with [fsspec](https://filesystem-spec.readthedocs.io/en/latest/features.html#pyarrow-integration), which offers a more convenient API for certain file system operations.

# Using an ArrowFSWrapper to wrap the existing pyarrow client did we already created

# +
from fsspec.implementations.arrow import ArrowFSWrapper

fs_wrapper_hdfs = ArrowFSWrapper(hdfs)
fs_wrapper_hdfs.ls(f'/user/{username}/week3/')
# -

# Or directly using an fsspec client.

# +
import fsspec

fs = fsspec.filesystem('hdfs', url=hadoop_fs)
fs.ls(f'/user/{username}/week3/')
# -

# For a list of all the protocols supported by fsspec ...

fsspec.available_protocols()

# **Q1**: Use the knowledge you've gained above, and refer to the PyArrow documentation for creating folders, as you did in Exercise 1. 
#
# However, note that you will not be able to change the permissions."


