# -*- coding: utf-8 -*-
# # Pandas and scikit-learn

# ## 1. Getting familiar with pandas
#
# Pandas is a package providing vectorized operations. It makes data wrangling easier by providing a dataframe structure with column names and indices.
#

# !./get_data.sh

import pandas as pd
import numpy as np
import random

# ### 1.1 Create two data frames
#
# Generate time series by using the pandas [date_range](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html) function to create timestamps at different frequencies, and use numpy's [randn](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html) to generate random values from a normal distribution.

dates = pd.date_range('20240101', '20251231', freq='24h')
data_1 = pd.DataFrame(np.random.randn(731,2), index=dates, columns=['x1','x2'])

dates = pd.date_range('20240101', '20251231', freq='12h')
data_2 = pd.DataFrame(np.random.randn(1461,1), index=dates, columns=['x3'])

# ### 1.2 Merge the data frames
#
# Join the two DataFrames on their timestamp indices using the pandas [merge](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html#pandas.DataFrame.merge) function.

data_3 = data_1.merge(data_2, how='inner', left_index=True, right_index=True)

# ### 1.3 Inspect the Pandas DataFrame
#
# Use the pandas functions head(), describe(), and info(). Then, repeat the process with the _how_ parameter set to _inner_, _left_, _right_, and _outer_.

data_3.head()

data_3.describe()

data_3.info()

# ### 1.4 Pandas DataFrame Aggregations and Resampling
#
# Using pandas [groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html#pandas.DataFrame.groupby), you can group rows that satisfy a condition and compute aggregations within each group, such as the sum of values in a column for each group.
#
# Additionally, if the index is a time series, as in this example, pandas provides a built-in function for resampling time series dataframes ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html)).
#
# Use resample to calculate the median at 2-day intervals."

data_4 = data_3.resample('2D').median()
data_4.head()


# ### 1.5 DataFrames Column operations
#
# Using the pandas DataFrame.apply method you can create a new column $y=x_1^3/100 - x_1x_2/2 + x_2^2/400$.
#
# The pandas [concat](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html#pandas.concat) function can be used to concatenate two DataFrames either vertically (using axis=0) if they have the same columns, or horizontally (using axis=1) if they have the same number of rows.

def xor(x1,x2):
    if (x1**3/100 - x1*x2/2 + x2**2/400)>0:
        return 1
    else:
        return 0


y = pd.DataFrame(data_4.apply(lambda row: xor(row['x1'],row['x2']), axis=1), columns=['y'])
data = pd.concat([data_4[['x1','x2','x3']], y], axis=1)

# ### 1.6 Data Visualization
#
# Pandas offers the [scatter_matrix](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html) function to visualize interactions between DataFrame columns. Using other visualization libraries, such as Seaborn, you can achieve similar insights with more visually appealing plots.

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(data, hue='y')
plt.show()

# ### 1.7  Writing Pandas DataFrames to files
#
# Using the pandas [to_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html#pandas.DataFrame.to_csv) function, save the dataframe to file in CSV (comma-separated values) format.

data.to_csv('./test_pythoncourse.csv')

# ### 1.4  Reading Pandas DataFrames from files
#
# Using the pandas [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv) function, read the DataFrame from the CSV file, omitting the header and column names, or try setting header=0 and names=None.

data = pd.read_csv('test_pythoncourse.csv',index_col=0)
data

data.head()

# ## 2. Exercise: Supervised learning: DNA sequence classification

# The DNA double helix is made up of 2 anti-parallel polymer strands. These strands are complementary to each other: an `A` can only be paired with a `T` on the opposite strand, and a `C` with a `G`. For example, a segment of DNA could look like this:
#
# -- `AAGCT`-->
#
# <-`TTCGA`--
#
# The strands are antiparallel, meaning that any region will be read in the opposite direction on the other strand. Therefore, a biological sequence on one strand can be reverse-complemented to obtain the sequence read on the other strand.
#
# When working on natural languages, sentences are usually decomposed into words or "tokens". Although DNA sequences do not contain words, it is common to decompose them into k-mers (akin to n-grams in natural languages). A k-mer is a continuous sequence of k nucleotides.
#
# For example, decomposing `AACAT` into k-mers with k=3 (tri-mers) yields `[AAC, ACA, CAT]`.
#
# More info is provided in [background.md](./background.md).
#
# Change the code to replace all the ```###TODO###``` with the appropriate instructions.

# ### 2.1: Read the data
#
# We will use a dataset consisting of 106 DNA sequences. All sequences are the same length (57 nucleotides), but half of them are bacterial promoters.
#
# The input table contains 3 columns:
#
# * Whether each sequence is a promoter (+) or not (-).
# * A unique identifier
# * A DNA sequence of 57 nucleotides.
#
# The dataset is stored as a CSV file.
#
# > Note: A promoter is a regulatory DNA sequence located just before the start of a gene and modulating its activity.

data_path = './data/promoter_sequences.csv'

seq_df = pd.read_csv(data_path, names=["promoter", "name", "sequence"])
seq_df

# ### 2.2: Clean the data
#
# The DNA sequences contain unwanted (tabs `\t`) characters, use [pandas string methods](https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling) on the column to address the problem.

seq_df['sequence'] = seq_df['sequence'].replace("\t", "")

# ### 2.3: Extract features
#
# Use the `get_all_kmer_freqs` function provided below to extract k-mer frequencies from each sequence. Use k-mer frequencies as features  and add them as columns to the dataframe (1 column / k-mer freq. in the dataframe).
#
# The code is provided, can you understand what it does?
#
# The new dataframe for k=4 should look like:
#
# ```
# name sequence                    aaaa     aaac    aaag    aaat    aaca    aacc    aacg   ...  
# S10  tactagcaatacgcttgcgttcg...  0.0000   0.0000  0.0000  0.0000  0.0000  0.0185  0.0185 ...
# ```

# +
from typing import Iterator, List
import itertools
from collections import defaultdict

def yield_kmers(seq: str, k: int = 4) -> Iterator[str]:
    for s in range(len(seq) - k + 1):
        yield seq[s : s + k].lower()


def rev_comp(seq: str) -> str:
    comp_map = seq.maketrans("acgt", "tgca")
    comp_seq = seq.translate(comp_map)
    rev_comp = comp_seq[::-1]
    return rev_comp


def get_kmers_vocab(k: int):
    prods = itertools.product("acgt", repeat=k)
    return [k for k in map("".join, prods) if k < rev_comp(k)]

def get_all_kmer_freqs(seq: str, k: int = 4) -> List[float]:
    vocab = get_kmers_vocab(k)
    freqs = defaultdict(float)
    n_kmers = len(seq) - k + 1
    # Extract k-mers from input sequence
    for kmer in yield_kmers(seq, k=k):
        cano = min(kmer, rev_comp(kmer))
        freqs[cano] += 1 / n_kmers
    return [freqs[k] for k in vocab]


# -

get_kmers_vocab(k=4)[:15]

# Use the functions seen in the previous section to complete this code.

# Solution
k = 4
freqs = np.vstack(seq_df.sequence.apply(lambda x: get_all_kmer_freqs(x, k=k)))
features = pd.concat(
    [
        seq_df[['name', 'sequence']],
        pd.DataFrame(freqs, columns=get_kmers_vocab(4))
    ],
    axis=1,
)
features

# ### 2.4: Sequence classification with scikit-learn
#
# The scikit-learn (`sklearn`) library, provides many utilities and standard models for machine learning. We will attempt to classify sequences into promoter or non-promoter using a random-forest model, as it does not suffer from the high dimensionality of our dataset and is easily interpretable.

# Train a regression model to predict promoter / non-promoter status
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

X = np.array(features.drop(columns=['name', 'sequence']))
y = seq_df.promoter
clf = RandomForestClassifier()

# Split the data into 2 sets (e.g. with `train_test_split` to train the model on the first set and check predictions on the second.

# +
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y)
clf.fit(X_train, y_train)
accuracy_score(y_test, clf.predict(X_test))
# -

# Run the cell several times, what do you observe in terms of prediction accuracy ?

# To properly assess the model's performance, we should use a cross-validation strategy. Here, the dataset is relatively small, so we can use [leave-one-out](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html) crossvalidation.
# This method involves repeatedly excluding one sample from the training set and attempting to predict it, until all samples have been tested. It is slow; if there are N samples, it involves training the model N times on N-1 samples.
#
# Implement Leave-one-out cross validation using the `sklearn` API.

# +
# %%time
# Solution
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
cv = LeaveOneOut().split(X)
preds = np.zeros(y.shape, dtype=str)
for train_idx, test_idx in cv:
    clf.fit(X[train_idx, :], y[train_idx])
    pred = clf.predict(X[test_idx, :])
    preds[test_idx] = pred
    
ConfusionMatrixDisplay.from_predictions(y, preds)
plt.title(f"LOO cross validation accuracy: {accuracy_score(y, preds):.2f}")
# -


# ### 2.5: Interpreting the model
#
# As mentioned previously, random forests (and tree-based models in general) are readily interpretable by measuring the importance of each feature in the decision trees.
#
# Extract feature importances from the model and visualize them (use matplotlib barh). What are sequence motifs are important for prediction ? What do you think of it ? (Hint: [useful info here](https://en.wikipedia.org/wiki/Promoter_(genetics)#Elements))

plt.figure(figsize=(10, 10))
# Refit on whole dataset
clf.fit(X_train, y_train)
imp_idx = np.argsort(clf.feature_importances_)
plt.barh(features.columns[2:][imp_idx][-10:], clf.feature_importances_[imp_idx][-10:])

# ### 2.6: Hyperparameter tuning with gridsearch
#
# The default model hyperparameters may be suboptimal for our dataset. In order to find the optimal combination of parameters, we can probe the model performance across parameter space. This operation is called grisearch and `sklearn` also implements a set of utility functions to facilitate the process, such as [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

import yaml
from sklearn.model_selection import GridSearchCV
# Search for optimal parameters (uses 5-fold cv for validation by default)
params = {
    "max_features": [3, 5, 10, 20, 50],
    "n_estimators": [2 ** p for p in range(6, 8)],
}
# Note n_jobs is the number of parallel jobs, this should depend on your machine's CPUs
gs = GridSearchCV(clf, param_grid=params, verbose=True, n_jobs=1)
gs.fit(X, y)
print(
    f"Random forest hyper-parameter search:\n"
    f"Obtained a best accuracy of {gs.best_score_:.2f} "
    f"with the following parameters: \n{yaml.dump(gs.best_params_)}",
)
clf.set_params(**gs.best_params_)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title(f"LOO cross validation accuracy: {accuracy_score(y_test, pred):.2f}")

# Is it a good model? did we overfit, or underfit?

# ## Exercise: Unsupervised learning with K-means

# ### 3.1 Principal Component Analysis
#
# Apply PCA to the K-mer frequencies. Store the first two principal components and their true cluster index (ground truth) into a new dataframe. Visualize the sequences using these principal components (as x and y axes), colored by promoter status.

# %%time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
n_components=3
pca = PCA(n_components=n_components)
pcs = pca.fit_transform(X)
sns.pairplot(pd.DataFrame(pcs[:, :n_components]).assign(prom=y), hue='prom')
plt.show()

# Use K-means clustering to identify 2 clusters what proportion of points are correctly assigned (note that KMeans may inverse the labels/colors)?
# Compare visually, is it a good approach?

# %%time
from sklearn.cluster import KMeans
km = KMeans(2)
km.fit(X)
pred = km.predict(X)
sns.pairplot(pd.DataFrame(pcs[:, :n_components]).assign(prom=pred), hue='prom')
plt.show()

