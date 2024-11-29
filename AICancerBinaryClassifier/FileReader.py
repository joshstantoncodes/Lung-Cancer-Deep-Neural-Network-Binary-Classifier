""""
    Lung Cancer Binary Classifier using miRNA Expression
    This program reads and assigns a simple "1" or "0" binary to miRNA samples
    taken from patients with lung cancer, using datasets acquired from the
    open access Cancer Genome Atlas, specifying whether the sample came from a
    patient with cancer or a non-cancer sample.

    Author: Josh Stanton
    Date: November 29th, 2024
"""


import logging
from uuid import UUID
from os import walk, path
from tqdm import tqdm
import pandas as pd


logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger("experiment_notebook")
ARTIFACT_NAME = "complete_data.parquet"


def load_data(
    root: str,
    dir_positives: str,
    dir_negatives: str,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load data from the provided directories and stores the compiled dataframe in file system.
    If such a cache is present, the function will load from it instead.
    """
    print("Root path:", path.join("./",root))
    print("Positive samples path:", path.join(root, dir_positives))
    print("Negative samples path:", path.join(root, dir_negatives))
    try:
        pathJoin = path.join(root, ARTIFACT_NAME)
        print(pathJoin)
        data = pd.read_parquet(pathJoin)
        LOGGER.info("Loaded dataset from cache.")

        # If file was found, do NOT redo combination.
        return data
    except FileNotFoundError:
        LOGGER.info(
            "Could not find dataset in cache, will attempt to build from scratch."
        )

    positives = read_raw_data_from_directory(path.join(root, dir_positives))
    positives["cancer"] = 1
    negatives = read_raw_data_from_directory(path.join(root, dir_negatives))
    negatives["cancer"] = 0
    data = pd.concat([positives, negatives])
    LOGGER.info("Finished building dataset from scratch.")
    if cache:
        dataset_filepath = path.join(root, ARTIFACT_NAME)
        if not path.exists(dataset_filepath):
            LOGGER.info("Persisting dataset in '%s'.", dataset_filepath)
            # Save to Parquet for further analysis.
            data = data.loc[~(data==0).all(axis="columns")]
            data.to_parquet(dataset_filepath, index=None)
    return data


def read_raw_data_from_directory(dirname: str) -> pd.DataFrame:
    """
    Convenience function for traversing all subdirectories from the provided starting directory,
    loading all relevant files into memory, and into a usable format.

    Parameters
    ----------
    dirname : str
        The name of the directory (absolute or relative) containing data.

    Returns
    -------
    A single DataFrame containing all data from the directory.
    """
    LOGGER.info("Walking data directory %s and reading in files...", dirname)
    rows = []

    for root, _, files in tqdm(
        walk("Databases/miRNA Files - Lung Cancer", topdown=False)
    ):
        for filename in files:
            # Workaround for identifying UUID.
            try:
                _ = UUID(filename.split(".")[0])
            except ValueError:
                LOGGER.debug(
                    "Skipped file because it did not start with a UUID-like string: %s",
                    filename,
                )
                continue
            rows.append(read_txt_data_file(root, filename))
    data = pd.DataFrame(rows).reset_index()
    del data["index"]
    return data


def read_txt_data_file(directory: str, filename: str) -> pd.Series:
    """
    Read a miRNA file into memory.

    Parameters
    ----------
    directory : str
        The directory that contains the file. Could be nested, so this string could be multiple, and
        it always contains the root directory.
    filename : str
        The name of the file to be loaded, including file ending.

    Returns
    -------
    A series of floats where each index corresponds to a MiRNA ID.
    """
    filepath = path.join(directory, filename)
    LOGGER.debug("Loading data from file %s", filepath)
    data = pd.read_csv(filepath, index_col="miRNA_ID", sep="\t")
    return data["reads_per_million_miRNA_mapped"]


data = load_data(
    root="Databases",
    dir_positives="miRNA Files - Lung Cancer",
    dir_negatives="miRNA Files - Normal",
)

print(data)

print(f"Dataset contains a total of {len(data)} samples.")
label_occurrences = data["cancer"].value_counts()

print(
    "Using 20% of data for validation, left with "
    f"{round(len(data)*0.8)} training samples."
)

num_negatives = label_occurrences[0]
print(
    f"{num_negatives} of the samples ({round(num_negatives/len(data)*100, 2)}%) are negative."
)

num_positives = label_occurrences[1]
print(
    f"{num_positives} of the samples ({round(num_positives/len(data)*100, 2)}%) are positive."
)

feature_cols = [col for col in data if col.startswith("hsa")]

print(
    f"Our input feature vector will likely be {len(feature_cols)}-dimensional."
)

dead_cols = []
for col in feature_cols:
    std = data[col].std()
    if std == 0:
        dead_cols.append(col)
print(f"{len(dead_cols)} columns are dead!")