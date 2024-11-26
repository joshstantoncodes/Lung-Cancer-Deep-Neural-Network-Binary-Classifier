import logging
from uuid import UUID
from os import walk, path
from tqdm import tqdm
import pandas as pd

# Note that I have configured the logger not to print debug statements while actually using some.
# If you want to debug further, set to `logging.DEBUG`.
logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger("experiment_notebook")
ARTIFACT_NAME = "complete_data.csv"


def load_data(
    root: str,
    dir_positives: str,
    dir_negatives: str,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load data from the provided directories and stores the compiled dataframe in your file system.
    If such a cache is present, the function will load from it instead.
    """
    try:
        data = pd.read_csv(path.join(root, ARTIFACT_NAME))
        LOGGER.info("Loaded dataset from cache.")
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
            data.to_csv(dataset_filepath, index=None)
    return data


def read_raw_data_from_directory(dirname: str) -> pd.DataFrame:
    """
    Convenience function for traversing all subdirectories from the provided starting directory,
    loading all relevant files into memory, and into a usable format.

    @Josh: You could conceivably use the filepath as the index (or just add it as a column) so you
    can later map individual rows in your dataset onto the source datasets. I just didn't bother
    because I didn't really see a value here.

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
        walk("data/miRNA Files - Lung Cancer", topdown=False)
    ):
        for filename in files:
            # Ugly workaround for identifying UUID because I couldn't be FUCKED to write a regex rn.
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
    # Drop the index column that no longer serves as the index because pandas be funky.
    del data["index"]
    return data


def read_txt_data_file(directory: str, filename: str) -> pd.Series:
    """
    Read a miRNA file into memory.

    Parameters
    ----------
    directory : str
        The directory that contains the file. Could be nested, so this string could be multiple and
        it always contains the root directory.
    filename : str
        The name of the file to be loaded, including file ending.

    Returns
    -------
    A series of floats where each index corresponds to a MiRNA ID.
    """
    filepath = path.join(directory, filename)
    LOGGER.debug("Loading data from file %s", filepath)
    # We assume consistently tabular style in all txt files. This is technically a bad assumption to
    # make but I'm not writing code for a product.
    data = pd.read_csv(filepath, sep="\t", index_col="miRNA_ID")
    # We can drop all irrelevant data to return a simple feature vector.
    return data["reads_per_million_miRNA_mapped"]