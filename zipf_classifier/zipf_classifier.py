import re
import json
import os
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, save_npz, vstack, load_npz
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from seaborn import heatmap
from tqdm import tqdm
from typing import Iterator, Generator, List, Tuple


class ZipfClassifier:

    def __init__(self, n_jobs: int=1):
        """Create a new instance of ZipfClassifier
            n_jobs:int, number of parallel jobs to use.
        """
        self._classifier, self._classes, self._n_jobs, self._regex = None, None, n_jobs, re.compile(
            r"\W+")

    def _get_directories(self, path: str) -> List[str]:
        """Return the directories inside the first level of a given path.
            path:str, the path from where to load the first level directories list.
        """
        return next(os.walk(path))[1]

    def _lazy_file_loader(self, directory: str, files: List[str])->Generator:
        """Yield lazily loaded files.
            directory:str, the directory of given files list
            files: List[str], list of files
        """
        for file in files:
            with open("{directory}/{file}".format(directory=directory, file=file), "r") as f:
                yield f.read()

    def _lazy_directory_loader(self, directory: str)->Generator:
        """Yield lazily directory content
            directory:str, directory from where to load the documents
        """
        return (self._lazy_file_loader(path, files) for path, dirs, files in os.walk(directory) if not dirs)

    def _counter_from_path(self, files: list) -> Counter:
        """Return a counter representing the files in the given directory.
            files:list, paths for the files to load.
        """
        c = Counter()
        for file in files:
            c.update((w for w in re.split(self._regex, file.lower()) if w))
        return c

    def _counters_from_file_iterator(self, file_iterator: Iterator) -> list:
        """Return list of counters for the documents found in given root."""
        return (
            self._counter_from_path(files) for files in file_iterator
        )

    def _counters_to_frequencies(self, counters: list) -> csr_matrix:
        """Return a csr_matrix representing sorted counters as frequencies.
            counters:list, the list of Counters objects from which to create the csr_matrix
        """
        keys = self._keys
        frequencies = np.empty((len(counters), len(keys)))
        non_zero_rows_number = 0
        for counter in counters:
            if not counter:
                continue
            indices, values = np.array(
                [(keys[k], v) for k, v in counter.items() if k in keys]).T
            row_sum = np.sum(values)
            if row_sum:
                frequencies[non_zero_rows_number][indices] = values / row_sum
                non_zero_rows_number += 1
        return csr_matrix(frequencies[:non_zero_rows_number])

    def _build_dataset(self, root: str) -> csr_matrix:
        """Return a csr_matrix with the vector representation of given dataset.
            root:str, root of dataset to load
        """
        return self._counters_to_frequencies(
            self._counters_from_file_iterator(root))

    def _build_keymap(self, counters: list) -> dict:
        """Return an enumeration of the given counter keys as dictionary.
            counters:list, the list of Counters objects from which to create the keymap
        """
        print("Determining keyset from {n} counters.".format(n=len(counters)))
        keyset = set()
        for counter in counters:
            keyset |= set(counter)
        self._keys = {k: i for i, k in enumerate(keyset)}

    def _build_training_dataset(self, root: str) -> dict:
        """Return a dictionary representing the training dataset at the given root."""
        dataset_counters = {
            document_class:
            self._counters_from_file_iterator(self._lazy_directory_loader("{root}/{document_class}".format(
                root=root, document_class=document_class)))
            for document_class in self._get_directories(root)
        }

        self._build_keymap([
            counter for counters in dataset_counters.values()
            for counter in counters
        ])

        sparse_matrices = {
            key: self._counters_to_frequencies(counters)
            for key, counters in dataset_counters.items()
        }

        return {
            key: matrix
            for key, matrix in sparse_matrices.items()
        }

    def _kmeans(self, k: int, points: csr_matrix,
                iterations: int) -> tuple:
        """Return a tuple containing centroids and predictions for given data with k centroids.
            k:int, number of clusters to use for k-means
            points:csr_matrix, points to run kmeans on
            iterations:int, number of iterations of kmeans
        """
        print("Running kmeans on {n} points with k={k} and {m} iterations.".format(
            n=points.shape[0], k=k, m=iterations))
        kmeans = KMeans(
            n_clusters=k, max_iter=iterations, n_jobs=self._n_jobs)
        kmeans.fit(points)
        return kmeans.cluster_centers_, kmeans.predict(points)

    def _representative_points(self,
                               points: csr_matrix,
                               k: int,
                               iterations: int,
                               points_percentage: float,
                               distance_percentage: float) -> csr_matrix:
        """Return representative points for given set, using given percentage `points_percentage` and moving points of `distance_percentage`.
            points:csr_matrix, points from which to extract the representative points
            k:int, number of clusters
            iterations:int, number of iterations of kmeans
            points_percentage:float, percentage of points to use as representatives
            distance_percentage:float, percentage of distance to move representatives towards respective centroid
        """
        centroids, predictions = self._kmeans(k, points, iterations)

        print("Determining representative points.")

        representatives = centroids

        distances = np.squeeze(
            np.asarray(
                np.power(points - centroids[predictions], 2).sum(axis=1)))
        for i in tqdm(range(k), leave=False, total=k):
            cluster = points[predictions == i]
            Ni = cluster.shape[0]
            ni = np.floor(points_percentage * Ni).astype(int)
            representatives = np.vstack([
                representatives, cluster[np.argpartition(
                    distances[predictions == i].reshape(
                        (Ni, )), ni)[-ni:]] * (1 - distance_percentage) + centroids[i] * distance_percentage
            ])
        return csr_matrix(representatives)

    def _build_classifier(self, dataset: dict,
                          k: int,
                          iterations: int,
                          points_percentage: float,
                          distance_percentage: float) -> tuple:
        """Build classifier for given dataset.
            dataset:str, root of given dataset
            k:int, number of clusters
            iterations:int, number of iterations of kmeans
            points_percentage:float, percentage of points to use as representatives
            distance_percentage:float, percentage of distance to move representatives towards respective centroid
        """
        print("Determining representative points for {n} classes.".format(
            n=len(dataset.keys())))
        return np.array(list(dataset.keys())), [
            self._representative_points(
                data, k, iterations, points_percentage, distance_percentage)
            for data in dataset.values()
        ]

    def fit(self, path: str, k: int,
            iterations: int,
            points_percentage: float,
            distance_percentage: float) -> tuple:
        """Load the dataset at the given path and fit classifier with it.
            path:str, the path from where to load the dataset
            k:int, number of clusters
            iterations:int, number of iterations of kmeans
            points_percentage:float, percentage of points to use as representatives
            distance_percentage:float, percentage of distance to move representatives towards respective centroid
        """
        self._classes, self._representatives = self._build_classifier(
            self._build_training_dataset(path), k, iterations, points_percentage, distance_percentage)

    def load(self, directory: str):
        """Load the trained classifier from given directory.
            path:str, the path from where to load the trained classifier.
        """
        self._classes, self._representatives = zip(*[(doc.split(".")[0], load_npz("{path}/{doc}".format(path=path, doc=doc))) for path, dirs,
                                                     docs in os.walk(directory) for doc in docs if doc.endswith(".npz")])

        self._classes = np.array(self._classes)

        with open(
                "{directory}/keys.json".format(directory=directory), "r") as f:
            self._keys = json.load(f)

    def save(self, path: str):
        """Save the trained classifier to given directory.
            path:str, the path to save the trained classifier.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        [
            save_npz(
                "{path}/{matrix_class}.npz".format(
                    path=path, matrix_class=matrix_class), matrix)
            for matrix, matrix_class in zip(self._representatives, self._classes)
        ]

        with open(
                "{path}/keys.json".format(path=path), "w") as f:
            json.dump(self._keys, f)

    def _setup_axis(self, subplot_width: int, suplot_position: int, title: str, x_margins: Tuple[float, float], y_margins: Tuple[float, float])->mpl.axes.SubplotBase:
        """Return characterized subplot axis in given position.
            subplot_width:int, length of subplot rows
            suplot_position:int, position of axis in given subplot
            title:str, title of subplot
            x_margins:Tuple[float, float], margins of horizontal axis
            y_margins:Tuple[float, float], margins of vertical axis
        """
        ax = plt.subplot(2, subplot_width, suplot_position)
        ax.grid()
        ax.set_title(title)
        ax.set_xlim(*x_margins)
        ax.set_ylim(*y_margins)
        ax.legend(loc='upper right')
        return ax

    def _svd(self, dataset: csr_matrix, predictions: np.ndarray, originals: np.ndarray, labels: list, directory: str, title: str):
        """Plot SVD with 2 components of predicted dataset.
            dataset: csr_matrix, classified dataset 
            predictions: np.ndarray, predicted labels of dataset
            originals: np.ndarray, original labels of dataset
            labels: list, unique labels of original dataset
            directory: str, directory where to save the given 
            title: str,
        """
        reduced = TruncatedSVD(n_components=2).fit_transform(
            StandardScaler(with_mean=False).fit_transform(dataset))
        columns = ("original", "prediction")
        maximum_x, maximum_y = np.max(reduced, axis=0) + 0.1
        minimum_x, minimum_y = np.min(reduced, axis=0) - 0.1
        margins = (minimum_x, maximum_x), (minimum_y, maximum_y)

        df = pd.concat(
            [
                pd.DataFrame(data=reduced, columns=['a', 'b']),
                pd.DataFrame({
                    columns[0]: predictions,
                    columns[1]: originals
                })
            ],
            axis=1)
        plt.figure(figsize=(20, 8))
        colors = ["red", "green", "blue", "orange", "purple", "black"]
        n = len(labels) + 1

        cumulative_original_ax = self._setup_axis(n, n, "Originals", *margins)
        cumulative_prediction_ax = self._setup_axis(
            n, 2*n, "Predictions", *margins)

        for i, (label, color) in enumerate(zip(labels, colors), 1):
            original_ax = self._setup_axis(
                n, i, "Original {label}".format(label=label), *margins)
            prediction_ax = self._setup_axis(
                n, n+1, "Prediction {label}".format(label=label), *margins)
            for ax, column in zip(((original_ax, cumulative_original_ax), (prediction_ax, cumulative_prediction_ax)), columns):
                indices = df[column] == label
                ax.scatter(
                    df.loc[indices, 'a'],
                    df.loc[indices, 'b'],
                    c=color,
                    label=label,
                    s=20)

        plt.savefig(
            "{directory}/{title} - Truncated SVD.png".format(directory=directory, title=title))
        plt.clf()

    def _heatmap(self, axis: mpl.axes.SubplotBase, data: np.matrix, labels: list, title: str, fmt: str):
        """ Plot given matrix as heatmap.
            data:np.matrix, the matrix to be plotted.
            labels:list, list of labels of matrix data.
            title:str, title of given image.
            fmt:str, string formatting of digids
        """
        heatmap(
            data,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=fmt,
            cmap="YlGnBu",
            cbar=False)
        axis.yticks(rotation=0)
        axis.xticks(rotation=0)
        axis.set_title(title)

    def _plot_confusion_matrices(self, confusion_matrix: np.matrix, labels: list, path: str, title: str):
        """ Plot default and normalized confusion matrix.
            confusion_matrix:np.matrix, the confusion matrix to be plot.
            labels:list, list of labels of matrix data.
            path:str, the path were to save the matrix.
            title:str, the title for the documents.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        plt.figure(figsize=(8, 4))
        self._heatmap(plt.subplot(1, 1, 1), confusion_matrix,
                      labels, "Confusion matrix", "d")
        self._heatmap(plt.subplot(1, 1, 2), confusion_matrix.astype(np.float) /
                      confusion_matrix.sum(axis=1)[:, np.newaxis], labels, "Normalized confusion matrix", "0.4g")
        plt.suptitle(title)
        plt.savefig("{path}/{title} - Confusion matrices.png".format(path=path,
                                                                     title=title))
        plt.clf()

    def _save_results(self, directory: str, name: str, dataset: csr_matrix, originals: np.ndarray, predictions: np.ndarray, labels: List[str]):
        """Save classification results.
            directory:str, path to directory where to save results
            name:str, project name
            dataset:csr_matrix, classified compiled dataset
            originals:np.ndarray, original labels
            predictions:np.ndarray, predicted labels
            labels:List[str], list of unique labels
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_npz(
            "{directory}/{name}-dataset.npz".format(directory=directory, name=name), dataset)
        np.save("{directory}/{name}-originals.npz".format(directory=directory,
                                                          name=name), originals)
        np.save("{directory}/{name}-predictions.npz".format(directory=directory,
                                                            name=name), predictions)
        self._svd(dataset, originals, predictions, labels,
                  directory, name)
        self._plot_confusion_matrices(confusion_matrix(
            originals, predictions, labels=labels), labels, directory, name)

    def _classify(self, dataset: csr_matrix) -> Tuple[csr_matrix, np.ndarray]:
        """Return a tuple with classified dataset and classification vector.
            dataset:csr_matrix, dataset to classify
        """
        return dataset, self._classes[np.argmin(
            [
                np.min(euclidean_distances(dataset, c), axis=1)
                for c in self._representatives
            ],
            axis=0)]

    def classify_directory(self, directory: str) -> Tuple[csr_matrix, np.ndarray]:
        """Load the dataset at the given path and run trained classifier with it.
            directory:str, the path from where to load the dataset
        """
        return self._classify(self._build_dataset(self._lazy_directory_loader(directory)))

    def classify_texts(self, texts: List[str]) -> Tuple[csr_matrix, np.ndarray]:
        """Return the classification of given texts
            texts:List[str], the texts to classify
        """
        return self._classify(self._build_dataset([texts]))

    def classify_text(self, text: str) -> Tuple[csr_matrix, np.ndarray]:
        """Return the classification of given text
            text:str, the text to classify
        """
        return self.classify_texts([text])

    def set_seed(self, seed: int):
        """Set random seed.
            seed:int, the random seed to use for the test.
        """
        np.random.seed(seed)
        random.seed(seed)

    def test(self, path: str):
        """Run test on the classifier over given directory, considering top level as classes.
            path:str, the path from where to run the test.
        """
        directories = self._get_directories(path)
        print("Running {n} tests with the data in {path}.".format(
            n=len(directories), path=path))
        labels, datasets, predictions = zip(*[(directory, *self.classify_directory("{path}/{directory}".format(
            path=path, directory=directory)))
            for directory in directories])
        self._save_results("results", path.replace("/", "_"), vstack(datasets), np.repeat(
            labels, [len(p) for p in predictions]), np.concatenate(predictions), labels)
