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


class ZipfClassifier:

    def __init__(self, n_jobs: int=1, seed: int=42):
        """Create a new instance of ZipfClassifier
            n_jobs:int, number of parallel jobs to use.
            seed:int, random seed to reproduce results.
        """
        self._classifier = None
        self._classes = None
        self._n_jobs = n_jobs
        self._regex = re.compile(r"\W+")
        self._seed = seed
        random.seed(self._seed)
        np.random.seed(self._seed)

    def _get_directories(self, path: str) -> list:
        """Return the directories inside the first level of a given path.
            path:str, the path from where to load the first level directories list.
        """
        return next(os.walk(path))[1]

    def _find_files(self, root: str) -> list:
        """Return leaf files from given `root`."""
        print("Searching files within directory {root}".format(root=root))
        return [[
            "{path}/{file}".format(path=path, file=file) for file in files
        ] for path, dirs, files in os.walk(root) if not dirs]

    def _counter_from_path(self, files: list) -> Counter:
        """Return a counter representing the files in the given directory.
            files:list, paths for the files to load.
        """
        c = Counter()
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                c.update((w for w in re.split(
                    self._regex, f.read().lower()) if w))
        return c

    def _counters_from_root(self, root: str) -> list:
        """Return list of counters for the documents found in given root."""
        return [
            self._counter_from_path(files) for files in self._find_files(root)
        ]

    def _counters_to_frequencies(self, counters: list) -> csr_matrix:
        """Return a csr_matrix representing sorted counters as frequencies.
            counters:list, the list of Counters objects from which to create the csr_matrix
        """
        print("Converting {n} counters to sparse matrix.".format(
            n=len(counters)))
        keys = self._keys
        frequencies = np.empty((len(counters), len(keys)))
        i = 0
        for counter in counters:
            if not counter:
                continue
            indices, values = np.array(
                [(keys[k], v) for k, v in counter.items() if k in keys]).T
            row_sum = np.sum(values)
            if row_sum:
                frequencies[i][indices] = values / row_sum
                i += 1
        return csr_matrix(frequencies[:i])

    def _build_dataset(self, root: str) -> csr_matrix:
        """Return a csr_matrix with the vector representation of given dataset.
            root:str, root of dataset to load
        """
        return self._counters_to_frequencies(
            self._counters_from_root(root))

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
            self._counters_from_root("{root}/{document_class}".format(
                root=root, document_class=document_class))
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
            n_clusters=k, random_state=self._seed, max_iter=iterations, n_jobs=self._n_jobs)
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

    def classify(self, path: str) -> list:
        """Load the dataset at the given path and run trained classifier with it.
            path:str, the path from where to load the dataset
        """
        dataset = self._build_dataset(path)
        return dataset, self._classes[np.argmin(
            [
                np.min(euclidean_distances(dataset, c), axis=1)
                for c in self._representatives
            ],
            axis=0)]

    def _svd(self, dataset: csr_matrix, predictions: np.ndarray, originals: np.ndarray, labels: list, path: str, title: str):
        if not os.path.exists(path):
            os.makedirs(path)
        random.seed(self._seed)
        np.random.seed(self._seed)
        svd = TruncatedSVD(n_components=2)
        reduced = svd.fit_transform(
            StandardScaler(with_mean=False).fit_transform(dataset))
        columns = ("original", "prediction")
        maximum_x, maximum_y = np.max(reduced, axis=0)
        minimum_x, minimum_y = np.min(reduced, axis=0)
        margin = 0.05
        margin_x, margin_y = maximum_x * margin, maximum_y * margin
        maximum_x, maximum_y, minimum_x, minimum_y = maximum_x + \
            margin_x, maximum_y + margin_y, minimum_x - margin_x, minimum_y - margin_y
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
        i = 1
        cumulative_original_ax = plt.subplot(2, n, n)
        cumulative_prediction_ax = plt.subplot(2, n, 2 * n)
        cumulative_original_ax.grid()
        cumulative_prediction_ax.grid()
        cumulative_original_ax.set_title("Originals")
        cumulative_prediction_ax.set_title("Predictions")
        cumulative_original_ax.set_xlim(minimum_x, maximum_x)
        cumulative_original_ax.set_ylim(minimum_y, maximum_y)
        cumulative_prediction_ax.set_xlim(minimum_x, maximum_x)
        cumulative_prediction_ax.set_ylim(minimum_y, maximum_y)

        for label, color in zip(labels, colors):
            original_ax = plt.subplot(2, n, i)
            prediction_ax = plt.subplot(2, n, n + i)
            original_ax.grid()
            prediction_ax.grid()
            original_ax.set_xlim(minimum_x, maximum_x)
            original_ax.set_ylim(minimum_y, maximum_y)
            prediction_ax.set_xlim(minimum_x, maximum_x)
            prediction_ax.set_ylim(minimum_y, maximum_y)

            i += 1
            original_ax.set_title("Original {label}".format(label=label))
            prediction_ax.set_title("Prediction {label}".format(label=label))
            for ax in (original_ax, cumulative_original_ax):
                indices = df[columns[0]] == label
                ax.scatter(
                    df.loc[indices, 'a'],
                    df.loc[indices, 'b'],
                    c=color,
                    label=label,
                    s=20)
            for ax in (prediction_ax, cumulative_prediction_ax):
                indices = df[columns[1]] == label
                ax.scatter(
                    df.loc[indices, 'a'],
                    df.loc[indices, 'b'],
                    c=color,
                    label=label,
                    s=20)

            prediction_ax.legend(loc='upper right')
            original_ax.legend(loc='upper right')

        cumulative_prediction_ax.legend(loc='upper right')
        cumulative_original_ax.legend(loc='upper right')

        plt.savefig(
            "{path}/{title}.png".format(path=path, title=title))
        plt.clf()

    def _heatmap(self, data: np.matrix, labels: list, title: str, fmt: str):
        """ Plot given matrix as heatmap.
            data:np.matrix, the matrix to be plotted.
            labels:list, list of labels of matrix data.
            title:str, title of given image.
            fmt:str, string formatting of digids
        """
        plt.figure(figsize=(8, 8))
        heatmap(
            data,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=fmt,
            cmap="YlGnBu",
            cbar=False)
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        plt.title(title)

    def _plot_confusion_matrices(self, confusion_matrix: np.matrix, labels: list, path: str, title: str):
        """ Plot default and normalized confusion matrix.
            confusion_matrix:np.matrix, the confusion matrix to be plot.
            labels:list, list of labels of matrix data.
            path:str, the path were to save the matrix.
            title:str, the title for the documents.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self._heatmap(confusion_matrix, labels, title, "d")
        plt.savefig("{path}/{title}.png".format(path=path, title=title))
        plt.clf()
        normalized_title = "Normalized {title}".format(title=title)
        self._heatmap(confusion_matrix.astype(np.float) /
                      confusion_matrix.sum(axis=1)[:, np.newaxis], labels, normalized_title, "0.4g")
        plt.savefig("{path}/{title}.png".format(path=path,
                                                title=normalized_title))
        plt.clf()

    def _save_results(self, dataset, originals, predictions, name):
        path = "results"
        if not os.path.exists(path):
            os.makedirs(path)
        save_npz("{path}/{name}-dataset.npz".format(path=path, name=name), dataset)
        np.save("{path}/{name}-originals.npz".format(path=path,
                                                     name=name), originals)
        np.save("{path}/{name}-predictions.npz".format(path=path,
                                                       name=name), predictions)

    def test(self, path: str) -> list:
        """Run test on the classifier over given directory, considering top level as classes.
            path:str, the path from where to run the test.
        """
        directories = self._get_directories(path)
        print("Running {n} tests with the data in {path}.".format(
            n=len(directories), path=path))
        labels, datasets, datasets_predictions = zip(*[(directory, *self.classify("{path}/{directory}".format(
            path=path, directory=directory)))
            for directory in directories])
        originals = np.repeat(labels, [len(p) for p in datasets_predictions])
        predictions = np.concatenate(datasets_predictions)
        full_dataset = vstack(datasets)
        name = path.replace("/", "_")
        self._save_results(full_dataset, originals, predictions, name)
        self._svd(full_dataset, originals, predictions, labels,
                  "results/truncated_svd", name)
        self._plot_confusion_matrices(confusion_matrix(
            originals, predictions, labels=labels), labels, "results/confusion_matrices", name)
