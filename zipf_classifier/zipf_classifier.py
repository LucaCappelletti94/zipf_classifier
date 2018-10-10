import re
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, save_npz, vstack
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.metrics import confusion_matrix
from collections import Counter

from tqdm import tqdm


class ZipfClassifier:

    def __init__(self):
        """Create a new instance of ZipfClassifier"""
        self._classifier = None
        self._classes = None
        self._regex = re.compile(r"\W+")

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
                c.update((w for w in re.split(self._regex, f.read()) if w))
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
        for j, counter in tqdm(enumerate(counters), leave=False, total=len(counters)):
            if len(counter) == 0:
                continue
            indices, values = np.array(
                [(keys[k], v) for k, v in counter.items() if k in keys]).T
            frequencies[j][indices] = values
            frequencies[j] /= np.sum(frequencies[j])
        return csr_matrix(frequencies)

    def _build_dataset(self, root: str) -> csr_matrix:
        """Return a csr_matrix with the vector representation of given dataset.
            root:str, root of dataset to load
        """
        return self._counters_to_frequencies(
            self._counters_from_root(root)) / self._corpus_frequencies

    def _build_keymap(self, counters: list) -> dict:
        """Return an enumeration of the given counter keys as dictionary.
            counters:list, the list of Counters objects from which to create the keymap
        """
        print("Determining keyset from {n} counters.".format(n=len(counters)))
        keyset = set()
        for counter in tqdm(counters, leave=False):
            keyset |= set(counter)
        self._keys = {k: i for i, k in enumerate(keyset)}

    def _build_corpus_frequencies(self, matrices):
        """Build the corpus frequencies vector."""
        corpus = vstack(matrices)
        self._corpus_frequencies = corpus.sum(axis=0) / corpus.shape[0]

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

        self._build_corpus_frequencies(
            [matrix for matrix in sparse_matrices.values()])

        return {
            key: matrix / self._corpus_frequencies
            for key, matrix in sparse_matrices.items()
        }

    def _kmeans(self, k: int, points: csr_matrix,
                iterations: int = 50) -> tuple:
        """Return a tuple containing centroids and predictions for given data with k centroids.
            k:int, number of clusters to use for k-means
            points:csr_matrix, points to run kmeans on
            iterations:int, number of iterations of kmeans
        """
        kmeans = KMeans(n_clusters=k, random_state=42,
                        max_iter=iterations, n_jobs=-1)
        kmeans.fit(points)
        return kmeans.cluster_centers_, kmeans.predict(points)

    def _representative_points(self,
                               points: csr_matrix,
                               p: float = 0.1,
                               a: float = 0.2) -> csr_matrix:
        """Return representative points for given set, using given percentage `p` and moving points of `a`.
            points:csr_matrix, points from which to extract the representative points
            p:float, percentage of points to use as representatives
            a:float, percentage of distance to move representatives towards respective centroid
        """

        N = points.shape[0]
        k = np.ceil(N * p**2).astype(int)

        centroids, predictions = self._kmeans(k, points)

        representatives = centroids

        distances = np.squeeze(
            np.asarray(
                np.power(points - centroids[predictions], 2).sum(axis=1)))
        for i in tqdm(range(k), leave=False):
            cluster = points[predictions == i]
            Ni = cluster.shape[0]
            ni = np.floor(p * Ni).astype(int)
            representatives = np.vstack([
                representatives, cluster[np.argpartition(
                    distances[predictions == i].reshape(
                        (Ni, )), ni)[-ni:]] * (1 - a) + centroids[i] * a
            ])
        return csr_matrix(representatives)

    def _build_classifier(self, dataset: dict) -> tuple:
        """Build classifier for given dataset.
            dataset:str, root of given dataset
        """
        print("Determining representative points for {n} classes.".format(
            n=len(dataset.keys())))
        return np.array(list(dataset.keys())), [
            self._representative_points(data)
            for data in tqdm(dataset.values(), leave=False)
        ]

    def fit(self, path: str):
        """Load the dataset at the given path and fit classifier with it.
            path:str, the path from where to load the dataset
        """
        self._classes, self._representatives = self._build_classifier(
            self._build_training_dataset(path))

    def save(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        [
            save_npz(
                "{directory}/{matrix_class}.npz".format(
                    directory=directory, matrix_class=matrix_class), matrix)
            for matrix, matrix_class in zip(self._classes, self._classifier)
        ]
        with open(
                "{directory}/keys.json".format(directory=directory), "w") as f:
            json.dump(self._keys, f)

    def classify(self, path: str) -> list:
        """Load the dataset at the given path and run trained classifier with it.
            path:str, the path from where to load the dataset
        """
        dataset = self._build_dataset(path)
        return self._classes[np.argmin(
            [
                np.min(euclidean_distances(dataset, c), axis=1)
                for c in self._representatives
            ],
            axis=0)]

    def test(self, path: str) -> list:
        """Run test on the classifier over given directory, considering top level as classes.
            path:str, the path from where to run the test.
        """
        directories = self._get_directories(path)
        print("Running {n} tests with the data in {path}.".format(
            n=len(directories), path=path))
        predictions = [(directory,
                        self.classify("{path}/{directory}".format(
                            path=path, directory=directory)))
                       for directory in tqdm(directories, leave=False)]
        y_true = np.array([])
        y_pred = np.array([])
        labels = []
        for original, prediction in predictions:
            y_true = np.hstack([y_true, np.repeat(original, prediction.size)])
            y_pred = np.hstack([y_pred, prediction])
            labels.append(original)
        matrix = confusion_matrix(y_true, y_pred, labels=labels)

        with open("{path}/confusion_matrix.json".format(path=path), "w") as f:
            json.dump({
                "confusion_matrix": matrix.tolist(),
                "labels": labels
            }, f)
