import re
import json
import os
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, save_npz, vstack, load_npz
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from seaborn import heatmap
from wordcloud import WordCloud
from tqdm import tqdm
import math
import warnings
from typing import Iterator, Generator, List, Tuple, Dict, Union


class ZipfClassifier:

    def __init__(self, n_jobs: int=1):
        """Create a new instance of ZipfClassifier
            n_jobs:int, number of parallel jobs to use.
        """
        self._classifier, self._classes, self._n_jobs, self._regex = None, None, n_jobs, re.compile(
            r"\W*\d*")
        self._seed = 2007

    def _get_directories(self, path: str) -> List[str]:
        """Return the directories inside the first level of a given path.
            path:str, the path from where to load the first level directories list.
        """
        return next(os.walk(path))[1]

    def _lazy_file_loader(self, directory: str, files: List[str])->Generator[str, None, None]:
        """Yield lazily loaded files.
            directory:str, the directory of given files list
            files: List[str], list of files
        """
        for file in files:
            with open("{directory}/{file}".format(directory=directory, file=file), "r", encoding='utf-8') as f:
                yield f.read()

    def _lazy_directory_loader(self, directory: str)->Generator[Generator[str, None, None], None, None]:
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
            c.update((w for w in re.split(self._regex, file.lower())
                      if w and len(w) > 1 and w not in self._stopwords))
        return c

    def _counters_from_file_iterator(self, file_iterator: Iterator) -> List[Counter]:
        """Return list of counters for the documents found in given root."""
        return [
            self._counter_from_path(files) for files in file_iterator
        ]

    def _counters_to_frequencies(self, counters: list) -> csr_matrix:
        """Return a csr_matrix representing sorted counters as frequencies.
            counters:list, the list of Counters objects from which to create the csr_matrix
        """
        print("Converting {n} counters to sparse matrix.".format(
            n=len(counters)))
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
                frequencies[non_zero_rows_number][indices] = values / \
                    np.max(values)
                non_zero_rows_number += 1
        return csr_matrix(frequencies[:non_zero_rows_number])

    def _build_dataset(self, root: str) -> csr_matrix:
        """Return a csr_matrix with the vector representation of given dataset.
            root:str, root of dataset to load
        """
        return self._counters_to_frequencies(
            self._counters_from_file_iterator(root))

    def _build_keymap(self, counters: List[Counter]) -> Dict[str, int]:
        """Return an enumeration of the given counter keys as dictionary.
            counters:list, the list of Counters objects from which to create the keymap
        """
        print("Determining keyset from {n} counters.".format(n=len(counters)))
        keyset = set()
        for counter in counters:
            keyset |= set(counter)
        self._keys = {k: i for i, k in enumerate(keyset)}
        self._words = np.array(list(self._keys.keys()))

    def _build_training_dataset(self, root: str) -> Dict[str, csr_matrix]:
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

    def _kmeans(self,  points: csr_matrix) -> tuple:
        """Return a tuple containing centroids and predictions for given data with k centroids.
            points:csr_matrix, points to run kmeans on
        """
        print("Running kmeans on {n} points with k={k} and {m} iterations.".format(
            n=points.shape[0], k=self._k, m=self._iterations))
        kmeans = KMeans(
            n_clusters=self._k, max_iter=self._iterations, random_state=self._seed, n_jobs=self._n_jobs)
        kmeans.fit(points)
        return kmeans.cluster_centers_, kmeans.predict(points)

    def _representative_points(self,
                               points: csr_matrix,
                               points_percentage: float=0.1,
                               distance_percentage: float=0.2) -> csr_matrix:
        """Return representative points for given set, using given percentage `points_percentage` and moving points of `distance_percentage`.
            points:csr_matrix, points from which to extract the representative points
            points_percentage:float, percentage of points to use as representatives
            distance_percentage:float, percentage of distance to move representatives towards respective centroid
        """
        centroids, predictions = self._kmeans(points)

        print("Determining representative points.")

        representatives = centroids

        distances = np.squeeze(
            np.asarray(
                np.power(points - centroids[predictions], 2).sum(axis=1)))
        for i in tqdm(range(self._k), leave=False):
            mask = predictions == i
            cluster = points[mask]
            Ni = cluster.shape[0]
            ni = np.floor(points_percentage * Ni).astype(int)
            partition = np.argpartition(
                distances[mask].reshape(
                    (Ni, )), -ni)[-ni:]

            cluster_representatives = cluster[partition] * (
                1 - distance_percentage) + centroids[i] * distance_percentage
            if representatives is None:
                representatives = cluster_representatives
            else:
                representatives = np.vstack([
                    representatives, cluster_representatives
                ])
        matrix = csr_matrix(representatives)
        self._representatives_sizes.append(matrix.shape[0])
        return matrix

    def _build_classifier(self, dataset: Dict[str, csr_matrix]) -> Tuple[np.ndarray, csr_matrix]:
        """Return a tuple with dataset classes and centroids.
            dataset:[str, csr_matrix], dictionary representing the training dataset.
        """
        print("Determining centroids for {n} classes.".format(
            n=len(dataset.keys())))
        return np.array(list(dataset.keys())), vstack([
            self._representative_points(data)
            for data in dataset.values()
        ])

    def fit(self, path: str, k: int,
            iterations: int, stopwords_path: str = None) -> Tuple[np.ndarray, csr_matrix]:
        """Load the dataset at the given path and fit classifier with it.
            path:str, the path from where to load the dataset
            k:int, number of clusters
            iterations:int, number of iterations of kmeans
            stopwords_path: str, path to stopwords file.
        """
        self._k, self._iterations = k, iterations
        self._load_stop_words(stopwords_path)
        self._representatives_sizes = []
        self._classes, self._representatives = self._build_classifier(
            self._build_training_dataset(path))

    def _format_classes_path(self, directory: str)->str:
        """Return formatted classes path.
            directory:str, the directory for the classes file.
        """
        return "{directory}/classes.npy".format(directory=directory)

    def _format_representatives_sizes_path(self, directory: str)->str:
        """Return formatted representatives_sizes path.
            directory:str, the directory for the representatives_sizes file.
        """
        return "{directory}/representatives_sizes.json".format(directory=directory)

    def _format_keys_path(self, directory: str)->str:
        """Return formatted keys path.
            directory:str, the directory for the keys file.
        """
        return "{directory}/keys.json".format(directory=directory)

    def _format_classifier_path(self, directory: str, k: int, iterations: int)->str:
        """Return formatted classifier path.
            directory:str, the directory for the classifier file.
            k:int, number of clusters
            iterations:int, number of iterations of kmeans
        """
        return "{directory}/classifier-{k}-{iterations}.npz".format(
            directory=directory, k=k, iterations=iterations)

    def _format_paths(self, directory: str, k: int, iterations: int)->Tuple[str, str, str]:
        """Return formatted classes, keys and classifier paths.
            directory:str, the directory for the classifier file.
            k:int, number of clusters
            iterations:int, number of iterations of kmeans
        """
        return self._format_classes_path(directory), self._format_representatives_sizes_path(directory), self._format_keys_path(directory), self._format_classifier_path(directory, k, iterations)

    def _load_stop_words(self, stopwords_path: str):
        """Load stopwords json file from given path.
            stopwords_path: str, path to stopwords file.
        """
        if stopwords_path is not None:
            with open(stopwords_path, "r", encoding='utf-8') as f:
                self._stopwords = json.load(f)
        else:
            self._stopwords = []

    def load(self, directory: str, k: int, iterations: int, stopwords_path: str=None):
        """Load the trained classifier from given directory.
            directory:str, the directory for the classifier file.
            k:int, number of clusters
            iterations:int, number of iterations of kmeans
            stopwords_path: str, path to stopwords file.
        """
        classes_path, representatives_sizes_path, keys_path, classifier_path = self._format_paths(
            directory, k, iterations)
        self._k = k
        self._iterations = iterations
        self._classes = np.load(classes_path)
        with open(keys_path, "r", encoding='utf-8') as f:
            self._keys = json.load(f)
        with open(representatives_sizes_path, "r", encoding='utf-8') as f:
            self._representatives_sizes = json.load(f)
        self._words = np.array(list(self._keys.keys()))
        self._representatives = load_npz(classifier_path)
        self._load_stop_words(stopwords_path)

    def save(self, directory: str):
        """Save the trained classifier to given directory.
            directory:str, the directory for the classifier file.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        classes_path, representatives_sizes_path, keys_path, classifier_path = self._format_paths(
            directory, self._k, self._iterations)
        np.save(classes_path, self._classes)
        with open(keys_path, "w") as f:
            json.dump(self._keys, f)
        with open(representatives_sizes_path, "w") as f:
            json.dump(self._representatives_sizes, f)
        save_npz(classifier_path, self._representatives)

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
        maximum_x, maximum_y = np.max(reduced, axis=0)*1.05
        minimum_x, minimum_y = np.min(
            reduced, axis=0)-np.max(reduced, axis=0)*.05
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
        n = len(labels) + 1
        plt.figure(figsize=(5*n, 10))
        colors = ["red", "green", "blue", "orange", "purple", "black"]

        cumulative_original_ax = self._setup_axis(n, n, "Originals", *margins)
        cumulative_prediction_ax = self._setup_axis(
            n, 2*n, "Predictions", *margins)

        for i, (label, color) in enumerate(zip(labels, colors), 1):
            original_ax = self._setup_axis(
                n, i, "Original {label}".format(label=label), *margins)
            prediction_ax = self._setup_axis(
                n, n+i, "Prediction {label}".format(label=label), *margins)

            for axs, column in zip(((original_ax, cumulative_original_ax), (prediction_ax, cumulative_prediction_ax)), columns):
                for ax in axs:
                    indices = df[column] == label
                    ax.scatter(
                        df.loc[indices, 'a'],
                        df.loc[indices, 'b'],
                        c=color,
                        label=label,
                        alpha=0.5,
                        s=20)

            original_ax.legend(loc='upper right')
            prediction_ax.legend(loc='upper right')

        cumulative_original_ax.legend(loc='upper right')
        cumulative_prediction_ax.legend(loc='upper right')

        plt.suptitle("{title} - Truncated SVD".format(title=title))

        plt.savefig(
            "{directory}/{title} - Truncated SVD.png".format(directory=directory, title=title))
        plt.clf()
        plt.close()

    def _heatmap(self, axis: mpl.axes.SubplotBase, data: np.matrix, labels: list, title: str, fmt: str):
        """ Plot given matrix as heatmap.
            axis: mpl.axes.SubplotBase, the subplot on which to plot
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
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        axis.set_title(title)

    def _plot_confusion_matrices(self, confusion_matrix: np.matrix, labels: list, directory: str, title: str):
        """Plot default and normalized confusion matrix.
            confusion_matrix:np.matrix, the confusion matrix to be plot.
            labels:list, list of labels of matrix data.
            directory:str, the directory were to save the matrix.
            title:str, the title for the documents.
        """
        plt.figure(figsize=(9, 4))
        plt.subplots_adjust(wspace=0.5)
        self._heatmap(plt.subplot(1, 2, 1), confusion_matrix,
                      labels, "Confusion matrix", "d")
        self._heatmap(plt.subplot(1, 2, 2), confusion_matrix.astype(np.float) /
                      confusion_matrix.sum(axis=1)[:, np.newaxis], labels, "Normalized confusion matrix", "0.4g")
        plt.suptitle("Confusion Matrices - {title}".format(title=title))
        plt.savefig("{directory}/{title} - Confusion matrices.png".format(directory=directory,
                                                                          title=title))
        plt.clf()
        plt.close()

    def _plot_wordcloud(self, axis: mpl.axes.SubplotBase, wc: WordCloud, words: List[str], label: str):
        """Plot a given wordcloud on given axis.
            axis: mpl.axes.SubplotBase, the subplot on which to plot
            wx: WordCloud, wordcloud object to use
            words: List[str], list of words to plot
            label:str, label for wordcloud
        """
        wordcloud = wc.generate(
            " ".join(words))
        axis.imshow(wordcloud, interpolation='bilinear')
        axis.set_title("Word cloud - {label}".format(label=label))
        plt.axis("off")

    def _plot_wordclouds(self, important_words: List[List[List[str]]], labels: List[str], path: str, title: str):
        """Plot word clouds.
            important_words: List[List[List[str]]], list of important words
            labels:List[str], list of unique labels
            directory:str, path to directory where to save results
            name:str, project name
        """
        n = len(labels) + 1
        plt.figure(figsize=(4*n, 4))
        global_words_list = []
        wc = WordCloud(width=400, height=400, random_state=self._seed)
        for i, (words_list, label) in enumerate(zip(important_words, labels), 1):
            words_set = [
                word for words in words_list for word in words]
            global_words_list += words_set
            self._plot_wordcloud(plt.subplot(1, n, i), wc, words_set, label)
        self._plot_wordcloud(plt.subplot(1, n, n), wc,
                             global_words_list, "All")
        plt.suptitle("Word Clouds - {title}".format(title=title))
        plt.savefig("{path}/{title} - Word Clouds.png".format(path=path,
                                                              title=title))
        plt.clf()
        plt.close()

    def _plot_representatives_points_usage(self, representative_points_usage: Tuple, labels: List[str], path: str, title: str):
        """Plot information of representatives points usage
            representative_points_usage: Dict, information on representatives points usage
            labels:List[str], list of unique labels
            directory:str, path to directory where to save results
            title:str, project title
        """
        n = len(representative_points_usage)+1
        plt.figure(figsize=(4*n, 4))
        stack_total = None
        total_ax = plt.subplot(1, n, n)
        total_ax.set_title("Total usage - All")
        for i, (info, label) in enumerate(zip(representative_points_usage, labels), 1):
            ax = plt.subplot(1, n, i)
            total = info["total"]
            ax.bar(np.arange(len(total)), total)
            total_ax.bar(np.arange(len(total)), total, bottom=stack_total)
            if stack_total is None:
                stack_total = total
            else:
                stack_total += total
            ax.set_title("Total usage - {label}".format(label=label))
        plt.suptitle(
            "Representatives points usage - {title}".format(title=title))
        plt.savefig("{path}/{title} - Representatives points usage.png".format(path=path,
                                                                               title=title))
        plt.clf()
        plt.close()

    def _save_results(self, directory: str, name: str, dataset: csr_matrix, originals: np.ndarray, predictions: np.ndarray, labels: List[str], important_words: List[List[List[str]]], representative_points_usage: Dict):
        """Save classification results.
            directory:str, path to directory where to save results
            name:str, project name
            dataset:csr_matrix, classified compiled dataset
            originals:np.ndarray, original labels
            predictions:np.ndarray, predicted labels
            labels:List[str], list of unique labels
            important_words: List[List[List[str]]], list of important words
            representative_points_usage: Dict, information on representatives points usage
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save("{directory}/{name}-originals".format(directory=directory,
                                                      name=name), originals)
        np.save("{directory}/{name}-predictions".format(directory=directory,
                                                        name=name), predictions)
        self._svd(dataset, originals, predictions, labels,
                  directory, name)
        self._plot_confusion_matrices(confusion_matrix(
            originals, predictions, labels=labels), labels, directory, name)
        self._plot_wordclouds(important_words, labels, directory, name)
        self._plot_representatives_points_usage(
            representative_points_usage, labels, directory, name)

    def _plot_scores(self, scores:List[float], directory:str, title:str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.plot(list(range(len(scores))), scores)
        plt.xlabel("Neighbours")
        plt.ylabel("Precision")
        plt.title("Precision for neighbours considered")
        plt.savefig("{directory}/{title} - Precision for neighbours considered.png".format(directory=directory,
                                                                               title=title))
        plt.clf()
        plt.close()

    def _determine_important_words(self, masks:np.ndarray, partitions:np.ndarray, predictions_indices:np.ndarray)->List[List[str]]:
        cluster_mask = masks[predictions_indices.reshape(-1, 1), np.arange(
            predictions_indices.shape[0]).reshape(-1, 1)].reshape(*partitions.shape)
        return [self._words[np.argmax(
                self._representatives[p[m]], axis=1)][0].tolist() for p, m in zip(partitions, cluster_mask)]

    def _determine_representative_points_usage(self, partitions:np.ndarray):
        representative_points_usage = {}
        representative_points_usage["total"] = np.bincount(
            partitions.flatten(), minlength=self._representatives.shape[0])
        return representative_points_usage

    def _classify(self, dataset: csr_matrix, neighbours: int, determine_important_words: bool, determine_representative_points_usage: bool) -> Union[Tuple[csr_matrix, np.ndarray], Tuple[csr_matrix, np.ndarray, List[List[str]]]]:
        """Return a tuple with classified dataset and classification vector.
            dataset:csr_matrix, dataset to classify
            neighbours:int, number of neighbours to consider for classification.
            determine_important_words:bool, whetever to determine the important words for documents classification.
            determine_representative_points_usage:bool, whetever to determine the usage of representative points.
        """
        if self._distances is None:
            distances = cosine_distances(dataset, self._representatives)
            self._distances = distances
        else:
            distances = self._distances
        partitions = np.argpartition(distances, neighbours, axis=1)[
            :, :neighbours]
        repeated_classes = np.repeat(
            self._classes, self._representatives_sizes)
        masks = repeated_classes[partitions].reshape(
            1, *partitions.shape) == self._classes.reshape(self._classes.size, 1, 1)

        cluster_distances = distances[np.arange(
            partitions.shape[0]).reshape(-1, 1), partitions]
        nan_cluster_distances = np.repeat(
            cluster_distances.reshape(1, *cluster_distances.shape), masks.shape[0], axis=0)
        nan_cluster_distances[~masks] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            nan_means = np.nanmean(nan_cluster_distances, axis=2)
        predictions_indices = np.nanargmin(
            nan_means,
            axis=0)
        # End mean
        predictions = self._classes[predictions_indices]
        results = (dataset, predictions)
        if determine_important_words:
            results = (*results, self._determine_important_words(masks, partitions, predictions_indices))
        if determine_representative_points_usage:
            results = (*results, self._determine_representative_points_usage(partitions))
        return results

    def classify_directory(self, directory: str, neighbours: int, determine_important_words: bool=False, determine_representative_points_usage: bool=False) -> Union[Tuple[csr_matrix, np.ndarray], Tuple[csr_matrix, np.ndarray, List[List[str]]]]:
        """Load the dataset at the given path and run trained classifier with it.
            directory:str, the path from where to load the dataset.
            neighbours:int, number of neighbours to consider for classification.
            determine_important_words:bool, whetever to determine the important words for documents classification.
            determine_representative_points_usage:bool, whetever to determine the usage of representative points.
        """
        return self._classify(self._build_dataset(self._lazy_directory_loader(directory)), neighbours, determine_important_words, determine_representative_points_usage)

    def classify_texts(self, texts: List[str], neighbours: int, determine_important_words: bool=False, determine_representative_points_usage: bool=False) -> Union[Tuple[csr_matrix, np.ndarray], Tuple[csr_matrix, np.ndarray, List[List[str]]]]:
        """Return the classification of given texts
            texts:List[str], the texts to classify.
            neighbours:int, number of neighbours to consider for classification.
            determine_important_words:bool, whetever to determine the important words for documents classification.
            determine_representative_points_usage:bool, whetever to determine the usage of representative points.
        """
        return self._classify(self._build_dataset([texts]), neighbours, determine_important_words, determine_representative_points_usage)

    def classify_text(self, text: str, neighbours: int, determine_important_words: bool=False, determine_representative_points_usage: bool=False) -> Union[Tuple[csr_matrix, np.ndarray], Tuple[csr_matrix, np.ndarray, List[List[str]]]]:
        """Return the classification of given text
            text:str, the text to classify.
            neighbours:int, number of neighbours to consider for classification.
            determine_important_words:bool, whetever to determine the important words for documents classification.
            determine_representative_points_usage:bool, whetever to determine the usage of representative points.
        """
        return self.classify_texts([text], neighbours, determine_important_words, determine_representative_points_usage)

    def set_seed(self, seed: int):
        """Set random seed to reproduce results.
            seed:int, the random seed to use for the test.
        """
        random.seed(seed)
        np.random.seed(seed)
        self._seed = seed

    def test(self, path: str, neighbours: Union[int, List[int]]):
        """Run test on the classifier over given directory, considering top level as classes.
            path:str, the path from where to run the test.
            neighbours:int, number of neighbours to consider for classification.
        """
        labels = self._get_directories(path)
        print("Running {n} tests with the data in {path}.".format(
            n=len(labels), path=path))
        if isinstance(neighbours, int):
            neighbours = [neighbours]
        datasets = [
            self._build_dataset(self._lazy_directory_loader("{path}/{directory}".format(
                    path=path, directory=label)))
            for label in labels
        ]
        precision_scores = []
        for n in tqdm(neighbours):
            self._distances = None
            _, predictions, important_words, representative_points_usage = zip(*[self._classify(dataset, n, True, True)
                for dataset, label in zip(datasets, labels)])
            originals = np.repeat(labels, [len(p) for p in predictions])
            self._save_results("results - {n}".format(n=n), path.replace(
                "/", "_"), vstack(datasets), originals, np.concatenate(predictions), labels, important_words, representative_points_usage)
            precision_scores.append(precision_score(originals, np.concatenate(predictions), labels = labels, average='weighted'))
        self._plot_scores(precision_scores, "precision_scores".format(n=n), path.replace("/", "_"))