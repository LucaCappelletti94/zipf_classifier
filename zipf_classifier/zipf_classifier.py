"""Classify dataset with given zipf distributions."""
from glob import glob
from json import dumps
from math import ceil
from multiprocessing import Lock, Manager, Process, Value, cpu_count
from multiprocessing.managers import BaseManager
from operator import sub
from os import walk
from os.path import join

from tqdm import tqdm
from zipf import Zipf
from zipf.factories import ZipfFromFile


class MyManager(BaseManager):
    """Return an overridable custom multiprocessing Manager."""

    pass


MyManager.register('tqdm', tqdm)


class ZipfBinaryClassifier:
    """Classify dataset with given zipf distributions."""

    def __init__(self, options=None):
        """Return ZipfBinaryClassifier with given options."""
        if options is None:
            options = {}
        options["sort"] = False
        self._options = options
        self._zipfs = {}
        self._factory = ZipfFromFile(options=options)
        self._my_manager = MyManager()
        self._my_manager.start()
        self._lock = Lock()

    def __repr__(self):
        """Return representation of ZipfFromFile."""
        return dumps(self._options, indent=4, sort_keys=True)

    __str__ = __repr__

    def add_zipf(self, path, expected):
        """Add a zipf for the given class to the classifier."""
        zipf = Zipf.load(path).normalize()
        if expected in self._zipfs:
            self._zipfs[expected].append(zipf)
        else:
            self._zipfs[expected] = [zipf]

    def _get_baseline(self):
        """Return a zipf obtained by normalizing and summing every zipf."""
        zipfs = [item for sublist in self._zipfs.values() for item in sublist]
        return (sum(zipfs) / len(zipfs)).render()

    def _chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def _get_paths(self, root):
        """Return list of all path under given root with .txt extension."""
        pattern = '*.txt'
        return [y for x in tqdm(
            walk(root),
            desc="Searching %s files" % pattern,
            unit=' dir',
            leave=True,
            unit_scale=True
        ) for y in tqdm(
            glob(join(x[0], pattern)),
            desc="Loading files from %s" % x[0].split("/")[0],
            unit=' file',
            leave=True,
            unit_scale=True
        )]

        print('\n')

    def _update_bar(self, bar, counter, n):
        """Increase len of total bar and given bar."""
        self._lock.acquire()
        bar.update(n)
        self._total_bar.update(n)
        counter.value += n
        self._lock.release()

    def _add_failure(self, path, success_distance, failure_distance,
                     classification_distance, normalized_distance):
        """Increase len of failure bar."""
        self._update_bar(self._failure_bar, self._errors, 1)
        with open(path, 'r') as f:
            text = f.read()
        self._failures.append({
            "text": text,
            "success_distance": success_distance,
            "failure_distance": failure_distance,
            "classification_distance": classification_distance,
            "normalized_distance": normalized_distance
        })

    def _add_success(self):
        """Increase len of success bar."""
        self._update_bar(self._success_bar, self._successes, 1)

    def _add_incertain(self):
        """Increase len of incertain bar."""
        self._update_bar(self._incertain_bar, self._incertains, 1)

    def _add_classification_distance(self, value):
        self._classification_distance.value += value

    def _add_norm_cls_dist(self, value):
        self._norm_cls_dist.value += value

    def _get_distances(self, path, metric):
        zipf = self._factory.run(path)
        denominator = (zipf + self._baseline) / 2
        norm = (zipf / denominator).render()

        return {_class: sum([metric(norm, z) for z in zipfs]) / len(zipfs)
                for _class, zipfs in self._zipfs.items()}

    def _test(self, path, successes, failures, expected, _metric, resolution):
        """Execute for expected value a test on file at given path."""
        self._get_distances(path, metric)

        classification_distance = abs(sub(*distances.values()))
        normalized_distance = classification_distance / sum(distances.values())

        self._add_classification_distance(classification_distance)
        self._add_norm_cls_dist(normalized_distance)

        if classification_distance < resolution:
            self._add_incertain()
        elif success_distance < failure_distance:
            self._add_success()
        else:
            self._add_failure(path, success_distance, failure_distance,
                              classification_distance, normalized_distance)

    def _tests(self, paths, expected, metric, resolution):
        """Execute batch of tests on given paths for expected value."""
        successes = self._zipfs[expected]
        failures = self._zipfs[not expected]
        [self._test(path, successes, failures, expected, metric, resolution)
         for path in paths]

    def _get_bar(self, total, label, position):
        """Return a custom loading bar."""
        return self._my_manager.tqdm(
            total=total,
            desc=label,
            unit=' zipf',
            leave=True,
            position=position,
            unit_scale=True
        )

    def _init_bars(self, total):
        """Init all loading bars."""
        self._total_bar = self._get_bar(total, 'Tested files', 0)
        self._success_bar = self._get_bar(total, 'Successes', 1)
        self._failure_bar = self._get_bar(total, 'Failures', 2)
        self._incertain_bar = self._get_bar(total, 'Incertain', 3)

    def _init_counters(self):
        """Init all counters."""
        self._errors = Value('i', 0)
        self._successes = Value('i', 0)
        self._incertains = Value('i', 0)

    def _close_bars(self):
        """Close all loading bars."""
        self._total_bar.close()
        self._success_bar.close()
        self._failure_bar.close()
        self._incertain_bar.close()

    def render_baseline(self):
        """Render baseline zipf."""
        print("Calculating baseline.")
        self._baseline = self._get_baseline()
        print("\nNormalizing given zipfs")
        for expected, zipfs in tqdm(
            self._zipfs.items(),
            desc="Normalizing zipfs",
            unit=' class',
            leave=True,
            unit_scale=True
        ):
            for i, zipf in tqdm(
                enumerate(zipfs),
                total=len(zipfs),
                desc="Normalizing zipfs of class %s" % expected,
                unit=' zipf',
                leave=True,
                unit_scale=True
            ):
                zipfs[i] = (zipf / self._baseline).render()

    def run(self, root, expected, metric, resolution=1e-4):
        """Execute tests on all files under given root, using given metric."""
        paths = self._get_paths(root)
        total_paths = len(paths)
        self._init_bars(total_paths)
        self._init_counters()

        self._classification_distance = Value('d', 0)
        self._norm_cls_dist = Value('d', 0)
        self._failures = Manager().list()

        chunks = self._chunks(paths, ceil(total_paths / cpu_count()))
        processes = [Process(target=self._tests, args=(
            chk, expected, metric, resolution)) for chk in chunks]
        [p.start() for p in processes]
        [p.join() for p in processes]

        self._close_bars()
        print("\n" * 4)
        return {
            "total": total_paths,
            "errors": self._errors.value,
            "successes": self._successes.value,
            "incertain": self._incertains.value,
            "classification_distance": self._classification_distance.value,
            "normalized_classification_distance": self._norm_cls_dist.value,
            "failures": list(self._failures)
        }

    def classify(self, path, metric, resolution=1e-4):
        """Return the classification of text at given path."""
        distance = self._get_distances(path, metric)

        if abs(sub(*distances.values())) < resolution:
            return None
        return min(distances, key=distances.get)
