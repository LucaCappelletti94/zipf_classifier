"""Classify dataset with given zipf distributions."""
from glob import glob
from json import dumps
from math import ceil, isclose
from multiprocessing import Lock, Process, cpu_count
from multiprocessing.managers import BaseManager
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
        return [y for x in walk(root) for y in glob(join(x[0], '*.txt'))]

    def _update_bar(self, bar, n):
        """Increase len of total bar and given bar."""
        self._lock.acquire()
        bar.update(n)
        self._total_bar.update(n)
        self._lock.release()

    def _add_failure(self, n=1):
        """Increase len of failure bar."""
        self._update_bar(self._failure_bar, n)

    def _add_success(self, n=1):
        """Increase len of success bar."""
        self._update_bar(self._success_bar, n)

    def _add_incertain(self, n=1):
        """Increase len of incertain bar."""
        self._update_bar(self._incertain_bar, n)

    def _test(self, path, successes, failures, expected, metric, resolution):
        """Execute for expected value a test on file at given path."""
        zipf = self._factory.run(path)
        denominator = (zipf + self._baseline) / 2
        normalized = (zipf / denominator).render()
        success_distance = sum([metric(normalized, s / denominator)
                                for s in successes]) / len(successes)
        failure_distance = sum([metric(normalized, f / denominator)
                                for f in failures]) / len(failures)

        if isclose(success_distance, failure_distance, rel_tol=resolution):
            self._add_incertain()
        elif (success_distance < failure_distance) == expected:
            self._add_success()
        else:
            self._add_failure()

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

    def _close_bars(self):
        """Close all loading bars."""
        self._total_bar.close()
        self._success_bar.close()
        self._failure_bar.close()
        self._incertain_bar.close()

    def render_baseline(self):
        """Render baseline zipf."""
        self._baseline = self._get_baseline()

    def run(self, root, expected, metric, resolution=1e-5):
        """Execute tests on all files under given root, using given metric."""
        paths = self._get_paths(root)
        total_paths = len(paths)
        self._init_bars(total_paths)

        chunks = self._chunks(paths, ceil(total_paths / cpu_count()))
        processes = [Process(target=self._tests, args=(
            chk, expected, metric, resolution)) for chk in chunks]
        [p.start() for p in processes]
        [p.join() for p in processes]

        self._close_bars()
        print("\n" * 4)
