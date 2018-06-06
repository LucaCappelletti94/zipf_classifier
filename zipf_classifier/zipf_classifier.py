"""Classify dataset with given zipf distributions."""
import math
import operator
import sys
from collections import defaultdict
from glob import glob
from json import dumps
from multiprocessing import Lock, Process, Value, cpu_count
from multiprocessing.managers import BaseManager, DictProxy
from operator import sub
from os import walk
from os.path import isdir, join

from tqdm import tqdm
from zipf import Zipf
from zipf.factories import ZipfFromDir, ZipfFromFile


class MyManager(BaseManager):
    pass


MyManager.register('defaultdict', defaultdict, DictProxy)


class ZipfClassifier:
    """Classify dataset with given zipf distributions."""

    def __init__(self, options=None):
        """Return ZipfClassifier with given options."""
        if options is None:
            options = {}
        options["sort"] = False
        self._options = options
        self._zipfs = {}
        self._file_factory = ZipfFromFile(options=options)
        self._dir_factory = ZipfFromDir(options=options)

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

    def chunks(self, l):
        """Yield successive n-sized chunks from l."""
        n = math.ceil(cpu_count())
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def _test(self, test_couples, metric, results, lock):
        success = 0
        failures = 0
        unclassified = 0
        mistakes = defaultdict(int)
        for path, expectation in test_couples:
            prediction = self.classify(path, metric)
            if prediction == expectation:
                success += 1
            elif prediction is None:
                unclassified += 1
            else:
                failures += 1
                key = "Mistook %s for %s" % (expectation.capitalize(), prediction.capitalize())
                mistakes[key] += 1

            lock.acquire()
            self._i.value += 1
            lock.release()
            sys.stderr.write('\rDone testing {0:%}'.format(
                self._i.value / self._total))

        lock.acquire()
        results["success"] += success
        results["failures"] += failures
        results["unclassified"] += unclassified
        for key, value in mistakes.items():
            results[key] += value
        lock.release()

    def test(self, test_couples, metric):
        """Run prediction test on all given test_couples."""
        chunked = self.chunks(test_couples)
        ps = []
        mgr = MyManager()
        mgr.start()
        r = mgr.defaultdict(int)
        lock = Lock()
        self._total = len(test_couples)
        self._i = Value('i', 0)
        [ps.append(Process(target=self._test, args=(c, metric, r, lock)))
         for c in chunked]
        [p.start() for p in ps]
        [p.join() for p in ps]
        return dict(r)

    def _get_zipf(self, path):
        """Return the zipf from a given path."""
        if isdir(path):
            return self._dir_factory.run(path)
        if path.endswith('.json'):
            return Zipf.load(path)
        return self._file_factory.run(path)

    def _predict(self, path, metric):
        zipf = self._get_zipf(path)
        prediction = ""
        prediction_value = math.inf
        best_second_value = math.inf
        
        for C, Z in self._zipfs.items():
            d = sum([metric(zipf, z) for z in Z]) / len(Z)
            if d < prediction_value:
                prediction = C
                prediction_value = d
            elif d < best_second_value:
                best_second_value = d
        return prediction, abs(prediction_value - best_second_value)

    def classify(self, path, metric, res=1e-4):
        """Return the classification of text at given path."""
        prediction, delta = self._predict(path, metric)
        if delta < res:
            return None
        return prediction
