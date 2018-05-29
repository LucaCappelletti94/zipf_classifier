from glob import glob
from math import ceil
from multiprocessing import Lock, Manager, Process, cpu_count
from multiprocessing.managers import BaseManager
from os import walk
from os.path import join

from dictances import normal_total_variation
from tqdm import tqdm
from zipf import Zipf
from zipf.factories import ZipfFromDir, ZipfFromFile


class MyManager(BaseManager):
    pass


MyManager.register('tqdm', tqdm)


class ZipfClassifier:
    def __init__(self, path, successes, fails, options=None):
        print("Loading zipfs")
        self._successes = list(self._load(path, successes, "successes"))
        self._fails = list(self._load(path, fails, "fails"))
        print("Building baseline")
        self._baseline = self._build_baseline()
        print("Normalizing to baseline")
        self._normalize(self._successes, "successes")
        self._normalize(self._fails, "fails")
        if options is None:
            options = {}
        options["sort"] = False
        self._factory = ZipfFromFile(options=options)
        self._lock = Lock()
        self._my_manager = MyManager()
        self._my_manager.start()

    def _load(self, path, names, label):
        for name in tqdm(names, desc='Loading %s zipfs' % label, leave=True):
            yield Zipf.load("%s/%s.json" % (path, name))

    def _build_baseline(self):
        n = len(self._successes) + len(self._fails)
        total = Zipf()
        total = sum(
            self._successes,
        ) + sum(
            self._fails
        )

        return (total / n).render()

    def _normalize(self, l: list, lb: str):
        for i, z in tqdm(
            enumerate(l),
            total=len(l),
            unit="zipf",
            desc='Normalizing %s' % lb,
            leave=True
        ):
            l[i] = (z / self._baseline).render()

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def _get_paths(self, root):
        return [y for x in walk(root) for y in glob(join(x[0], '*.txt'))]

    def _test_paths(self, paths, expected, metric):
        successes = 0
        neutrals = 0
        fails = 0
        total = 0
        failed_cases = []
        successes_number = len(self._successes)
        fails_number = len(self._fails)
        fail_paths = []
        run = self._factory.run
        update = self._progress.update
        _successes = self._successes
        _fails = self._fails
        for path in paths:
            zipf = run(path)
            normalized = (zipf / ((zipf + self._baseline) / 2)).render()
            success_distance = 0
            fail_distance = 0

            for success in _successes:
                success_distance += metric(normalized, success)
            success_distance /= successes_number

            for fail in _fails:
                fail_distance += metric(normalized, fail)
            fail_distance /= fails_number

            if success_distance == fail_distance:
                neutrals += 1
            elif (success_distance < fail_distance) == expected:
                successes += 1
            else:
                with open(path, "r") as f:
                    fail_paths.append(f.read())
                fails += 1

            total += 1
            update()

        self._lock.acquire()
        self._info["successes"] += successes
        self._info["neutrals"] += neutrals
        self._info["fails"] += fails
        self._failed_texts += fail_paths
        self._lock.release()

    def run(self, root, expected, metric):
        paths = self._get_paths(root)
        processes = []
        self._info = Manager().dict()
        self._info["successes"] = 0
        self._info["neutrals"] = 0
        self._info["fails"] = 0
        self._failed_texts = Manager().list()
        total = len(paths)
        self._progress = self._my_manager.tqdm(total=total)
        for chunk in self.chunks(paths, ceil(total / cpu_count())):
            p = Process(target=self._test_paths,
                        args=(chunk, expected, metric))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        self._progress.close()

        # for text in self._failed_texts[:5]:
        #     print("=" * 20)
        #     print(text)

        for key, value in self._info.items():
            print("%s: %s, %s" % (key, value, str(value / total * 100) + "%"))


if __name__ == "__main__":
    path = "/Users/lucacappelletti/Datasets/zipf_datasets/zipfs/0"
    fails = [
        "wikipedia",
        "personal_trainer"
    ]
    successes = ["ricette_zafferano"]

    options = {}

    print("Starting classifier")

    classifier = ZipfClassifier(
        path, successes, fails, options=options)

    dataset = "/Users/lucacappelletti/Datasets/zipf_datasets/for_datasets"
    recipes = "%s/recipes" % dataset
    non_recipes = "%s/non_recipes" % dataset

    print("Running Positive test")
    classifier.run(recipes, True, normal_total_variation)

    print("Running Negative test")
    classifier.run(non_recipes, False, normal_total_variation)
