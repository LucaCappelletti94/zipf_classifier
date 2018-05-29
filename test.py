"""Usage example of zipf_classifier."""
from json import load
from os import walk

from dictances import (bhattacharyya, hellinger, jensen_shannon,
                       kullback_leibler, normal_total_variation)
from tqdm import tqdm

from zipf_classifier import ZipfBinaryClassifier


def get_options(root):
    """Return the options in json dictionary at given path."""
    with open("%s/options.json" % root, 'r') as f:
        return load(f)


test_root = "/Users/lucacappelletti/Datasets/zipf_datasets/zipfs"
dataset_root = "/Users/lucacappelletti/Datasets/zipf_datasets/for_datasets"
test_zipfs = [x[0] for x in walk(test_root) if x[0] != test_root]

metrics = [bhattacharyya, hellinger, jensen_shannon,
           kullback_leibler, normal_total_variation]

for i, current_test in enumerate(test_zipfs):
    print("\033[1mRunning test %s of %s\033[0;0m" % (i + 1, len(test_zipfs)))
    classifier = ZipfBinaryClassifier(get_options(current_test))
    print("Current classifier has options: %s\n" % classifier)

    zipfs = {
        True: [
            "ricette_zafferano"
        ],
        False: [
            "wikipedia",
            "personal_trainer"
        ]
    }

    datasets = {
        True: [
            "recipes"
        ],
        False: [
            "non_recipes"
        ]
    }

    z_path = current_test + "/%s.json"

    [classifier.add_zipf(z_path % path, expected) for expected, paths in tqdm(
        zipfs.items(),
        desc="Loading zipfs",
        unit=' class',
        leave=True,
        unit_scale=True
    ) for path in tqdm(
        paths,
        desc="Loading zipfs of class %s" % expected,
        unit=' zipf',
        leave=True,
        unit_scale=True
    )]

    print("\n")

    classifier.render_baseline()

    print("\n")

    for expected, paths in datasets.items():
        for path in paths:
            print("\033[1mWorking on %s \033[0;0m\n" % path)
            for metric in metrics:
                print("\033[1mUsing metric %s \033[0;0m\n" % metric.__name__)
                classifier.run("%s/%s" % (dataset_root, path),
                               expected, normal_total_variation)
