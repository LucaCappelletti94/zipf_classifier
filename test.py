"""Usage example of zipf_classifier."""
from json import load
from os import walk

from dictances import normal_total_variation

from zipf_classifier import ZipfBinaryClassifier


def get_options(root):
    """Return the options in json dictionary at given path."""
    with open("%s/options.json" % root, 'r') as f:
        return load(f)


test_root = "/Users/lucacappelletti/Datasets/parsed_zipfs/zipfs"
dataset_root = "/Users/lucacappelletti/Datasets/zipf_datasets/for_datasets"
test_zipfs = [x[0] for x in walk(test_root) if x[0] != test_root]

for current_test in test_zipfs:
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

    for expected, paths in zipfs.items():
        for path in paths:
            classifier.add_zipf("%s/%s.json" % (current_test, path), expected)

    classifier.render_baseline()

    for expected, paths in datasets.items():
        for path in paths:
            print("\033[1mWorking on %s \033[0;0m" % path)
            classifier.run("%s/%s" % (dataset_root, path),
                           expected, normal_total_variation)
