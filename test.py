"""Usage example of zipf_classifier."""
from json import dump, load
from os import walk

from dictances import (hellinger, jensen_shannon, kullback_leibler,
                       normal_total_variation)
from tqdm import tqdm

from zipf_classifier import ZipfClassifier


def get_options(root):
    """Return the options in json dictionary at given path."""
    with open("%s/options.json" % root, 'r') as f:
        return load(f)


test_root = "/Users/lucacappelletti/Datasets/zipf_datasets/zipfs"
dataset_root = "/Users/lucacappelletti/Datasets/zipf_datasets/for_datasets"
test_zipfs = [x[0] for x in walk(test_root) if x[0] != test_root]

metrics = [hellinger, jensen_shannon,
           kullback_leibler, normal_total_variation]

tests_results = []

for i, current_test in enumerate(test_zipfs):
    print("\033[1mRunning test %s of %s\033[0;0m" % (i + 1, len(test_zipfs)))
    classifier = ZipfClassifier(get_options(current_test))
    print("Current classifier has options: %s\n" % classifier)

    zipfs = {
        'recipe': [
            "ricette_zafferano"
        ],
        'non_recipe': [
            "wikipedia",
            "personal_trainer"
        ]
    }

    datasets = {
        'recipe': [
            "recipes"
        ],
        'non_recipe': [
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

    test_results = {}

    for expected, paths in datasets.items():
        expected_result = {}
        for path in paths:
            path_results = {}
            print("\033[1mWorking on %s \033[0;0m\n" % path)
            for metric in metrics:
                print("\033[1mUsing metric %s \033[0;0m\n" % metric.__name__)
                path_results[metric.__name__] = classifier.run("%s/%s" % (dataset_root, path),
                                                               expected, metric)
            expected_result[path] = path_results
        test_results[expected] = expected_result
    tests_results.append({
        "number": i,
        "options": str(classifier),
        "results": test_results
    })

with open('test_results.json', 'w') as f:
    dump(tests_results, f, sort_keys=True, indent=4)
