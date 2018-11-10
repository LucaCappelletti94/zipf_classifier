from zipf_classifier import ZipfClassifier, split
import os
import shutil
import re

os.chdir('./tests')


def purge(pattern):
    for f in os.listdir():
        if re.search(pattern, f):
            shutil.rmtree(f)


def test_version():
    root = "dataset"
    stopwords = "../stopwords/en_stopwords.json"
    seed = 1242
    n_jobs = 1
    training_percentage = 0.7
    k = 10
    neighbours = 5
    iterations = 300

    split(root, training_percentage=training_percentage, seed=seed)

    for dataset in next(os.walk(root))[1]:
        classifier, new_classifier = ZipfClassifier(
            n_jobs=n_jobs), ZipfClassifier(n_jobs=n_jobs)
        classifier.set_seed(seed)
        new_classifier.set_seed(seed)
        path = "{root}/{dataset}".format(root=root, dataset=dataset)
        classifier.fit("training-{path}".format(path=path),
                       k, iterations, stopwords)
        classifier.save("trained-{path}".format(path=path))
        new_classifier.load(
            "trained-{path}".format(path=path), k, iterations,  stopwords)
        new_classifier.test("testing-{path}".format(path=path), range(1, 20))

    # Cleaning up
    purge("\w+-dataset")
    purge("result-\d+")
    assert True
