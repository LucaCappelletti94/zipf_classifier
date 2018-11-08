from zipf_classifier import ZipfClassifier, split
import os
import shutil
import filecmp

os.chdir('./tests')


def test_version():
    root = "test_dataset"
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
        classifier.fit("training-{path}".format(path=path), 10, 100, 0.1, 0.2)
        classifier.save("trained-{path}".format(path=path))
        new_classifier.load("trained-{path}".format(path=path))
        new_classifier.test("testing-{path}".format(path=path))

    differences = filecmp.dircmp("expected results", "results").diff_files

    # Cleaning up
    shutil.rmtree("testing-test_dataset")
    shutil.rmtree("training-test_dataset")
    shutil.rmtree("trained-test_dataset")
    shutil.rmtree("results")
    assert not differences
