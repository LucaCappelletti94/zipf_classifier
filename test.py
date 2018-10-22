from zipf_classifier import ZipfClassifier, split
import os

root = "datasets"
seed = 1242
n_jobs = 1
training_percentage = 0.7

split(root, training_percentage=training_percentage, seed=seed)

for dataset in next(os.walk(root))[1]:
    print("Working on dataset {dataset}.".format(dataset=dataset))
    classifier, new_classifier = ZipfClassifier(
        n_jobs=n_jobs, seed=seed), ZipfClassifier(n_jobs=n_jobs, seed=seed)
    path = "{root}/{dataset}".format(root=root, dataset=dataset)
    classifier.fit("training-{path}".format(path=path), 10, 100, 0.1, 0.2)
    classifier.save("trained-{path}".format(path=path))
    new_classifier.load("trained-{path}".format(path=path))
    new_classifier.test("testing-{path}".format(path=path))
