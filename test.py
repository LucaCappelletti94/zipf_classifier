from zipf_classifier import ZipfClassifier, split
from tqdm import tqdm
import os

root = "test"
split(root)

for dataset in tqdm(next(os.walk(root))[1]):
    classifier = ZipfClassifier()
    path = "{root}/{dataset}".format(root=root, dataset=dataset)
    classifier.fit("training-{path}".format(path=path))
    classifier.test("testing-{path}".format(path=path))
