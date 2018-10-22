import os
import random
import math
import glob
import shutil
from tqdm import tqdm


def split(directory, training_percentage=0.6, seed=42):
    classes = {
        class_directory: next(os.walk(class_directory))[1]
        for class_directory in filter(
            lambda f: os.path.isdir(f),
            glob.glob('{directory}/*/*'.format(directory=directory)))
    }
    for documents in classes.values():
        random.seed(seed)
        random.shuffle(documents)
    for class_directory, shuffled_documents in tqdm(classes.items(), leave=False):
        n = math.ceil(len(shuffled_documents) * training_percentage)
        for goal, documents in tqdm([("training", shuffled_documents[:n]),
                                     ("testing", shuffled_documents[n:])], leave=False):
            for document in tqdm(documents, leave=False):
                path = "{class_directory}/{document}".format(
                    class_directory=class_directory, document=document)
                target = "{goal}-{path}".format(goal=goal, path=path)
                if not os.path.exists(target):
                    shutil.copytree(path, target)
