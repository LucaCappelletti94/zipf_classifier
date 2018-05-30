"""Usage example of zipf_classifier."""
from dictances import normal_total_variation
from zipf.factories import ZipfFromDir

from zipf_classifier import ZipfClassifier

# This options are those which the classifier will use to render the zipfs
# from the texts at given paths.
options = {}

classifier = ZipfClassifier(options)

# The dataset on which the zipfs are build SHOULD NOT contain the data on which
# you will run the classifier, otherwise it's obvious it'll get all data right.

# Let's generate an example zipf from the training dataset A_1.
training_dataset_A_path = "path/to/my/training/dataset/A/1"
# We load the texts from the files with extension 'txt'
# NB: no particular formatting is required
dataset_extensions = ["txt"]
# The zipf will be saved in the following position:
zipf_A_1_path = "path/to/my/zipf/A/1.json"

my_factory = ZipfFromDir(use_cli=True, options=options)
my_zipf = my_factory.run(training_dataset_A_path, dataset_extensions)
my_zipf.save(zipf_A_1_path)

# We proceed assuming that the other zipfs are already rendered.

zipf_A_2_path = "path/to/my/zipf/A/2.json"
zipf_A_3_path = "path/to/my/zipf/A/3.json"

zipf_B_1_path = "path/to/my/zipf/B/1.json"
zipf_B_2_path = "path/to/my/zipf/B/2.json"

test_dataset_A_path = "path/to/my/test/dataset/A"
test_dataset_B_path = "path/to/my/test/dataset/B"

class_A = 'A'
class_B = 'B'

# Adding the zipfs of class A
classifier.add_zipf(zipf_A_1_path, class_A)
classifier.add_zipf(zipf_A_2_path, class_A)
classifier.add_zipf(zipf_A_3_path, class_A)

# Adding the zipfs of class B
classifier.add_zipf(zipf_B_1_path, class_B)
classifier.add_zipf(zipf_B_2_path, class_B)

# Rendering the baseline
classifier.render_baseline()

# We choose as metric the normalized total variation

metric = normal_total_variation

# Running the classifier on A and B test set.
results_A = classifier.run(test_dataset_A_path, class_A, metric)
results_B = classifier.run(test_dataset_B_path, class_B, metric)

# Running the classifier on a generic text.
result = classifier.classify("path/to/my/text.txt", metric)
