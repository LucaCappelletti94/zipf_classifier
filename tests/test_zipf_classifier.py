import os

from dictances import intersection_squared_hellinger, jensen_shannon

from zipf_classifier import ZipfClassifier


def test_zipf_classifier():
    current_path = os.path.dirname(__file__) + "/../test_data"
    class_A = "A"
    class_B = "B"
    zipf_A = "%s/%s.json" % (current_path, class_A)
    zipf_B = "%s/%s.json" % (current_path, class_B)
    path_A = "%s/%s" % (current_path, class_A)
    path_B = "%s/%s" % (current_path, class_B)
    # Creating the classifier
    classifier = ZipfClassifier()
    # Adding learning zipfs to the classifier
    classifier.add_zipf(zipf_A, class_A)
    classifier.add_zipf(zipf_B, class_B)
    # Loading tests
    A = [("%s/%s" % (path_A, z), class_A)
         for z in os.listdir(path_A) if z.endswith('.json')]
    B = [("%s/%s" % (path_B, z), class_B)
         for z in os.listdir(path_B) if z.endswith('.json')]
    tests = A + B
    # Running tests
    assert (
        {'success': 51, 'failures': 5, 'unclassified': 0,
            'mean_delta': 0.08164044130948692, 'Mistook B for A': 5},
        {'success': 56, 'failures': 0, 'unclassified': 0,
            'mean_delta': 5.481168422138327}
    ) == (
        classifier.test(tests, jensen_shannon),
        classifier.test(tests, intersection_squared_hellinger)
    )
