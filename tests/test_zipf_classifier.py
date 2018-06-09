import os
from math import isclose

from dictances import intersection_squared_hellinger, jensen_shannon

from zipf_classifier import ZipfClassifier


def close_dict(a: dict, b: dict)->bool:
    for key, val in a.items():
        if key not in b:
            return False
        if not isclose(val, b[key], abs_tol=1e-15):
            return False
    return True


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
    expected_JS = {'success': 51, 'failures': 5, 'unclassified': 0,
                   'mean_delta': 0.08164044130948692, 'Mistook B for A': 5}
    expected_ISH = {'success': 55, 'failures': 1, 'unclassified': 0,
                    'mean_delta': 0.16704064864641854, 'Mistook B for A': 1}
    assert close_dict(expected_ISH, classifier.test(tests, intersection_squared_hellinger)
                      ) and close_dict(expected_JS, classifier.test(tests, jensen_shannon))
