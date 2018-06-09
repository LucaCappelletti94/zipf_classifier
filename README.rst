================
Zipf classifier
================

|travis| |coveralls| |sonar_quality| |sonar_maintainability| |code_climate_maintainability| |pip|

Introduction
-------------
ZipfClassifier is a classifier that, *even though in principle usable on any distribution*, leverages the assumption that some kind of datasets su as:

- texts
- `images (paper here)`_
- `spoken language (paper here)`_

follow the `Zipf law`_.

Dependecies
------------
ZipfClassifier uses `zipf`_, another package o' mine. I also suggest to use `dictances`_ for the metrics used in classification.

Installation
------------
.. code:: shell

    pip install zipf_classifier

Working examples and explanation
--------------------------------
A `jupyter notebook`_ is available with a full explanation, three working examples and respective link to datasets.

License
========
This package is licensed under MIT license.

FAQs
======
Frequenctly asked questions down below.

Generally, which metric do you suggest?
---------------------------------------
Experimental analysis suggests that, in particular when the **learning set** distributions contain a significant greater number of events than the distribution from the document you are trying to classify, the `intersection_squared_hellinger`_ seemed to work best.

.. |travis| image:: https://travis-ci.org/LucaCappelletti94/zipf.png
   :target: https://travis-ci.org/LucaCappelletti94/zipf_classifier

.. |coveralls| image:: https://coveralls.io/repos/github/LucaCappelletti94/zipf_classifier/badge.svg?branch=master
    :target: https://coveralls.io/github/LucaCappelletti94/zipf_classifier

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=zipf.lucacappelletti&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/zipf_classifier.lucacappelletti

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=zipf.lucacappelletti&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/zipf_classifier.lucacappelletti

.. |pip| image:: https://badge.fury.io/py/zipf_classifier.svg
    :target: https://badge.fury.io/py/zipf_classifier

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/c758496736a2c9cecbff/maintainability
   :target: https://codeclimate.com/github/LucaCappelletti94/zipf_classifier/maintainability
   :alt: Maintainability

.. _dictances: https://github.com/LucaCappelletti94/dictances
.. _zipf: https://github.com/LucaCappelletti94/zipf
.. _images (paper here): http://www.dcs.warwick.ac.uk/bmvc2007/proceedings/CD-ROM/papers/paper-288.pdf
.. _spoken language (paper here): http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0033993
.. _Zipf law: https://en.wikipedia.org/wiki/Zipf%27s_law
.. _jupyter notebook: https://github.com/LucaCappelletti94/zipf_classifier/blob/master/readme%20notebook%20full%20example%20and%20explanation.ipynb
.. _intersection_squared_hellinger: https://github.com/LucaCappelletti94/dictances/blob/master/dictances/intersection_squared_hellinger.py
