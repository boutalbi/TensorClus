Installation
============

You can install **TensorClus** with all the dependencies with::

    pip install TensorClus

It will install the following libraries:

    - numpy
    - pandas
    - scipy
    - scikit-learn
    - matplotlib
    - coclust
    - tensorly
    - tensorflow


Running the tests
'''''''''''''''''

In order to run the tests, you have to install nose, for example with::

  pip install nose

You also have to get the datasets used for the tests::

  git clone https://github.com/boutalbi/TensorClus.git

And then, run the tests::

  cd cclust_package
  nosetests --with-coverage --cover-inclusive --cover-package=TensorClus
