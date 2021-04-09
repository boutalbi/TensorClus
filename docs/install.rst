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

Install from GitHub repository
''''''''''''''''''''''''''''''

To clone TensorClus project from github::

  # Install git LFS via https://www.atlassian.com/git/tutorials/git-lfs
  # initialize Git LFS
  git lfs install Git LFS initialized.
  git init Initialized
  # clone the repository
  git clone https://github.com/boutalbi/TensorClus.git
  cd TensorClus
  # Install in editable mode with `-e` or, equivalently, `--editable`
  pip install -e .

.. note::  The latest TensorClus development sources are available on https://github.com/boutalbi/TensorClus


Running the tests
'''''''''''''''''

In order to run the tests, you have to install nose, for example with::

  pip install nose

You also have to get the datasets used for the tests::

  git clone https://github.com/boutalbi/TensorClus.git

And then, run the tests::

  cd cclust_package
  nosetests --with-coverage --cover-inclusive --cover-package=TensorClus
