from setuptools import setup, Extension, find_packages


def readme():
    with open('README.md') as f:
        return f.read()



setup(name='TensorClus',
      version='0.0.2',
      description='TensorClus is a Python package for clustering of three-way tensor data',
      long_description=readme(),
      long_description_content_type='text/markdown',
      project_urls={
          "Bug Tracker": "https://github.com/boutalbi/TensorClus/issues",
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ],
      url='https://github.com/boutalbi/TensorClus',
      author='Rafika Boutalbi,Mohamed Nadif, Lazhar Labiod',
      author_email='boutalbi.rafika@gmail.com',
      keywords='Tensor Clustering framework',
      platforms=['*nix'],
      license='BSD 3-Clause License',
      packages=find_packages(),
      setup_requires=['pip>=19.0',"numpy==1.16.1",'pandas==1.0.3', 'scipy==1.2.1', 'scikit-learn==0.22.1','matplotlib==3.1.3', 'coclust'],
      install_requires=[
          'pip>=19.0','numpy==1.16.1','pandas==1.0.3', 'scipy==1.2.1', 'scikit-learn==0.22.1','matplotlib==3.1.3', 'coclust'
      ],
      extras_require={
        'alldeps': (
            'numpy==1.16.1',
            'pandas==1.0.3',
            'scipy==1.2.1',
            'scikit-learn==0.22.1',
            'matplotlib==3.1.3',
            'coclust',
            'tensorly'

        )
      },
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
