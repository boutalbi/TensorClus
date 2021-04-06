from setuptools import setup, Extension


def readme():
    with open('README.md') as f:
        return f.read()



setup(name='TensorClus',
      version='0.0.1',
      description='TensorClus is a Python package for clustering of three-way tensor data',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: GNU License',
          'Programming Language :: Python :: 3.7',
          'Operating System :: OS Independent',
          'Framework :: AsyncIO',
          'Topic :: Scientific/Engineering :: Information Analysis'
      ],
      url='https://github.com/boutalbi/TensorClus',
      author='Rafika Boutalbi,Mohamed Nadif, Lazhar Labiod',
      author_email='boutalbi.rafika@gmail.com',
      keywords='Tensor Clustering framework',
      platforms=['*nix'],
      license='GNU',
      packages=['TensorClus'
                ],
      setup_requires=["numpy==1.19.2", 'scipy', 'scikit-learn','matplotlib', 'coclust'],
      install_requires=[
          'numpy==1.19.2', 'scipy', 'scikit-learn','matplotlib', 'coclust'
      ],
      extras_require={
        'alldeps': (
            'numpy==1.19.2',
            'scipy',
            'scikit-learn',
            'matplotlib',
            'coclust',
            'tensorly'

        )
      },
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
