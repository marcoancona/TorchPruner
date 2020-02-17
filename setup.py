from setuptools import setup, find_packages

setup(name='torchpruner',
      version='0.1',
      description='On-the-fly Structured Pruning for PyTorch',
      url='https://github.com/marcoancona/TorchPruner',
      author='Marco Ancona (ETH Zurich)',
      author_email='marco.ancona@inf.ethz.ch',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'scipy',
      ],
      extras_require={
            "torch": ["torch>=1.2.0"]
      },
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )