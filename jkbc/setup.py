from setuptools import setup

setup(name='jkbc',
      version='0.0.2',
      description='Basecaller p10',
      url='https://github.com/Jgfrausing/basecaller-p10',
      author='Jonatan Groth Frausing, Kasper Dissing Bargsteen',
      author_email='kasper@bargsteen.com',
      license='MIT',
      packages=['jkbc'],
      install_requires=[
          'biopython',
          'fast_ctc_decode',
          'numpy',
          'h5py',
          'pytest',
      ],
      zip_safe=False)
