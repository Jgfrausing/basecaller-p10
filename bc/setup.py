from setuptools import setup

setup(name='bc',
      version='0.1',
      description='Basecaller p10',
      url='https://github.com/Jgfrausing/basecaller-p10',
      author='Jonatan Groth Frausing, Kasper Dissing Bargsteen',
      author_email='kasper@bargsteen.com',
      license='MIT',
      packages=['bc'],
      install_requires=[
          'biopython',
      ],
      zip_safe=False)
