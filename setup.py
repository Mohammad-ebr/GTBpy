from setuptools import setup, find_packages
from os import path
working_directory = path.abspath(path.dirname(__file__))

with open(path.join(working_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='GTBpy',
    version='0.0.1',
    # url='https://github.com/yourname/yourproject',
    author='Mohammad Ebrahimi',
    author_email='mohammad.ebrahimi.gtbpy@gmail.com',
    description='Search index from Google Trends and bubble tests',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'os',
        'numpy',
        'pandas',
        'numba',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'sklearn',
        'scipy',
        'math',
        'ast',
        'IPython',
    ]
)