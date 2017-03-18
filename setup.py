"""
sorry, ironhead can not remember everything.

    python setup.py sdist
    twine upload dist/ore-0.0.1.tar.gz
    twine upload -r test dist/ore-0.0.1.tar.gz
"""
from setuptools import setup


setup(
    name='ore',
    version='0.0.2',
    url='https://github.com/imironhead/ml_ore',
    description='ask me later',
    long_description='ask me later please',
    author='ironhead',
    # author_email='',
    license='MIT',
    packages=['ore'],
    classifiers=[
        'Programming Language :: Python :: 2.7'
    ],
    install_requires=['lmdb', 'numpy', 'pillow', 'scipy', 'six']
)
