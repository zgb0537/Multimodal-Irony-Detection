from setuptools import setup

setup(name = 'nlp_features',
    version = '0.1',
    description = 'Calculate different textual features for Tumblr posts.',
    author = 'David Tomás',
    author_email = 'dtomas@dlsi.ua.es',
    packages = ['nlp_features'],
    install_requires = ['numpy>=1.15.1', 'spacy>=2.1.8', 'spacymoji>=2.0.0'])
