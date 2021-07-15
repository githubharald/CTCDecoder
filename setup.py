from setuptools import setup

setup(
    name='ctc_decoder',
    version='1.0.0',
    description='Connectionist Temporal Classification decoders.',
    author='Harald Scheidl',
    packages=['ctc_decoder'],
    install_requires=['editdistance', 'numpy']
)
