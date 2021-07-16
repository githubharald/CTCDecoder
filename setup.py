from setuptools import setup

setup(
    name='ctc-decoder',
    version='1.0.0',
    description='Connectionist Temporal Classification decoders.',
    author='Harald Scheidl',
    packages=['ctc_decoder'],
    url="https://github.com/githubharald/CTCDecoder",
    install_requires=['editdistance', 'numpy'],
    python_requires=">=3.6"
)
