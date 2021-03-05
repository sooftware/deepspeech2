from distutils.core import setup

setup(
    name='deepspeech2',
    version='latest',
    author='Soohwan Kim',
    author_email='sh951011@gmail.com',
    url='https://github.com/sooftware/conformer',
    install_requires=[
        'torch>=1.4.0',
        'numpy',
    ],
    keywords=['asr', 'speech_recognition', 'conformer', 'end-to-end'],
    python_requires='>=3.7'
)
