from setuptools import find_packages, setup

setup(
    name='pyIEEM',
    packages=find_packages("src", exclude=["*.tests"]),
    package_dir={'': 'src'},
    version='0.0',
    description='Integrated Epidemiologic Economic Model in Python 3',
    author='Tijs Alleman, BIOSPACE, Ghent University',
    license='MIT',
    install_requires=[
        'pySODM',
        'pandas',
        'numpy',
        'matplotlib'
    ],
    extras_require={
        "develop":  ["pytest",
                     "sphinx",
                     "numpydoc",
                     "sphinx_rtd_theme",
                     "myst_parser[sphinx]"],
    }
)