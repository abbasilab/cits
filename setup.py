import setuptools

with open('README.md','r') as fh:
    README = fh.read()

VERSION = "1.8.3"

setuptools.setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name = 'cits',
    version = VERSION,
    author = 'Rahul Biswas',
    description = 'CITS algorithm for inferring causality from time series data',
    long_description= README,
    long_description_content_type = 'text/markdown',
    # Core deps: base CITS, the CPU skeleton, GPU/Version-B wiring, and the
    # weighted graph (networkx) all run on Python + these. rpy2/R is NOT a
    # core requirement: it is needed only for the optional non-Gaussian HSIC
    # conditional-independence test (extras_require['hsic']). cuPC is a
    # compiled CUDA artifact (not on PyPI); see the README GPU setup section.
    install_requires=['numpy','scipy','pandas','networkx',],
    extras_require={
        # Non-Gaussian HSIC CI test via R's kpcalg (also requires an R
        # install with the kpcalg package; see README).
        'hsic': ['rpy2'],
        # Plotting helpers (cits.plot_graph / cits.plot_matrix). networkx is
        # already a core dependency; matplotlib is only needed for plotting.
        'viz': ['matplotlib'],
    },
    url='https://github.com/biswasr/CITS',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',  # 'from __future__ import annotations' requires 3.7+
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
