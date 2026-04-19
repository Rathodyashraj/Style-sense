

from setuptools import setup, find_packages

setup(
    name            = "outfit_compatibility",
    version         = "1.0.0",
    description     = (
        "Graph-Based Stylistic Compatibility and Harmony Scorer "
        "using the Polyvore Outfits dataset."
    ),
    author          = "Outfit Compatibility Scorer",
    python_requires = ">=3.9",

    # Discover all sub-packages under src/
    packages        = find_packages(where="."),

    # Install the CLI scripts as console entry-points
    entry_points    = {
        "console_scripts": [
            "oc-extract  = scripts.extract_features:main",
            "oc-train    = scripts.train:main",
            "oc-evaluate = scripts.evaluate:main",
            "oc-infer    = scripts.infer:main",
        ],
    },

    # Runtime dependencies — kept in sync with requirements.txt
    install_requires = [
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "open-clip-torch>=2.23.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "h5py>=3.9.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
    ],

    extras_require = {
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "notebook>=7.0.0",
        ],
    },

    classifiers = [
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
