from Cython.Build import cythonize
from setuptools import Extension, find_namespace_packages, setup

ext_modules = cythonize(
    [
        Extension(
            "flare.data.pipeline.production.processing.native._scan",
            ["src/flare/data/pipeline/production/processing/native/_scan.pyx"],
            extra_compile_args=["-Ofast", "-march=native", "-funroll-loops"],
        ),
        Extension(
            "flare.data.pipeline.production.builder._fast_parse",
            ["src/flare/data/pipeline/production/builder/_fast_parse.pyx"],
            extra_compile_args=["-Ofast", "-march=native", "-funroll-loops"],
        ),
    ],
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    },
)

setup(
    name="flare",
    version="0+unreleased",
    description="Flare Machine Learning Framework",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["flare*"]),
    ext_modules=ext_modules,
    install_requires=[
        "boto3",
        "chex",
        "cython>=3.0",
        "flax",
        "jax",
        "msgpack>=1.0",
        "numba",
        "numpy>=1.22.0, <2.5",
        "optax",
        "orbax-checkpoint",
        "orjson",
        "pyarrow>=14.0",
        "pydantic",
        "pyiceberg",
        "ray[data]>=2.9",
        "s3fs",
        "scipy>=1.14",
        "tyro",
        "fluxway",
    ],
    extras_require={
        "gpu": [
            "cupy-cuda12x>=13.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio",
            "ruff>=0.16.1",
        ],
    },
    python_requires=">=3.12",
)
