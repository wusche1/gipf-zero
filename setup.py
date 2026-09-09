from setuptools import Extension, setup
import pybind11

setup(
    name="gipf-engine",
    version="0.1.0",
    ext_modules=[
        Extension(
            "gipf_engine",
            ["engine/gipf_engine.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O3"],
        )
    ],
)
