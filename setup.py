from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("line_graph",
                      ["line_graph.cpp"]),
]

setup(
    name="example",
    version="0.0.1",
    author="Isaac Peterson",
    description="A small example project",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
