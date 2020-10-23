import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dynamicopy",
    version="0.1",
    author="Stella Bourdin",
    author_email="stella.bourdin@lsce.ipsl.fr",
    description="A set of tool to use and analyse netCDF data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # TODO : Upload to Github when done
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
