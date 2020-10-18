import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ptycho", # Replace with your own username
    version="0.0.1",
    author="DeLicht",
    author_email="ukskosana@gmail.com",
    description="Seminar project on a few concepts ptychography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Unathi-Skosana/ptycho",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
