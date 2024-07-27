 from setuptools import setup, find_packages
 from pathlib import Path

# Read the contents of requirements.txt
requirements = Path('requirements.txt').read_text().splitlines()


setup(
    name="DCD-MUSIC",
    version="1.0.0",
    author="Arad Gast",
    author_email="aradgast1@gmail.com",
    description="The repository for source code of the DCD-MUSIC",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aradgast/DCD-MUSIC",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    },
)
