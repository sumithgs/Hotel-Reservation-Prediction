from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="hotel_reserv",
    version="0.0.1",
    author="Sumith",
    packages=find_packages(),
    install_requires=requirements,
)