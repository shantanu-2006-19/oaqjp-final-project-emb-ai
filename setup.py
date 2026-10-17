from setuptools import setup, find_packages

setup(
    name="EmotionDetection",
    version="0.1.0",
    description="Final project: emotion detection example package",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "flask",
        "ibm-watson",
        "ibm-cloud-sdk-core",
    ],
)
