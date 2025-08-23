from setuptools import setup, find_packages
from pathlib import Path

DESCRIPTION = "Fx-Encoder++ for audio effects representation"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name="fxencoder_plusplus",
    version="0.1.5",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "laion-clap",
        "torchlibrosa",
        "huggingface_hub"
    ],
    author="ytsrt66589",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SonyResearch/Fx-Encoder_PlusPlus",
    author_email='ytsrt66589@gmail.com', 
    license='CC BY-NC 4.0'
)

