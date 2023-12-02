from setuptools import setup
from setuptools import find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='nav2d',
    packages=find_packages(),
    version='0.0.1',
    license='MIT License',
    description='2D Navigation Gym environment for Reinforcement Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Xiaoyu Wang',
    author_email='cnxiaoyu.wang@mail.utoronto.ca',
    url='https://github.com/xxArbiter/nav2d',
    download_url="https://github.com/xxArbiter/nav2d/archive/refs/tags/0.0.1.tar.gz"
)