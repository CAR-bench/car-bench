from setuptools import find_packages, setup

setup(
    name="car_bench",
    version="0.1.0",
    description="The Car-Bench package",
    long_description=open("README.md").read(),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "annotated_types>=0.7.0",
        "datasets>=3.0.0",
        "huggingface_hub>=0.25.0",
        "litellm>=1.80.16",
        "matplotlib>=3.10.5",
        "mcp>=1.0.0",
        "networkx>=3.4.2",
        "numpy>=2.3.2",
        "pandas>=2.3.1",
        "pydantic>=2.11.7",
        "pyvis>=0.3.2",
        "requests>=2.32.4",
        "seaborn>=0.13.2",
        "setuptools>=65.5.0",
        "tiktoken>=0.8.0",
        "tqdm>=4.67.1",
    ],
)
