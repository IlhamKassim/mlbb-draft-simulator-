from setuptools import setup, find_packages

setup(
    name="mlbb_counter_system",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask",
        "requests",
        "werkzeug",
    ],
) 