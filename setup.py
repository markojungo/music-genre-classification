from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'deepdish>=0.3.6',
    'cloudml-hypertune>=0.1.0.dev6',
    'google-cloud-storage>=1.14.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='AI Platform | Training | PyTorch | Music Genre Classification Model'
)