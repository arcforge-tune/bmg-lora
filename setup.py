from setuptools import setup, find_packages

setup(
    name='bmg-lora',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A scalable fine-tuning application for models on Intel Arc Hardware using IPEX.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'pyyaml',
        'scikit-learn',
        'loguru',
        'intel-ipex',
        'other-required-libraries'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)