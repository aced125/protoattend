#!/usr/bin/env python

"""The setup script."""
# fmt: off
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0',
    'tqdm',
    'pyyaml',
    'torch',
    'pytorch-lightning',
    'sparsemax',
    'numpy',
    # 'torchvision',
    # 'pillow',
    # 'pandas',
    # 'matplotlib',
    # 'seaborn'
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Laksh Aithani",
    author_email='lakshaithanii@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Protoattend library for interpretable machine learning",
    entry_points={
        'console_scripts': [
            'protoattend=protoattend.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='protoattend',
    name='protoattend',
    packages=find_packages(include=['protoattend', 'protoattend.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/aced125/protoattend',
    version='0.1.3',
    zip_safe=False,
)
# fmt: on
