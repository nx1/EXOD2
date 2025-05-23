from setuptools import setup, find_packages

setup(
    name='exod',
    version='0.1.0',
    description='X-ray Outburst Detector',
    authors=['Norman Khan', 'Erwan Quintin'],
    author_emails=['Norman.Khan@irap.omp.eu', 'Erwan.Quintin@irap.omp.eu'],
    url='https://github.com/nx1/EXOD2',
    packages=find_packages(),  # Automatically discover and include all packages in the project

    # Define project dependencies
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'astropy',
        'requests',
        'tqdm',
        'scikit-image',
        'pandas',
        'cmasher',
        'statsmodels',
        'opencv-python',
        'photutils',
        'astroquery',
        'pytest',
        'Flask',
        'tables'
    ],

    # Include additional files like data files, package data, etc.
    include_package_data=True,

    # Entry points allow scripts to be specified as executables
    entry_points={
        'console_scripts': [
            'xod-cli = xod.cli:main',  # Example entry point for a command-line script
        ],
    },

    # Define classifiers to categorize your project
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)

