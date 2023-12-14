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
        'numpy=>1.26.2',
        'scipy=>1.11.4',
        'matplotlib=>3.8.2',
        'astropy=>6.0.0',
        'requests=>2.31.0',
        'tqdm=>4.66.1',
        'scikit-image=>0.22.0',
        'pandas=>2.1.4',
        'cmasher=>1.6.3',
        'statsmodels=>0.14.0',
        'opencv-python=>4.8.1.78',
        'photutils-1.10.0',
        'astroquery=>0.4.6'
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

