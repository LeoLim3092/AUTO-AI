from setuptools import setup, find_packages


setup(
    name='AUTOAI',
    version='1.0',
    packages=find_packages(),
    description='An automatic porgram for training ML models with sklearn package.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Leo',
    author_email='limweeshin1991@gamil.com',
    url='',
    install_requires=[
        'scikit-learn==1.1.3',
        'xgboost==1.7.4',
        'lightgbm==3.4.0',
        'mlxtend==0.22.0',
        'joblib==1.2.0',
        'numpy==1.25.0',
        'matplotlib==3.7.0',
        'seaborn==0.12.0',
        'pandas==1.6.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points={
        'console_scripts': [
            # For example: 'scriptname = packagename.module:function'
        ],
    },
    include_package_data=True,
    package_data={
        # Include any necessary files
        '': ['*.txt', '*.rst', '*.md'],
        'hello': ['*.msg'],  # Adjust as necessary for your package structure
    },
)
