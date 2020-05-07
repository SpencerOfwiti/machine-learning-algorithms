# Machine Learning Algorithms

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) 
![GitHub repo size](https://img.shields.io/github/repo-size/SpencerOfwiti/machine-learning-algorithms.svg)
![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
[![contributors](https://img.shields.io/github/contributors/SpencerOfwiti/machine-learning-algorithms.svg)](https://github.com/SpencerOfwiti/machine-learning-algorithms/contributors)

Implementation of common Machine Learning algorithms and Neural Networks on a variety of datasets and use cases using Scikit-Learn, TensorFlow and PyTorch.

## Table of contents
* [Build Status](#build-status)
* [Built With](#built-with)
* [Features](#features)
* [Code Example](#code-example)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Contributions](#contributions)
* [Bug / Feature Request](#bug--feature-request)
* [Authors](#authors)
* [License](#license)
* [Acknowledgements](#acknowledgments)

## Build Status

[![Build Status](https://travis-ci.com/SpencerOfwiti/machine-learning-algorithms.svg?branch=master)](https://travis-ci.com/SpencerOfwiti/machine-learning-algorithms)

## Built With
* [Python 3.6](https://www.python.org/) - The programming language used.
* [SciKit Learn](https://scikit-learn.org/stable/) - The machine learning library used.
* [TensorFlow](https://www.tensorflow.org/) - The machine learning platform used.
* [PyTorch](https://pytorch.org/) - The machine learning framework used.
* [Travis CI](https://travis-ci.com/) - CI-CD tool used.

## Features

- Supervised Learning
- Unsupervised Learning
- Neural Networks
- Reinforcement Learning
- Recommender Systems
- Natural Language Processing
- Hyperparameter Tuning

## Code Example

```python
# Converting a tf.Keras model to a TensorFlow Lite model.
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

## Prerequisites

What things you need to install the software and how to install them

* **python 3**

Linux:
```
sudo apt-get install python3.6
```

Windows:

Download from [python.org](https://www.python.org/downloads/windows/) 

Mac OS:
```
brew install python3
```

* **pip**

Linux and Mac OS:
```
pip install -U pip
```

Windows:
```
python -m pip install -U pip
```

## Installation

Clone this repository:
```
git clone https://github.com/SpencerOfwiti/machine-learning-algorithms
```

To set up virtual environment and install dependencies:
```
source setup.sh
```

To run python scripts:
```
# Convert model to TensorFlow lite
python3 Neural\ Networks/tflite_converter.py
```

## Contributions

To contribute, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).


## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/SpencerOfwiti/machine-learning-algorithms/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/SpencerOfwiti/machine-learning-algorithms/issues/new). Please include sample queries and their corresponding results.

## Authors

* **[Spencer Ofwiti](https://github.com/SpencerOfwiti)** - *Initial work* 
    
[![github follow](https://img.shields.io/github/followers/SpencerOfwiti?label=Follow_on_GitHub)](https://github.com/SpencerOfwiti)
[![twitter follow](https://img.shields.io/twitter/follow/SpencerOfwiti?style=social)](https://twitter.com/SpencerOfwiti)

See also the list of [contributors](https://github.com/SpencerOfwiti/machine-learning-algorithms/contributors) who participated in this project.

## License

This project is licensed under the MIT license - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Crash Course AI](https://www.youtube.com/playlist?list=PL8dPuuaLjXtO65LeD2p4_Sb5XQ51par_b) from inspiration behind some of the projects.
