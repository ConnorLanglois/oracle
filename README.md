<div align="center">
  <h3><a href="#">🔮</a></h3>
</div>
<h1 align="center">Oracle™</h1>
<h3 align="center">An automated AI-backed stock prediction application.</h3>
<div align="center">
	<a href="https://github.com/ConnorLanglois/oracle/issues">Report Bug</a>
	·
	<a href="https://github.com/ConnorLanglois/oracle/issues">Request Feature</a>
</div>
<br>

## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Copyright](#copyright)

## About The Project

Oracle™ is an application built to predict the movement of the stock market.

By leveraging state-of-the-art AI LSTM neural networks and vasts amount of data, Oracle™ learns which stock features signal a rise or fall in price.

Currently, it has about a 60% accuracy in predicting the daily movement of a stock.

Read the academic research paper [here](paper/An%20AI-based%20Approach%20to%20Stock%20Market%20Prediction.pdf)

### Built With

* [Python](https://www.python.org/) - The scripting language
* [Anaconda](https://www.anaconda.com) - The scientific package management and deployment library
* [Tensorflow](https://www.tensorflow.org) - The machine learning library
* [Keras](https://keras.io) - The high-level wrapper library over Tensorflow
* [Numpy](https://numpy.org) - The fast numeric library
* [Pandas](https://pandas.pydata.org) - The data frame library
* [Scikit-learn](https://scikit-learn.org) - Another machine learning library

## Getting Started

Follow these simple example steps to get a local copy up and running.

### Prerequisites

* Anaconda
	```
	https://www.anaconda.com/distribution/#download-section
	```

### Installation

On Windows, open Anaconda Prompt

On Mac/Linux, open terminal

1. Create the Oracle™ environment
	```shell
	conda create -n oracle
	```

2. Activate the Oracle™ environment
	```shell
	conda activate oracle
	```

3. Install Tensorflow (also installs Python and Numpy)
	* CPU (if no dedicated GPU)
		```shell
		conda install tensorflow
		```
	* GPU
		```shell
		conda install tensorflow-gpu
		```

6. Install Keras
	```shell
	conda install keras
	```

5. Install Pandas
	```shell
	conda install pandas
	```

5. Install Scikit-learn
	```shell
	conda install scikit-learn
	```

4. Clone the repo
	```shell
	git clone https://github.com/ConnorLanglois/oracle.git
	```

## Usage

On Windows, open Anaconda Prompt

On Mac/Linux, open terminal

1. Activate the Oracle™ environment
	```shell
	conda activate oracle
	```

2. Navigate to [src](src)
	```shell
	cd ~/.../oracle/src
	```

2. Run the main file:
	```shell
	python oracle.py
	```

## License

This project is intentionally not licensed.

## Copyright

Copyright © 2019 Connor Langlois. All rights reserved.
