# Multi Layer Perceptron
***In this project I will implement a modular libray for feed forward deep learning models. Only numpy will be used.***

## How-To run it ?
You will need python 3.8 or more.
The following code ***MUST*** be run from the root of the project
tkinker must be installed on the system for the plots to function.
Setup environnement:   
```
python3.8 -m venv venv
source .env
pip install --upgrade pip
pip install -r requirements.txt
```
Train:
```python
python src/train.py
```
Test:
```python
python src/predict.py data/data.csv mymodel
```
