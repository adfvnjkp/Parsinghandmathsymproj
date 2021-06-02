## Files

- main.py: use main.py to get .lg files from a folder containing .inkml files using trained model from folder checkpoints
- model.py: most important model, the multi-dense encoder and decoder model
- dataset.py: defination of CROHME data
- train.py: train your neural network model from sracth. Change default parameters in code or use '--name parameter'
- checkpoint.py: a tool for saving and loading checkpoints
- ./datatools: some library to convert inkml file to png and generate groundtruth
- ./chechpoint: saved nn model
- ./tensorboard: tensorboard log
- ./data: put training dataset here and also groundtruth

