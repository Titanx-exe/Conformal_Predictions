# Robust Entity Ranking
## installation:
Run pip install -r requirements.txt

## training
Run the script generic_training.py
uncomment the according lines for the according datasets and models
for further settings see parameters.py

## evaluation
Run the script eval_existing.py
uncomment the according lines for the according datasets and models
for further settings see parameters.py

## noise
the implementation for the noise can be seen in the file optimizers/noise.py

## 
for loading the data the according datasets have to be downloaded from the according repository
lcquad2:https://github.com/AskNowQA/LC-QuAD2.0
mintaka: https://github.com/amazon-science/mintaka
for aida the files has to be in nif format:https://github.com/dice-group/gerbil

a more detailed description will be published with the repository.
