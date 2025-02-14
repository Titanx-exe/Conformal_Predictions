# Robust Entity Ranking
## installation:
Run pip install -r requirements.txt

## training/ evaluation
Run the script generic_training.py
for further settings see parameters.py
The evaluation scores are computed on the fly during training

## evaluation MS MARCO
Use the scrip eval_ms_marco_model.py

## noise
the implementation for the noise can be seen in the file optimizers/noise.py

## 
for loading the data the according datasets have to be downloaded from the according repository
lcquad2:https://github.com/AskNowQA/LC-QuAD2.0
mintaka: https://github.com/amazon-science/mintaka
for aida the files has to be in nif format:https://github.com/dice-group/gerbil, and for msmarco dataset:https://github.com/microsoft/MSMARCO-Document-Ranking
For preprocessing the MS MARCO dataset the script msmarco_preprocessing provides some code to generate th e required dictionaries and other files, used during training.
a more detailed description will be published with the repository.
