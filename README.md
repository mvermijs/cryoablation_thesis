# Cryoablation thesis project
This repository contains the code used to calculate the results of my thesis project.

The code is mostly divided over a number of python scripts and notebook files. I've tried to add numerous comments and explanation in the code to make it more understandable. Please note that you might have to change the file locations in the script for the code to work, as I had to use locally stored files only. Every notebook should be able to run when executed from top to bottom.

For the notebook files:
-	Preprocessing script: Contains the code used to clean and encode the data
-	(Buthsch et al. - Transformer): Contains the code used to calculate the Transformer model results
- 	(Di Mauro et al. - CNN): Contains the code used to calculate the Inception CNN model results
-	(Galanti et al. - LSTM): Contains the code used to calculate the LSTM model results
-	(Pasquadibisceglie et al., 2020): Contains the code used to calculate the Orange CNN results
- 	AUK: Contains the code used to calculate the kappa and roc curves for a single encoding/model combination (for use as example in the thesis)
- 	base_model: Contains the code used to calculate the randomly weighted and simple (logistic and linear) model performance
- 	Di_mauro_et_al_encoder: Contains the code to calculate the embedding + time differences encoding, which had to be calculated seperately as it is part of a prediction model
- 	Results analysis: Contains the code used to analyze and plot the results
- 	shapley: contains the code used to calculate the shapley values

For the folders:
-	other_lib: Contains a number of python files with functions that are frequently used across notebooks
- 	orange_lib: Contains a number of functions used by the Orange CNN model.
- 	transoformer_lib: contains a number of functions used by the Transformer model.


This code was developed using the following versions:
- Python version 3.8.8
- Tensorflow version 2.8.0
- Hyperas version 0.4.1
