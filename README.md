# DroughtResearchML

<p>Author: Tom Qiu <br>Credits: DroughtED (Christoph Minixhofer, Mark Swan, Calum McMeekin, Pavlos Andreadis)</p>

## Purpose

The CNN and LSTM models in the repository predict drought based on 21 meteorological and 30 soil features, with the dataset from DroughtED. Specifically, the LSTM code came from the creators of DroughtED, but are modified by Qiu to conduct feature ablation. The purpose of the feature ablation is to reduce the size of the input and increase efficiency and accuracy of the model. The CNN model is built by Qiu and is used to add another layer to the feature ablation conducted.

## Usage

There are several pieces of code in this repository. First, ```Heatmap.py``` creates a heatmap of the US based on the values of the USDM category the drought in a particular region is categorized to be. ```CNN.py``` and ```LSTM.py``` is the source code for the models. ```Results-Processing``` contains code to process the output from the models (the predictions). The data in ```Data``` is already preprocessed, according to the method of pre-processing conducted by the creators of DroughtED. The pre-processed data, which is scraped off the internet and processed by the team of DroughtED, can be accessed here: <https://drive.google.com/drive/folders/1zEIRk3ZLCqw_as2bLsFRukiJpR2PBQv1?usp=drive_link>. The heatmap data can be found here: <https://drive.google.com/drive/folders/1vBsnma-e0zBmWDyutkW418gFmy3XoqdQ?usp=drive_link>. The results processing code can be run when the output of the model is compiled into a csv file.
