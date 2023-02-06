Predicting phone price based on processing text from trade offers

Project structure:
src - working files (jupyter notebooks and .py function files)
data - csv files with scrapped and processed data
models - machine learning models files

Chronological order of working files in src:
1. data_standardization.ipynb
2. data_preprocessing.ipynb
3. auto_model.ipynb
4. data_visualization.ipynb

Code from those notebooks is distilled into functions in util.py and classes.py

Part 1 - data standardizing 
Data supplied along the task (recruitment_task.csv) had samples mostly in 2 general forms:
1. Samples with features separated by a comma with large text features such as "Name" and "Description" having quotes inside of them 
    as a "quotechar", which forbids data processing tools to treat commas inside those texts as separators.
2. Samples with features separated by a comma, but also enclosed in a quote character from start to finish
3. Outlier samples that do not belong to first 2 groups (around 100 samples, discarded)
Having those 2 differently formatted sample groups I needed to standardize them into one format - which is the first one.
is done in (data_standardization.ipynb), also moving the finished function to (util.py).

Part 2 - data analysis and preprocessing
After tackling the problem of standardizing data I have started to explore and transform the data according to the 
problem specification (data_preprocessing.ipynb):
1. At first I check visually if data loads properly after standardizing
2. I study how many empty fields are in the data and also I look at different values and their
   respective count in given column
3. The first preprocessing step is dropping all samples where 'Price' is NaN, because those
   are useless in this problem
4. I filter the data to only have samples with Condition=='Używany', Type=='Sprawny', 
   and Brand=='iPhone', according to the specification of the problem. The dataset contains approximately 2/3 of 'Używane'
   samples, so this is the biggest sample cut off. Type and Brand filters only affect no more than few hundred samples
5. At this point I decide to drop all the columns that do not carry important information according 
   to the specification of the problem, e.g. 'Voivodeship' or 'URL'
6. It is also important to cut off samples that do not offer actual product, but an accessory or a service,
    such as phone case or a repair service - I apply this by dropping all samples below value 1000 in 'Price'
   This is probably not the best way to do this, but I assumed that any working iPhone 11 model wouldn't be offered
   for such a low price
7. Converting all strings in the data to lowercase
8. Creating new column that counts how many days passed since earliest date in the data
9. Concatenating few columns into one, in order to pass only 1 column to the model later on (to tweak)

Both notebooks resulted in two functions (described in detail in util.py) used in latter notebooks. 
Data preprocessing could be more generalized and parametrized.

Part 3 - NLP model
Having processed data in form of concatenated text used to feed the model in one column and target price in the other
I proceed to build a basic model with pretrained polish language model 'dkleczek/bert-base-polish-uncased-v1' 
from Hugging Face transformers library, which is based on BERT model from Google. 

Preparation of the data for the model consists of several parts:
1. Scaling 'Price' values to range (-1, 1)
2. Splitting the dataset into training, validation and test groups
3. Tokenization of each of the sets
4. Converting tokens into torch-type dataset along with the labels

Torch-type dataset is the entry to training and validation phase of the process
During training (and at the end of it) at every epoch model metrics and parameters are saved.
This allows us to later load the model and performs predictions with it.

Part 4 - Data visualisation 
data_visualization.ipynb contains both changes in metric values and possible predictions for any model trained
before on the dataset. Metrics used to measure performance of the model are (compute_metrics in util.py): 
1. mean square error
2. root mean square error
3. mean absolute error
4. r2 score

WIP User interface in streamlit WIP









