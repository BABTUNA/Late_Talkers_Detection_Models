# Multimodal Impariment Detection for Children 

This is the code for baseline machine learning and deep learning fusion model. 

### Dataset Background 
- this project uses a children late language impairment dataset, in which children are classified as normal talkers or impaired
- There are 19 children in total -> 12 late talkers and 7 typically developing
- The dataset in split into 3 different task for classification (bike, doctor, and retell)
- each dataset has 3 modalites extracted : Text, acoustic, and visual 

### Preprocessing Folder
- contains two version temporal and non-temporal preprocess: temporal considers all time stamps while non-temporal removes time dimesion by averaging values from all timestamps
- the data is tokenized for both version: tokenization is done to increase data sample sizes (chunk data from multiple children to produce more samples)

### ML folder
- contains baselines code for machine learning mode: logisitc regression, ada boost, random forest, decision tree
- ML models run on k fold stratify spliting rather than validation split due to limited data samples
- there is a script that will run all ML models at once

### DL folder
- contains multimodal approaches with different fusion methods: concate, multiply, add, self attention, and cross attention
- Each fusion method model contains different modalities for testing which modalities work best for different fusions
- the SH file is for running multiple hyperparameters settings: grid search method
- There is also BERT model variant which uses cross attention (that was found the best model) -> standard text embeddings is Text fast embeddings

### Final Note: there is more code for more models based off exisiting research papers, but that has not been included, the new model I'm working towards has also not been included

