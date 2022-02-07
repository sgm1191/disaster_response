# Disaster Response Pipeline Project

##Agregué esta línea pa probar la pull requestación.

This project is made as an example of implementation of a ML pipeline. The application shows some stats about disaster tweets dataset and can classify a message into one of 36 categories. This may be useful to monitor twitter in real time and provide aid to those who need it asap.
This application was made during a data science course on [Udacity](https://www.udacity.com/).

The ML pipeline applies some NLP techniques, and MLP neural networks to make the classification, python is used to serve it and bootsrap for and plotly for the visuals.

## Instalation

Look for the `environment.yml` file change last line

```yaml
prefix: /home/sgm/miniconda3/envs/data
```
to your own Anaconda instalation and env name and run command

```bash
conda env create -f environment.yml
```

or you can install with pip as follows

```bash
pip install -f requirements.txt
```

### Model
The model is not in this repo due to size restriction, instead you can download it [here](https://drive.google.com/file/d/1BVuCBuQdt449Ljh7hXNeB_NFzeY0HZg5/view?usp=sharing). The file is called `classifier.pkl` and you should place it in `models` dir.

## Project Structure

    ├── app
    │   ├── __init__.py
    │   ├── run.py
    │   └── templates
    │       ├── go.html
    │       └── master.html
    ├── data
    │   ├── disaster_categories.csv
    │   ├── disaster_messages.csv
    │   ├── DisasterResponse.db
    │   ├── __init__.py
    │   └── process_data.py
    ├── environment.yml
    ├── LICENSE
    ├── models
    │   ├── classifier.pkl
    │   ├── __init__.py
    │   └── train_classifier.py
    ├── notebooks
    │   ├── ML_pipeline_demo.ipynb
    │   └── process_data_demo.ipynb
    ├── README.md
    └── requirements.txt

- *app*: scripts to run the web server of the disaster response message page
- *app/templates*: front-end html pages for the web app
- *data*: you can find datasets here and a script to clean and process the data into a database
- *models*: the trained model dowloaded should be placed here, or you can train your own with the file `train_clasiffier.py` in this directory
- *notebooks*: there are some jupyter notebooks used to explore the data cleaning process and the machine learning pipeline process
- *environment.yml*, *requirements.txt*: files with the dependencies necessary to run the scripts in this repository, they are for anaconda and pip respectively.
- *README.md*: this file containing a summary of the project.

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves*
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    *Note: modify the parameters in `models/train_classifier.py` to reduce number of parameters as it can take a looong time to train.

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
