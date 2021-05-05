# Disaster Response Pipeline Project

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
