# Disaster Response Pipeline Project

## Instalation

Look for the `environment.yml` file change last line

```yaml
prefix: /home/sgm/miniconda3/envs/data
```
to your own Anaconda instalation and env name and run command

```yaml
conda env create -f environment.yml
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
