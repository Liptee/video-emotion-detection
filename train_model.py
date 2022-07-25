import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle

from sklearn.linear_model import (
    LogisticRegression, 
    RidgeClassifier
    )

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
    )

df = pd.read_csv('data/geusture.csv')

X = df.drop('class', axis=1)
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=124)

#pipelines
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model_file_name = f'models/model_geusture_{algo}.pkl'
    model = pipeline.fit(x_train, y_train)
    with open(model_file_name, 'wb') as f:
        pickle.dump(model, f)
