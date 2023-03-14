#!/usr/bin/env python

import pickle as pkl
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 

    `main` runs the targets in order of data=>analysis=>model.
    '''
    predictions_fp = 'predictions.csv'
    if 'test' in targets:
        data = pd.read_csv('test/testdata.csv')


        X_test = data['Cleaned Text']
        y_test = data['new_category']

        text_model = pkl.load(open('src/data/text_model.pkl', 'rb'))

        text_proba = pd.Series(pklmodel.predict_proba(X_test))



        predictions.to_csv(predictions_fp, index_label=False)

        return


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
