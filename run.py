#!/usr/bin/env python

import pickle as pkl
import pandas as pd
import sys
import xgboost as xgb
import numpy as np



def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 

    `main` runs the targets in order of data=>analysis=>model.
    '''
    predictions_fp = 'predictions.csv'
    if 'test' in targets:
        text_data = pd.read_csv('test/text_test_data.csv')['Cleaned Text'].fillna('')
        non_text_data = pd.read_csv('test/non_text_test_data.csv')

        text_model = pkl.load(open('test/models/text_model.pkl', 'rb'))
        non_text_model = pkl.load(open('test/models/non_text_model.pkl', 'rb'))

        text_proba = text_model.predict_proba(text_data)
        non_text_proba = non_text_model.predict_proba(non_text_data)

        ensemble_proba = text_proba * 0.9 + non_text_proba * 0.1

        predictions = pd.Series(np.argmax(ensemble_proba, axis=1))

        predictions.to_csv(predictions_fp, index_label=False)

        return


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
