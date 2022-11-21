import argparse
import os
import shutil
import tempfile
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mltable
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='Path to training data')
    parser.add_argument('--target_column', type=str, help='Name of target column')
    parser.add_argument('--model_output', type=str, help='Path of output model')
    args = parser.parse_args()
    return args

def main(args):

    print('Loading data ...')
    #data = pd.read_parquet(args.train_data)
    data = mltable.load(args.train_data).to_pandas_dataframe()
    target_column = args.target_column
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    y_train = train_data[target_column]
    X_train = train_data.drop(labels=target_column, axis='columns')
    y_test = test_data[target_column]
    x_test = test_data.drop(labels=target_column, axis='columns')

    print('Training model ...')
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    mlflow.log_metric('Final accuracy', float(acc))
    print('Test accuracy: {}'.format(acc))

    print('Saving output ...')
    with tempfile.TemporaryDirectory() as td:
        tmp_output_dir = os.path.join(td, 'my_model_dir')
        mlflow.sklearn.save_model(sk_model=model, path=tmp_output_dir)
        for file_name in os.listdir(tmp_output_dir):
            shutil.copy2(
                src=os.path.join(tmp_output_dir, file_name),
                dst=os.path.join(args.model_output, file_name),
            )

if __name__ == '__main__':
    args = parse_args()
    main(args)
