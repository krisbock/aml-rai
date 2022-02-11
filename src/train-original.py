import argparse
import os
import shutil
import tempfile
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='Path to training data')
    parser.add_argument('--target_column', type=str, help='Name of target column')
    parser.add_argument('--model_output', type=str, help='Path of output model')
    args = parser.parse_args()
    return args

def main(args):
    train_data = pd.read_parquet(args.train_data)
    y_train = train_data[args.target_column]
    X_train = train_data.drop(labels=args.target_column, axis='columns')

    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

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
