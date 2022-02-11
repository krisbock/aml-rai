import argparse
import os
import shutil
import tempfile
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

def split_label(dataset, target_feature):
    X = dataset.drop([target_feature], axis=1)
    y = dataset[[target_feature]]
    return X, y

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='Path to training data')
    parser.add_argument('--target_column', type=str, help='Name of target column')
    parser.add_argument('--model_output', type=str, help='Path of output model')
    args = parser.parse_args()
    return args

def main(args):
    data = pd.read_parquet(args.train_data)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    target = args.target_column
    X_train, y_train = split_label(train_data, target)
    X_test, y_test = split_label(test_data, target)

    clf = LGBMClassifier(n_estimators=5)
    model = clf.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    mlflow.log_metric('Final accuracy', float(acc))
    print('Test accuracy: {}'.format(acc))

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
