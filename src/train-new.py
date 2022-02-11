import argparse
import os
import shutil
import tempfile
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

def split_label(dataset, target_feature):
    X = dataset.drop([target_feature], axis=1)
    y = dataset[[target_feature]]
    return X, y

def clean_data(X, y, target_feature):
    features = X.columns.values.tolist()
    classes = y[target_feature].unique().tolist()
    pipe_cfg = {
        'num_cols': X.dtypes[X.dtypes == 'int64'].index.values.tolist(),
        'cat_cols': X.dtypes[X.dtypes == 'object'].index.values.tolist(),
    }
    num_pipe = Pipeline([
        ('num_imputer', SimpleImputer(strategy='median')),
        ('num_scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value='?')),
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    feat_pipe = ColumnTransformer([
        ('num_pipe', num_pipe, pipe_cfg['num_cols']),
        ('cat_pipe', cat_pipe, pipe_cfg['cat_cols'])
    ])
    X = feat_pipe.fit_transform(X)
    print(pipe_cfg['cat_cols'])
    return X, feat_pipe, features, classes
    
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

    target_feature = args.target_column

    X_train_original, y_train = split_label(train_data, target_feature)
    X_test_original, y_test = split_label(test_data, target_feature)


    X_train, feat_pipe, features, classes = clean_data(X_train_original, y_train, target_feature)
    y_train = y_train[target_feature].to_numpy()

    X_test = feat_pipe.transform(X_test_original)
    y_test = y_test[target_feature].to_numpy()

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
