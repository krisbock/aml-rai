# Training Responsible AI Models Using Azure ML

This repository demonstrates an example for how to train and evaluate models using the Azure ML CLI and Responsible AI dashboard.

The recommended demo flow is as follows:
```
az configure --defaults group=$rgName workspace=$wsName
```
1. Build a local conda environment using the `env.yml` file.
```
conda env create --file env.yml
conda activate rai_env
```
2. Test the training script in `src/train.py` locally to ensure that it works end-to-end.
```
mkdir ./outputs
python src/train.py --train_data data/original/train --target_column income --model_output ./outputs
```
3. Register the dataset using the AzureML CLI.
```
az ml data create --name Adult_Train_MLT --version 1 --path ./data/original/train --type mltable 
az ml data create --name Adult_Test_MLT --version 1 --path ./data/original/test --type mltable 
```
4. Submit a remote pipeline job with the Azure ML CLI 2.0 using the specification in the `job.yml` file.
```
az ml job create --file job.yml
```
5. Navigate to the Studio UI to monitor the training results and view the generated Responsible AI dashboard
6. Realize that the model puts a high feature importance on fields that should not be relevant for this problem space.
7. Use the `data\data_process.ipynb` notebook to further analyze the data, drop irrelevant columns and register a new dataset.
8. Register new datasets
```
az ml data create --name Adult_Train_MLT --version 2 --path ./data/updated/train --type mltable 
az ml data create --name Adult_Test_MLT --version 2 --path ./data/updated/test --type mltable 
```
9. Update the `job-update.yml` with the corresponding new dataset version and submit another job.

10. Go to the Studio UI to monitor the results again. Notice that accuracy became lower, but the model became less bias.
