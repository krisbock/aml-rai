# Training Responsible AI Models Using Azure ML

This repository demonstrates an example for how to train and evaluate models using the Azure ML CLI and Responsible AI dashboard.

The recommended demo flow is as follows:

1. Build a local conda environment using the `env.yml` file.
2. Test the training script in `src/train.py` locally to ensure that it works end-to-end.
3. Submit a remote pipeline job with the Azure ML CLI 2.0 using the specification in the `job.yml` file.
4. Navigate to the Studio UI to monitor the training results and view the generated Responsible AI dashboard
5. Realize that the model puts a high feature importance on fields that should not be relevant for this problem space.
6. Use the `data\data_process.ipynb` notebook to further analyze the data, drop irrelevant columns and register a new dataset.
7. Update the `job-update.yml` with the corresponding new dataset version and submit another job.
8. Go to the Studio UI to monitor the results again. Notice that accuracy became lower, but the model became less bias.
