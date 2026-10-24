## Dataset Files

The full CICIDS2017 dataset and generated train/test splits are not
included in this repository because of their file size.

Cleaned dataset can be found in the link below:
[CICIDS2017 Cleaned and Preprocessed](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed)

The project-specific dataset files, including the further-cleaned Version 3 dataset, data-quality audit records with V2, and 70/15/15 stratified train/validation/test splits of V3, can be found here: [NOVA Dataset Files](https://paceuniversity-my.sharepoint.com/:f:/r/personal/kv56727n_pace_edu/Documents/Nova_IT691_Dataset_Version_2_and_3?d=w6b005fdd5ccd4940845187154d6eef45&csf=1&web=1&e=xgkgab)


## CICIDS2017 Baseline Runner

A standalone Python runner for reproducible Decision Tree and Random Forest baseline experiments using the official CICIDS2017 training and validation datasets. It exports classification metrics, Normal-vs-Attack false positives/false negatives, and structured error cases for later AI analysis.

See [Baseline Runner Documentation](docs/cic_baseline_runner.md) for setup, execution, outputs, and data-quality considerations.
