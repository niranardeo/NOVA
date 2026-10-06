## Dataset

The full CICIDS2017 dataset and generated train/test splits are not
included in this repository because of their file size.

Cleaned dataset can be found in the link below:
[CICIDS2017 Cleaned and Preprocessed](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed)

##Split Dataset Files

The processed CICIDS2017 training, validation, and testing datasets are stored externally because
they are too large to host directly in this GitHub repository.

[Download the datasets from OneDrive](https://paceuniversity-my.sharepoint.com/:f:/g/personal/kv56727n_pace_edu/IgCpGdjKHwFbQI3mML1psxfGAb3UaCey_cNOsg3PItyZrb4?e=DVnBqj)

## CICIDS2017 Baseline Runner

A standalone Python runner for reproducible Decision Tree and Random Forest baseline experiments using the official CICIDS2017 training and validation datasets. It exports classification metrics, Normal-vs-Attack false positives/false negatives, and structured error cases for later AI analysis.

See [Baseline Runner Documentation](docs/cic_baseline_runner.md) for setup, execution, outputs, and data-quality considerations.