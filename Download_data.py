###Download  files from Kaggle
import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files('brycecf/give-me-some-credit-dataset', path='.', unzip=True)
