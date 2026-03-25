from chronic_disease_risk.data_sources.nhanes_download import download_from_config


if __name__ == "__main__":
    download_from_config(config_path="configs/nhanes.yaml")
