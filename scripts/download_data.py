from datasets import load_dataset
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_data():
    config = load_config()
    dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    dataset.to_json("data/imdb_data.json")
    print("Dataset downloaded and saved as JSON.")

if __name__ == "__main__":
    download_data()
