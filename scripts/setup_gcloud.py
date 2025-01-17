import yaml
from google.cloud import storage

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_bucket_if_not_exists(bucket_name, project_id):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if not bucket.exists():
        client.create_bucket(bucket_name, location="US")
        print(f"Bucket {bucket_name} created successfully.")
    else:
        print(f"Bucket {bucket_name} already exists.")

if __name__ == "__main__":
    config = load_config()
    create_bucket_if_not_exists(
        bucket_name=config["gcloud"]["bucket_name"],
        project_id=config["gcloud"]["project_id"]
    )
