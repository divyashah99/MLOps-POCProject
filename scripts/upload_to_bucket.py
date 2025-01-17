from google.cloud import storage

def upload_to_bucket(bucket_name, source_file, destination_blob):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_file)
    print(f"File {source_file} uploaded to {destination_blob}.")

if __name__ == "__main__":
    upload_to_bucket(
        bucket_name="mlops-poc-project-bucket",
        source_file="data/imdb_data.json",
        destination_blob="imdb_data.json"
    )
