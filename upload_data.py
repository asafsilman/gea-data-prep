import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage.blob import Blob

PROCESSED_DIRECTORY = "data/processed"

if __name__ == "__main__":
    load_dotenv()

    client = storage.Client()

    for path in Path(PROCESSED_DIRECTORY).glob("*.tar.gz"):
        name = path.name
        print(f"Uploading {name} to bucket")

        blob_uri = f"{os.environ['PROCESSED_DATA_BUCKET']}/{name}"
        blob = Blob.from_string(blob_uri, client=client)
        blob.upload_from_filename(str(path))
