import os
from pathlib import Path

from google.cloud import storage
from google.cloud.storage import Blob

from dotenv import load_dotenv; load_dotenv()

TF_RECORDS_DIRECTORY = "data/tfrecords"

if __name__=="__main__":
    root_dir = Path(TF_RECORDS_DIRECTORY)
    sub_dirs = [x for x in root_dir.iterdir() if x.is_dir()]

    print("Select dataset to upload")
    for i, sub_dir in enumerate(sub_dirs):
        print(f"[{i}] -  {sub_dir}")

    choice = -1
    while True:
        try:
            choice = int(input("Dataset: "))
            if choice in range(len(sub_dirs)):
                break
            else:
                print("Not a valid choice")
        except ValueError:
            print("Invalid input")

    selected_dir = sub_dirs[choice]

    client = storage.Client()

    for path in Path(selected_dir).glob("*.tfrecords"):
        name = path.name
        print(f"Uploading {name} to bucket")

        blob_uri = f"{os.environ['TF_RECORDS_BUCKET']}/{selected_dir.name}/{name}"
        blob = Blob.from_string(blob_uri, client=client)
        blob.chunk_size = 1024 * 1024 * 10
        blob.upload_from_filename(str(path))
