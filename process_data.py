import hashlib
import os
import csv

from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.api_core.exceptions import GoogleAPIError

load_dotenv()

CACHE_DIRECTORY = "data/raw"
PROCESSED_DIRECTORY = "data/processed"

CLIENT = storage.Client()

def sha256sum(file_name):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(file_name, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()

class FileMetaData:
    def __init__(self, file_name, file_hash):
        self.file_name = file_name
        self.file_hash = file_hash

        self.file_path = Path(CACHE_DIRECTORY) / file_name

    def _file_in_cache(self) -> bool:
        if self.file_path.exists():
            if sha256sum(self.file_path) == self.file_hash:
                return True
        return False

    def _download_to_cache(self) -> bool:
        blob_uri = f"{os.environ['RAW_DATA_BUCKET']}/{self.file_name}"

        blob = Blob.from_string(blob_uri, client=CLIENT)
        try:
            blob.download_to_filename(self.file_path)
        except GoogleAPIError:
            return False

        return self._file_in_cache()

    def file_accessible(self) -> bool:
        """Checks if the file is locally accessible.
        If it is not, attempt to download from GCP and checksum

        Returns:
            bool -- Is file accessible locally
        """
        if self._file_in_cache():
            return True
        return self._download_to_cache()

if __name__=="__main__":
    with open("datafiles.metadata.csv") as f:
        reader = csv.DictReader(f)
        for data in reader:
            meta_data = FileMetaData(data["dataFile"], data["fileHash"])
            print(data["dataFile"], meta_data.file_accessible())
