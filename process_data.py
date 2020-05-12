import hashlib
import os
import tempfile
import csv
import enum
import tarfile
import yaml
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.api_core.exceptions import GoogleAPIError

load_dotenv()

CACHE_DIRECTORY = "data/raw"
PROCESSED_DIRECTORY = "data/processed"
IMAGE_HEIGHT = IMAGE_WIDTH = 50

class DataLabel(enum.Enum): 
    TID = 0
    RPS = 1
    ISO = 2

class FeatureType(enum.Enum):
    gas_density = 0
    gas_kinematics = 1
    star_density = 2
    star_kinematics = 3

class GroupType(enum.Enum):
    m1 = 0
    m2 = 1
    m3 = 2
    m4 = 3
    m4a = 4
    m5 = 5
    m5a = 6
    m6 = 7
    m6a = 8
    m7 = 9
    m8 = 10
    m9 = 11

def sha256sum(file_name):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(file_name, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()

class FileMetaData:
    client = storage.Client()

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

        blob = Blob.from_string(blob_uri, client=self.client)
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

class RawDatasetMetaData:
    def __init__(self, file_metadata, label, feature_type, data_group,
                image_file_name, label_file_name, num_rows):
        self.file_metadata = file_metadata
        self.data_label = DataLabel[label]
        self.feature_type = FeatureType[feature_type]
        self.data_group_type = GroupType[data_group]

        self.image_file_name = image_file_name
        self.label_file_name = label_file_name
        self.num_rows = int(num_rows)
    
    def __str__(self):
        return f"RawDatsetMetadata: {self.data_group_type.name}-{self.feature_type.name}-{self.data_label.name}"

    def __repr__(self):
        return f"<{self.__str__()}>"

class ProcessedDataset:
    def __init__(self, raw_dataset_group, group_type, feature_type):
        self.raw_data = raw_dataset_group
        self.group_type = group_type
        self.feature_type = feature_type

        self.file_path = Path(PROCESSED_DIRECTORY) / f"{self.group_type.name}.{self.feature_type.name}.tar.gz"

    def valid_dataset(self) -> bool:
        # Check that all labels in dataset
        if len(self.raw_data) != len(DataLabel):
            return False
        
        # Check that all files have the same number of rows
        num_rows_0 = self.raw_data[0].num_rows
        if not all([x.num_rows == num_rows_0 for x in self.raw_data]):
            return False

        return True

    def _add_dataset(self, dataset, output_file) -> dict:
        if dataset.file_metadata.file_accessible():
            metadata = {}
            
            with tarfile.open(dataset.file_metadata.file_path, "r:gz") as f:
                # Extract image data
                image_data = np.loadtxt(f.extractfile(dataset.image_file_name))
                image_data = image_data.reshape(dataset.num_rows, IMAGE_HEIGHT, IMAGE_WIDTH)

                image_file = tempfile.NamedTemporaryFile("wb")
                np.save(image_file, image_data)
                
                # Extract force data
                force_data = np.loadtxt(f.extractfile(dataset.label_file_name))
                force_data = force_data.reshape(dataset.num_rows)

                force_file = tempfile.NamedTemporaryFile("wb")
                np.save(force_file, force_data)

                # Process label data
                label_data = np.zeros_like(force_data)
                label_data.fill(dataset.data_label.value)
                
                label_file = tempfile.NamedTemporaryFile("wb")
                np.save(label_file, label_data)

                # Write output
                outputs = [
                    ("image", image_file.name, f"{dataset.data_label.name}.image.data.npy"),
                    ("force", force_file.name, f"{dataset.data_label.name}.force.data.npy"),
                    ("label", label_file.name, f"{dataset.data_label.name}.label.data.npy")
                ]
                
                for feature, file_name, arcname in outputs:
                    output_file.add(file_name, arcname=arcname)
                    metadata[feature] = {
                        "file_name": arcname,
                        "hash": sha256sum(file_name)
                    }

                return metadata
        else:
            raise Exception(f"{dataset} Not available")

    def _add_metadata(self, datasets_metadata, output_file):
        metadata_file = tempfile.NamedTemporaryFile("w")

        num_rows_0 = self.raw_data[0].num_rows

        metadata = {}
        metadata["files"] = datasets_metadata
        metadata["num_rows"] = num_rows_0
        metadata["image_height"] = IMAGE_HEIGHT
        metadata["image_width"] = IMAGE_WIDTH

        metadata["group_type"] = self.group_type.name
        metadata["feature_type"] = self.feature_type.name

        yaml.dump(metadata, metadata_file)
        output_file.add(metadata_file.name, arcname=f"metadata.yml")
    
    def process_dataset(self):
        output_file = tarfile.open(self.file_path, "w:gz")

        datasets_metadata = {}
        for dataset in self.raw_data:
            metadata = self._add_dataset(dataset, output_file)
            datasets_metadata[dataset.data_label.name] = metadata

        self._add_metadata(datasets_metadata, output_file)

        output_file.close()

if __name__=="__main__":
    raw_datasets = []
    
    # Load raw datasets
    with open("datafiles.metadata.csv") as f:
        reader = csv.DictReader(f)
        for data in reader:
            file_meta_data = FileMetaData(data["dataFile"], data["fileHash"])
            
            raw_dataset = RawDatasetMetaData(
                file_meta_data,
                label=data["dataLabel"],
                feature_type=data["dataSimType"],
                data_group=data["dataSetLabel"],
                image_file_name=data["imageFileName"],
                label_file_name=data["labelFileName"],
                num_rows=data["imageFileNumRows"]
            )
            raw_datasets.append(raw_dataset)

    # Process datasets
    for group in GroupType:
        for feature_type in FeatureType:            
            # Create lambda filter function 
            filter_function = lambda x: \
                x.data_group_type == group and \
                x.feature_type == feature_type
            
            # Get the raw data set group
            raw_dataset_group = list(
                filter(
                    filter_function,
                    raw_datasets
                )
            )

            processed_dataset = ProcessedDataset(raw_dataset_group, group, feature_type)
            if processed_dataset.valid_dataset():
                print(f"Processing group: {group.name}-{feature_type.name}")
                processed_dataset.process_dataset()
