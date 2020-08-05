import os
import tarfile
import yaml
from io import BytesIO
from pathlib import Path
from random import shuffle

from process_data import FeatureType, GroupType, DataLabel, ProcessedDataset

import tensorflow as tf
import numpy as np

MAX_RECORDS_PER_FILE = 150

TF_RECORDS_DIRECTORY = "data/tfrecords"
PROCESSED_DIRECTORY = "data/processed"

EXPERIMENT_GROUPS = GroupType.m10
EXPERIMENT_FEATURES = FeatureType.gas_density |FeatureType.gas_kinematics

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class TFDataset:
    def __init__(self, groups, features):
        self._groups = groups
        self._features = features
        
        self.groups = self._get_groups()
        self.channels = self._get_channels()

        self.sources = self._get_sources()
    
    def _get_groups(self):
        return list(map(lambda x: x.name, filter(lambda g: g in self._groups, GroupType)))

    def _get_channels(self):
        channel_features = []

        for feature in FeatureType:
            if feature in self._features:
                channel_features.append(feature.name)

        return sorted(channel_features)

    def _get_sources(self):
        root_directory = Path(PROCESSED_DIRECTORY)

        sources = {}
        for group in self.groups:
            sources[group] = {}
            for channel in self.channels:
                sources[group][channel] = \
                    root_directory / f'{group}.{channel}.tar.gz'
        
        return sources

    def _extract_source(self, source):
        feature_data = {}
        for feature in source:
            archive = tarfile.open(source[feature], 'r:gz')
            metadata = yaml.safe_load(archive.extractfile('metadata.yml'))
            for label in metadata['files']:
                label_feature_data = feature_data.setdefault(label, {})

                image_data = label_feature_data.setdefault('image_data_raw', [])
                label_data = label_feature_data.setdefault('label_data_raw', [])
                force_data = label_feature_data.setdefault('force_data_raw', [])

                # due to bug in numpy, must read to bytes io https://github.com/numpy/numpy/issues/7989
                image_data_bytes = BytesIO()
                label_data_bytes = BytesIO()
                force_data_bytes = BytesIO()
                
                image_data_bytes.write(archive.extractfile(
                    metadata['files'][label]['image']['file_name']
                ).read())
                label_data_bytes.write(archive.extractfile(
                    metadata['files'][label]['label']['file_name']
                ).read())
                force_data_bytes.write(archive.extractfile(
                    metadata['files'][label]['force']['file_name']
                ).read())

                image_data_bytes.seek(0)
                label_data_bytes.seek(0)
                force_data_bytes.seek(0)

                image_data.append(np.load(image_data_bytes))
                label_data.append(np.load(label_data_bytes))
                force_data.append(np.load(force_data_bytes))

        return feature_data

    def extract_data(self):
        extracted_sources = {}
        
        for source in self.sources:
            source_data = self._extract_source(self.sources[source])

            for label in source_data:
                source_data[label]["image_data"] = np.stack(
                    source_data[label]["image_data_raw"],
                    axis=-1
                )

                label_data = source_data[label]["label_data_raw"][0].astype(np.int)
                source_data[label]["label_data"] = one_hot(
                    label_data,
                    len(DataLabel)
                )

                # Extract force data from source
                force_data = source_data[label]["force_data_raw"][0].astype(np.float64)

                source_data[label]["force_data"] = force_data

                extracted_sources.setdefault('image_data', []).append(source_data[label]["image_data"])
                extracted_sources.setdefault('label_data', []).append(source_data[label]["label_data"])
                extracted_sources.setdefault('force_data', []).append(source_data[label]["force_data"])

        # Return extracted data
        img = extracted_sources['image_data']
        lbl = extracted_sources['label_data']
        frc = extracted_sources['force_data']
        
        return \
            np.vstack(img), \
            np.vstack(lbl), \
            np.vstack(frc)

    def _serialise_features(self, image, label, force):
        features = {
            "image/height": _int64_feature(image.shape[0]),
            "image/width": _int64_feature(image.shape[1]),
            "image/channels": _int64_feature(image.shape[2]),
            "image/label": _bytes_feature(label.tobytes()),
            "image/force": _bytes_feature(force.tobytes()),
            "image/data": _bytes_feature(image.tobytes()),
        }

        return tf.train.Example(features=tf.train.Features(feature=features))

    def create_dataset(self, image_data, label_data, force_data, dataset_name=None):
        indexes = list(range(image_data.shape[0]))
        shuffle(indexes)

        if dataset_name is None:
            dataset_name = f"gea-{''.join(self.groups)}-{''.join(self.channels)}"

        dataset_dir = Path(TF_RECORDS_DIRECTORY) / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        writer_options = tf.io.TFRecordOptions(
            # compression_type="GZIP"
        )
        
        def get_writer(file_num):
            return tf.io.TFRecordWriter(
                str(dataset_dir / f"dataset-{file_num}.tfrecords"),
                options=writer_options)
        
        file_num = 0
        examples_written = 0
        writer = get_writer(file_num)
        for idx in indexes:
            image = image_data[idx]
            label = label_data[idx]
            force = force_data[idx]

            tf_example = self._serialise_features(image, label, force)
            writer.write(tf_example.SerializeToString())

            tf_example.SerializeToString()
            examples_written += 1
            if examples_written > MAX_RECORDS_PER_FILE:
                examples_written = 0
                file_num += 1
                writer = get_writer(file_num)


if __name__=="__main__":
    dataset = TFDataset(EXPERIMENT_GROUPS, EXPERIMENT_FEATURES)
    image_data, label_data, force_data = dataset.extract_data()

    dataset.create_dataset(image_data, label_data, force_data)

