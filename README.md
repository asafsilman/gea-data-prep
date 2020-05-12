# Galaxy Environment Analysis Data Preparation
This repo is used to prepare gea data sets to be uploaded to GCP

The raw data is located under `./data/raw`
The processed data is located under `./data/processed`

# Instructions to run
1. Create `.env` file from `.env.template` and fill in fields
2. Make sure the python dependencies are installed `pip install -r requirements.txt`
3. Run `python process_data.py` to generate the processed data
4. Run `python upload_data.py` to upload data to bucket
