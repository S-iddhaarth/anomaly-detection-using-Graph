import numpy as np
import pandas as pd
import dataprocessor
import featureExtraction
from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(extraction, start, end, step):
    # Process a chunk of data using feature extraction
    return extraction.get_VG_feature_map(start, end, step)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)  # Adjust the level as needed

    dataset = r'..\dataset\data'
    preprocessing = dataprocessor.dataPreprocessing()
    data = preprocessing.load_saved_sampled_super_class(r'..\dataset\preprocessed\subset_full')
    extraction = featureExtraction.featureExtractor(data)

    # Define the range and step for VG feature map extraction
    start = 1
    end = 100
    step = 15

    # Determine the number of chunks for multiprocessing
    num_chunks = 5
    chunk_size = (end - start) // num_chunks

    # Create arguments for each chunk
    args = [(extraction, start + i * chunk_size, start + (i + 1) * chunk_size, step) for i in range(num_chunks)]

    # Use multiprocessing Pool to distribute the work
    with Pool(num_chunks) as p:
        results = p.starmap(process_data, args)

    # Combine the results from all processes
    vgmap = {k: v for d in results for k, v in d.items()}

    # Load the results into a DataFrame and generate a parquet file
    df = extraction.load_as_dataframe(vgmap)
    extraction.generate_praquet(df, "new.parquet")

