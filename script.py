import numpy as np
import pandas as pd
import dataprocessor
import featureExtraction 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if __name__ == '__main__':
    dataset = r'..\dataset\data'
    preprocessing = dataprocessor.dataPreprocessing(r'../dataset/data',100)
    data = preprocessing.save_and_load_sampled_super_class(r'C:\Users\diwah\Desktop\mfc\dataset\preprocessed\disjoint_subset',save=True)
    extraction = featureExtraction.featureExtractor(data)
    
    vgmap = extraction.get_VG_feature_map(12,100,15)
    
    df = extraction.load_as_dataframe(vgmap,12,15)
    extraction.generate_praquet(df,"disjoint_subset.praquet")