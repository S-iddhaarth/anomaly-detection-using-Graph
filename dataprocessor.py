import pandas as pd
import numpy as np
import wfdb
import ast
import os
from ts2vg import NaturalVG


class dataPreprocessing:
    def __init__(self,path:str,sampling_rate:int) -> None:
        self.path = path
        self.sampling_rate = sampling_rate
        self.X,self.Y = self._load_from_dataset()
        self.superClass = ["NORM","MI","STTC","CD","HYP"]
        self.superClassCount_o = {"NORM":9514,"MI":5469,"STTC":5235,"CD":4898,"HYP":2649}
        superSet = self.Y['diagnostic_superclass'].tolist()
        self.allClass = np.array([str(i).replace('[', '').replace(']', '').replace("'", '') for i in superSet])

    def _load_raw_data(self,df:pd.DataFrame)->np.ndarray:

        """it takes the dataframe containing metadata, sampling rate and 
        root directory of dataset as input to sample the time series signal
        and return ndarray of sampled data

        Returns:
            ndarray: (data_points,signal,channel)
        """

        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(os.path.join(self.path,f)) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(os.path.join(self.path,f)) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data


    def _load_from_dataset(self)->tuple:

        """takes path of the dataset and sampling rate as input and returns
        X,Y tuple

        Returns:
            tuple: (data,class)
        """
        Y = pd.read_csv(os.path.join(self.path,'ptbxl_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        X = self._load_raw_data(Y)

        agg_df = pd.read_csv(os.path.join(self.path,'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        return X,Y

    def save_and_load_sampled_super_class(self,root:str,save=False)->dict:
        
        """takes a path as input and saves npy file corresponding to 
            all super class
        """
        
        value = {"NORM": np.zeros((9514,1000,12)), "MI": np.zeros((5469,1000,12)), 
                "STTC": np.zeros((5235,1000,12)), "CD": np.zeros((4898,1000,12)), 
                "HYP": np.zeros((2649,1000,12))}
        
        tracker = {"NORM": 0, "MI": 0, "STTC": 0, "CD": 0, "HYP": 0}
        
        for i in range(self.X.shape[0]):
            for j in self.superClass:
                if j in self.allClass[i]:
                    value[j][tracker[j]] = self.X[i] 
                    tracker[j]+=1
        if save:
            try:
                if not os.path.exists(root):
                    os.mkdir(root)
            except Exception as e:
                print('couldn\'t make the root directory :(\nplease make the directory and call the function again')
            for i,j in value.items():
                path = os.path.join(root,i)
                if not os.path.exists(path):
                    os.mkdir(path)
                np.save(os.path.join(path,"ts"),j)
            return value
    
    
    def load_saved_sampled_super_class(self,root:str)->dict:
        superClass = os.listdir(root)
        data = {}

        for i in superClass:
            file_path = os.path.join(root, i,'ts.npy')

            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                try:
                    data[i] = np.load(file_path)
                except PermissionError as e:
                    print(f"PermissionError: {e}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"Skipping {file_path}, not a file")

        print("Data loading complete.")
        return data
