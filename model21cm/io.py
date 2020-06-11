import pandas as pd
import os


class Data:


    def __init__(self, packagedata=None): 
        self.clear()
        if packagedata is not None: 
            if os.path.isfile(os.path.join(data_directory(), packagedata)): 
                self.load_package_data(packagedata)
            else: 
                self.load_user_data(packagedata)


    def load_file(self, datapath):
        self.data_path = correct_path(datapath)
        self.data_dir, self.filename = os.path.split(self.data_path)
        self.full_data = pd.read_csv(self.data_path)
        self.frequencies = self.full_data.iloc[:,0]
        self.temperatures = self.full_data.iloc[:,1]


    def load_user_data(self, datapath):
        """Loads .csv data file at specified path.""" 
        try: 
            self.load_file(datapath)
        except: 
            print("Error importing user data.")
            self.clear() 
        

    def load_package_data(self, dataset_name): 
        """Loads data file with specified name distributed with package."""
        try: 
            datapath = data_directory()
            datapath = os.path.join(datapath, dataset_name)
            self.load_file(datapath)
        except:
            print("Error importing package data.")
            self.clear()


    def snip_data(self, key, value): 
        """Snips dataset to points satisfying specified condition."""
        if key in self.full_data.keys(): 
            self.full_data = self.full_data.loc[self.full_data[key]==value]
            self.frequencies = self.full_data.iloc[:,0]
            self.temperatures = self.full_data.iloc[:,1]
        else: 
            print("Invalid key selection. Keys associated with dataset are: ")
            print(self.full_data.keys())


    def clear(self):
        """Clears attributes of data object.""" 
        self.filename = None
        self.data_dir = None
        self.data_path = None
        self.full_data = None
        self.frequencies = None
        self.temperatures = None
        self.source = None


def correct_path(pathname): 
    """Expands and checks file paths."""
    fix1 = os.path.expanduser(pathname)
    fix2 = os.path.expandvars(fix1)
    fix3 = os.path.normpath(fix2)
    fix4 = os.path.abspath(fix3)
    if not os.path.isfile(fix4): 
        raise Exception("Invalid path: {}".format(fix4))
    return fix4


def data_directory(): 
    """Returns absolute path of data directory in package distribution."""
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    path = os.path.join(path, 'data')
    return path
