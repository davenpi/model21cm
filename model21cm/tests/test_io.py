import os
import pandas
import numpy as np
import model21cm as m21
from unittest import TestCase

class TestIO(TestCase): 

    def test_has_datapath(self):
        test = m21.Data("edges_data.csv").filename
        self.assertTrue(isinstance(test, str))

    def test_dataframe_type(self): 
        test = m21.Data("edges_data.csv").full_data
        self.assertTrue(isinstance(test, pandas.core.frame.DataFrame))

    def test_frequencies_type(self): 
        test = m21.Data("edges_data.csv").frequencies
        self.assertTrue(isinstance(test, pandas.core.series.Series))

    def test_temp_type(self): 
        test = m21.Data("edges_data.csv").temperatures
        self.assertTrue(isinstance(test, pandas.core.series.Series)) 

    def test_preloaded_edges_data(self): 
        test = m21.edgesdata
        self.assertTrue(isinstance(test, m21.Data))

    def test_valid_freq_data(self): 
        test = m21.edgesdata.frequencies.values
        self.assertTrue(not(np.any(test<50) or np.any(test>100)))

    def test_clear(self): 
        d = m21.Data()
        d.data_dir = '/something/'
        d.clear()
        test = d.data_dir
        self.assertTrue(test==None)

#OS DEPENDENCY
#    def test_load_user_data(self):
#        PATH = os.path.join(os.getcwd(), "test.csv")
#        f = open(PATH,"w+")
#        f.write("f, t\n")
#        f.write("0, 1")
#        f.close()
#        testdata = m21.Data()
#        testdata.load_user_data(PATH)
#        if (testdata.frequencies[0]==0) and (testdata.temperatures[0]==1): 
#            self.assertTrue(True)
#        os.remove(PATH)

        



 
