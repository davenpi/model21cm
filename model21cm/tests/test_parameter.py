import os
import pandas
import model21cm as m21
from unittest import TestCase

class TestParameter(TestCase): 


    def test_valid_instantiation(self): 
        test = m21.Parameter('X', 10., 20.)
        self.assertTrue(test.prior is not None)

    def test_invalid_instantiation(self): 
        test = m21.Parameter('X', 10.)
        self.assertTrue((test.uniformpriorflag is None) and (test.jeffreyspriorflag is None))

    def test_reasonable_uniform(self): 
        test = m21.Parameter('X', 10., 20.)
        self.assertTrue(test.prior(15.)==0.1 and test.prior(25.)==0.)

    def test_resonable_jeffreys(self): 
        test = m21.Parameter('X')
        test.set_jeffreys_prior(10., 20.)
        self.assertTrue(test.prior(15.)>0. and test.prior(15.)<1.)

    def test_create_prior(self): 
        try: 
            test = m21.Parameter('X', 20., 10.)
        except ValueError: 
            self.assertTrue(True)

    def test_create_jeffreys_prior(self): 
        try: 
            test = m21.Parameter('X')
            test.set_jeffreys_prior(20., 10.)
        except ValueError: 
            self.assertTrue(True)

    def test_prior_1(self): 
        t1 = m21.Parameter('X', 10., 20.)
        VAL = 0.1
        DELTA = 0.0001
        test = ((t1.prior_at(15.) > (VAL-DELTA)) and t1.prior_at(15.) < (VAL+DELTA))
        self.assertTrue(test)

    def test_prior_2(self): 
        t1 = m21.Parameter('X')
        t1.set_jeffreys_prior(10., 20.) 
        VAL = 0.0961797
        DELTA = 0.0001
        test = ((t1.prior_at(15.) > (VAL-DELTA)) and t1.prior_at(15.) < (VAL+DELTA))
        self.assertTrue(test)     

    def test_prior_3(self): 
        t1 = m21.Parameter('X')
        try: 
            t1.prior_at(15.)
        except: 
            self.assertTrue(True)

    def test_prior_4(self):
        t1 = m21.Parameter('X', 10., 20.)
        VAL = -2.30259
        DELTA = 0.001
        test = ((t1.logprior_at(15.) > (VAL-DELTA)) and (t1.logprior_at(15.) < (VAL+DELTA)))
        self.assertTrue(test)

    def test_prior_5(self):
        t1 = m21.Parameter('X')
        t1.set_jeffreys_prior(10., 20.)
        VAL = -2.34154
        DELTA = 0.001
        test = ((t1.logprior_at(15.) > (VAL-DELTA)) and (t1.logprior_at(15.) < (VAL+DELTA)))
        self.assertTrue(test)

    def test_prior_6(self):
        t1 = m21.Parameter('X')
        try:
            t1.logprior_at(15.)
        except:
            self.assertTrue(True)
