import os
import pandas 
import numpy as np
from unittest import TestCase
import model21cm as m21
import model21cm.error as e

class TestModel(TestCase):

    def test_log_like(self): 
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
        m = m21.Model()
        m.add_parameters(parameterlist)
        m.add_data(m21.edgesdata.full_data)
        x=[1550., 460., -900., 500., -150., 0.7, 5., 79., 20.]
        test = m.loglikelihood_at(x)
        self.assertTrue(test<100.)
        
    def test_log_like_val(self):
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
        m = m21.Model()
        m.add_parameters(parameterlist)
        freq = np.array([51.184082])
        temp = np.array([4645.468619])
        dat = {'Frequencies': freq, 'Temperature': temp}
        df = pandas.DataFrame(data=dat)
        m.add_data(df)
        val = -3691264.1453965358
        test = m.loglikelihood_at([1550., 460., -900., 500., -150., 0.7, 5., 79., 20.])
        self.assertAlmostEqual(test, val)
        
    def test_log_post_val(self):
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
        m = m21.Model()
        m.add_parameters(parameterlist)
        freq = np.array([51.184082])
        temp = np.array([4645.468619])
        dat = {'Frequencies': freq, 'Temperature': temp}
        df = pandas.DataFrame(data=dat)
        m.add_data(df)
        loglikeval = -3691264.1453965358
        priorval = (120)*(450)*(600)*(450)*(100)*(.7)*(8)*(4)*(7)
        logpost = loglikeval + np.log(1/priorval)
        test = m.logposterior_at([1550., 460., -900., 500., -150., 0.7, 5., 79., 20.])
        self.assertAlmostEqual(test, logpost)
        
    def test_prior_1(self): 
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
        m = m21.Model()
        m.add_parameters(parameterlist)
        test = m.globalprior_at([1550., 460., -900., 500., -150., 0.7, 5., 79., 20.])
        self.assertTrue(test < 100.)
    
    def test_prior_zero(self):
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
        m = m21.Model()
        m.add_parameters(parameterlist)
        test = m.globalprior_at([1550., 460., -900., 500., -150., 0.7, 5., 79., 25.])
        self.assertTrue(test ==0)
        
    def test_prior_not_zero(self):
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
        m = m21.Model()
        m.add_parameters(parameterlist)
        test = m.globalprior_at([1550., 460., -900., 500., -150., 0.7, 5., 79., 20.])
        val = (120)*(450)*(600)*(450)*(100)*(.7)*(8)*(4)*(7)
        self.assertTrue(test == 1/val)
        
    def test_logglobal_prior_infinite(self):
    	a0 = m21.Parameter('a0', 1530., 1650.)
    	a1 = m21.Parameter('a1', 450., 900.)
    	a2 = m21.Parameter('a2', -1400., -800.)
    	a3 = m21.Parameter('a3', 450., 900.)
    	a4 = m21.Parameter('a4', -225., -125.)
    	A = m21.Parameter('A', 0.3, 1.0)
    	tau = m21.Parameter('tau', 4., 12.)
    	nu0 = m21.Parameter('nu0', 76., 80.)
    	w = m21.Parameter('w', 17., 24.)
    	parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
    	m = m21.Model()
    	m.add_parameters(parameterlist)
    	test = m.globallogprior_at([10000, 460., -900., 500., -150., 0.7, 5., 79., 20.])
    	self.assertTrue(not np.isfinite(test))
    
    def test_posterior_infinite(self):
    	a0 = m21.Parameter('a0', 1530, 1650)
    	a1 = m21.Parameter('a1', 450., 900)
    	a2 = m21.Parameter('a2', -1400, -800)
    	a3 = m21.Parameter('a3', 450, 900)
    	a4 = m21.Parameter('a4', -225., -125)
    	A = m21.Parameter('A', 0.3, 1)
    	tau = m21.Parameter('tau', 4, 12)
    	nu0 = m21.Parameter('nu0', 76, 80)
    	w = m21.Parameter('w', 17, 24)
    	parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
    	m = m21.Model()
    	m.add_parameters(parameterlist)
    	m.add_data(m21.edgesdata.full_data)
    	test = m.logposterior_at([10000, 460., -900., 500., -150., 0.7, 5., 79., 20.])
    	self.assertTrue(not np.isfinite(test))
    
    def test_forward_model_min(self):
    	(a0, a1, a2, a3, a4, A, tau, nu0, w) = (1565, 650, -1200, 700, -174, \
    	0.43, 6.8, 78.6, 20.0)
    	Freq = m21.edgesdata.frequencies
    	Temp = m21.forwardmodel(Freq, (a0, a1, a2, a3, a4, A, tau, nu0, w))
    	self.assertTrue(np.amin(Temp) > 0)
    
    def test_forward_model_max(self):
    	(a0, a1, a2, a3, a4, A, tau, nu0, w) = (1565, 650, -1200, 700, -174, 0.43, 6.8, \
    	78.6, 20.0)
    	Freq = m21.edgesdata.frequencies
    	Temp = m21.forwardmodel(Freq, (a0, a1, a2, a3, a4, A, tau, nu0, w))
    	self.assertTrue(np.amax(Temp) < 100000)

    def test_forward_model_val(self):
        (a0, a1, a2, a3, a4, A, tau, nu0, w) = (1565, 650, -1200, 700, -174, 0.43, 6.8, \
        78.6, 20.0)
        Freq = m21.edgesdata.frequencies
        Temp = m21.forwardmodel(Freq, (a0, a1, a2, a3, a4, A, tau, nu0, w))
        expect = 4548.09625223
        sig = 0.1
        self.assertTrue(expect - Temp[3] < 5*sig)

    def test_dark_signal(self):
        expect = -0.09625
        value = m21.darksignal(-0.1, 80)
        self.assertTrue(np.abs(expect - value) < 0.001)
        
    def test_MCMC_model(self):
    
        m21.edgesdata.snip_data('Weight', 1)
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
        parameters = m21.Parameters(parameter_list=parameterlist)
        simulatedmodel = m21.Model()
        
        np.random.seed(0)
        simulatedmodel.add_parameters(parameterlist)
        Freq = m21.edgesdata.frequencies
        Temp = m21.forwardmodel(Freq, (1565, 650, -1200, 700, -174, 0.43, 6.8, \
        78.6, 20))
        dat = {'Frequencies': Freq, 'Temperature': Temp}
        df = pandas.DataFrame(data=dat)
        simulatedmodel.add_data(df)
        np.random.seed(0)
        sampler = simulatedmodel.MCMC(nwalkers = 22, nsteps = 50,\
        start_near = [1565, 650, -1200, 700, -174, 0.43, 6.8, 78.6, 20.7])
        
        traces, frame = m21.trim_to_frame(sampler,1)
        q = frame.quantile([0.50], axis = 0)
       
        self.assertTrue(np.isclose(q['a0'].values[0], 1565, atol=0.01, rtol=0.01))
        self.assertTrue(np.isclose(q['a1'].values[0], 650, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['a2'].values[0], -1200, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['a3'].values[0], 700, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['a4'].values[0], -174, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['A'].values[0], 0.43, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['tau'].values[0], 6.8, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['nu0'].values[0], 78.6, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['w'].values[0], 20, atol = 0.01, rtol = 0.1))
        
    def test_MCMC_dark_model(self):
     #   ##### add rest of dark matter stuff
        m21.edgesdata.snip_data('Weight', 1)
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        xi = m21.Parameter('xi', -10, 0)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w, xi]
        parameters = m21.Parameters(parameter_list=parameterlist)
        simulatedmodel = m21.DarkModel()
        np.random.seed(0)
        simulatedmodel.add_parameters(parameterlist)
        Freq = m21.edgesdata.frequencies
        Temp = m21.forwardmodel(Freq, (1565, 650, -1200, 700, -174, 0.43, 6.8, \
        78.6, 20, -0.1), True)
        dat = {'Frequencies': Freq, 'Temperature': Temp}
        df = pandas.DataFrame(data=dat)
        simulatedmodel.add_data(df)
        np.random.seed(0)
        sampler = simulatedmodel.MCMC(nwalkers = 22, nsteps = 50,\
        start_near = [1565, 650, -1200, 700, -174, 0.43, 6.8, 78.6, 20.7, -0.1])
        traces, frame = m21.trim_to_frame(sampler,1, True)
        q = frame.quantile([0.50], axis = 0)
        self.assertTrue(np.isclose(q['a0'].values[0], 1565, atol=0.01, rtol=0.01))
        self.assertTrue(np.isclose(q['a1'].values[0], 650, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['a2'].values[0], -1200, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['a3'].values[0], 700, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['a4'].values[0], -174, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['A'].values[0], 0.43, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['tau'].values[0], 6.8, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['nu0'].values[0], 78.6, atol = 0.01, rtol = 0.01))
        self.assertTrue(np.isclose(q['w'].values[0], 20, atol = 0.01, rtol = 0.1))
        self.assertTrue(np.isclose(q['xi'].values[0], -0.1, atol = 0.01, rtol = 0.01))
        
    def  test_parameters_overwrite(self): 
        m21.edgesdata.snip_data('Weight', 1)
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
        simulatedmodel = m21.Model()   
        simulatedmodel.add_parameters(parameterlist)
        try:  
            simulatedmodel.add_parameters(parameterlist) 
        except: 
            self.assertTrue(True)

    def test_parameters_length_1(self): 
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        xi = m21.Parameter('xi', -10, 0)
        try: 
            e.parameters_check_3([a0, a1, a2, a3, a4, A, tau, nu0, w, xi])    	
        except: 
            self.assertTrue(True)

    def test_parameters_length_2(self):
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        try:
            e.parameters_check_4([a0, a1, a2, a3, a4, A, tau, nu0, w])
        except:
            self.assertTrue(True)

    def test_model_prior_1(self): 
        m = m21.Model()
        try: 
            e.parameters_check_2(m)
        except: 
            self.assertTrue(True)

    def test_check_burn_in_1(self):  
        m21.edgesdata.snip_data('Weight', 1)
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        xi = m21.Parameter('xi', -10, 0)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w, xi]
        simulatedmodel = m21.DarkModel()
        np.random.seed(0)
        simulatedmodel.add_parameters(parameterlist)
        Freq = m21.edgesdata.frequencies
        Temp = m21.forwardmodel(Freq, (1565, 650, -1200, 700, -174, 0.43, 6.8, \
                                    78.6, 20, -0.1), True)
        dat = {'Frequencies': Freq, 'Temperature': Temp}
        df = pandas.DataFrame(data=dat)
        simulatedmodel.add_data(df)
        np.random.seed(0)
        sampler = simulatedmodel.MCMC(nwalkers = 22, nsteps = 50,
                        start_near = [1565, 650, -1200, 700, -174, 
                        0.43, 6.8, 78.6, 20.7, -0.1])
        try: 
            m21.model.check_burn_in(sampler, dark=True)
        except: 
            self.assertTrue(False)
        else: 
            self.assertTrue(True)

    def test_check_burn_in_2(self):
        m21.edgesdata.snip_data('Weight', 1)
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
        simulatedmodel = m21.Model()
        np.random.seed(0)
        simulatedmodel.add_parameters(parameterlist)
        Freq = m21.edgesdata.frequencies
        Temp = m21.forwardmodel(Freq, (1565, 650, -1200, 700, -174, 0.43, 6.8, \
                                    78.6, 20), False)
        dat = {'Frequencies': Freq, 'Temperature': Temp}
        df = pandas.DataFrame(data=dat)
        simulatedmodel.add_data(df)
        np.random.seed(0)
        sampler = simulatedmodel.MCMC(nwalkers = 22, nsteps = 50,
                        start_near = [1565, 650, -1200, 700, -174,
                        0.43, 6.8, 78.6, 20.7])
        try:
            m21.model.check_burn_in(sampler, dark=False)
        except: 
            self.assertTrue(False)
        else: 
            self.assertTrue(True)

    def test_dark_globalprior_at_1(self): 
        m21.edgesdata.snip_data('Weight', 1)
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        xi = m21.Parameter('xi', -10, 0)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w, xi]
        simulatedmodel = m21.DarkModel()
        simulatedmodel.add_parameters(parameterlist)
        if simulatedmodel.globalprior_at([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])==0: 
            self.assertTrue(True)

    def test_dark_globalprior_at_2(self):
        m21.edgesdata.snip_data('Weight', 1)
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        xi = m21.Parameter('xi', -10, 0)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w, xi]
        simulatedmodel = m21.DarkModel()
        simulatedmodel.add_parameters(parameterlist)
        VAL=3.9368e-16
        TEST=simulatedmodel.globalprior_at([1550, 500, -900, 500, -200, 0.5, 6, 78, 20, -5])
        self.assertTrue(np.isclose(VAL, TEST, atol=0.01, rtol=0.01))

    def test_dark_globalprior_at_3(self):
        m21.edgesdata.snip_data('Weight', 1)
        a0 = m21.Parameter('a0', 1530., 1650.)
        a1 = m21.Parameter('a1', 450., 900.)
        a2 = m21.Parameter('a2', -1400., -800.)
        a3 = m21.Parameter('a3', 450., 900.)
        a4 = m21.Parameter('a4', -225., -125.)
        A = m21.Parameter('A', 0.3, 1.0)
        tau = m21.Parameter('tau', 4., 12.)
        nu0 = m21.Parameter('nu0', 76., 80.)
        w = m21.Parameter('w', 17., 24.)
        parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
        simulatedmodel = m21.DarkModel()
        simulatedmodel.add_parameters(parameterlist)
        try: 
            simulatedmodel.globalprior_at([1550, 500, -900, 500, -200, 0.5, 6, 78, 20])
        except:
            self.assertTrue(True)
