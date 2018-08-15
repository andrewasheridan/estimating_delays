# Data_Creator_R

from data_manipulation import *
from Data_Creator import Data_Creator


import numpy as np

class Data_Creator_R(Data_Creator):
    """Creates data in an alternate thread. R for regression.
    
    ## usage:
    ## data_maker = Data_Creator_R(num_flatnesses=250, mode = 'train')
    ## data_maker.gen_data() #before loop
    ## inputs, targets = data_maker.get_data() #start of loop
    ## data_maker.gen_data() #immediately after get_data()

    """

    def __init__(self,
                 num_flatnesses,
                 bl_data = None,
                 bl_dict = None,
                 gains = None,
                 abs_min_max_delay = 0.040):
        
        """
        Arguments
            num_flatnesses : int - number of flatnesses used to generate data.
                                   Number of data samples = 60 * num_flatnesses
            bl_data : data source. Output of get_seps_data()
            bl_dict : dict - Dictionary of seps with bls as keys. An output of get_or_gen_test_train_red_bls_dicts()
            gains : dict - Gains for this data. An output of load_relevant_data()
            
                                   
        """
        Data_Creator.__init__(self,
                              num_flatnesses = num_flatnesses,
                              bl_data = bl_data,
                              bl_dict = bl_dict,
                              gains = gains,
                              abs_min_max_delay = abs_min_max_delay)
        
             
    def _gen_data(self):
        
        # scaling tools
        # the NN likes data in the range (0,1)
        angle_tx  = lambda x: (np.asarray(x) + np.pi) / (2. * np.pi)
        angle_itx = lambda x: np.asarray(x) * 2. * np.pi - np.pi

        delay_tx  = lambda x: (np.array(x) + self._tau) / (2. * self._tau)
        delay_itx = lambda x: np.array(x) * 2. * self._tau - self._tau
        
        targets = np.random.uniform(low = -self._tau, high = self._tau, size = (self._num * 60, 1))
        applied_delay = np.exp(-2j * np.pi * (targets * self._nu + np.random.uniform()))



        assert type(self._bl_data) != None, "Provide visibility data"
        assert type(self._bl_dict) != None, "Provide dict of baselines"
        assert type(self._gains)   != None, "Provide antenna gains"

        if self._bl_data_c == None:
            self._bl_data_c = {key : self._bl_data[key].conjugate() for key in self._bl_data.keys()}

        if self._gains_c == None:
            self._gains_c = {key : self._gains[key].conjugate() for key in self._gains.keys()}


        def _flatness(seps):
            """Create a flatness from a given pair of seperations, their data & their gains."""

            a, b = seps[0][0], seps[0][1]
            c, d = seps[1][0], seps[1][1]


            return self._bl_data[seps[0]]   * self._gains_c[(a,'x')] * self._gains[(b,'x')] * \
                   self._bl_data_c[seps[1]] * self._gains[(c,'x')]   * self._gains_c[(d,'x')]

        inputs = []
        for _ in range(self._num):

            unique_baseline = random.sample(self._bl_dict.keys(), 1)[0]
            two_seps = [random.sample(self._bl_dict[unique_baseline], 2)][0]

            inputs.append(_flatness(two_seps))
            

        inputs = np.angle(np.array(inputs).reshape(-1,1024) * applied_delay)
        
        permutation_index = np.random.permutation(np.arange(self._num * 60))
        

        self._epoch_batch.append((angle_tx(inputs[permutation_index]), delay_tx(targets[permutation_index])))
