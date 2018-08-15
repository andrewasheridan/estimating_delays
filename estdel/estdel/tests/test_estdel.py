import unittest
from estdel.estdel import _DelayPredict, DelaySolver
#from estdel.estdel import N_FREQS
import numpy as np

N_FREQS = 1024
MAX_EST_MAG = 0.0400

DATA = np.exp(-2j * np.pi * np.arange(N_FREQS) * MAX_EST_MAG).reshape(-1, N_FREQS)

# number of rows (times) in visibility 
NUM_TIMES = 60

class test_DelayPredict(unittest.TestCase):

	# test __init__
	def test_init_data_is_list_or_array(self):

		data = set(range(1024))
		self.assertRaises(AssertionError, _DelayPredict, data)


	def test_init_data_capture(self):

		delayPredict = _DelayPredict(DATA)
		np.testing.assert_array_equal(delayPredict._data, DATA)


	def test_init_data_type_is_complex(self):

		data = np.real(DATA)
		self.assertRaises(AssertionError, _DelayPredict, data)


	def test_init_data_shape_is_N_FREQS(self):

		data = np.exp(-2j * np.pi * np.arange(N_FREQS + 1) * MAX_EST_MAG).reshape(-1, N_FREQS + 1)
		self.assertRaises(AssertionError, _DelayPredict, data)

	# test _angle_tx
	def test_angle_tx_scaling_lower_bound_is_zero(self):

		x = np.array([-4])
		self.assertRaises(AssertionError, _DelayPredict(DATA)._angle_tx, x)


	def test_angle_tx_scaling_upper_bound_is_one(self):

		x = np.array([4])
		self.assertRaises(AssertionError, _DelayPredict(DATA)._angle_tx, x)

	# test _predict ...
	# XXX: How do I test this?


# test VratioDelayMagnitude
#class test_VratioDelayMagnitude(unittest.TestCase):

	#def test_convert_predictions_bad_string(self):
	# XXX: How do i test this, it needs raw_predictions, but that doesnt exist until just before convert gets called




# test DelaySolver
class test_DelaySolver(unittest.TestCase):

	# test __init__
	def test_init_list_o_sep_pairs_capture(self):

		list_o_sep_pairs = [[(1, 2), (3, 4)]]
		data = np.tile(DATA, (NUM_TIMES, 1))
		delaySolver = DelaySolver(list_o_sep_pairs, data)
		np.testing.assert_array_equal(delaySolver._list_o_sep_pairs, list_o_sep_pairs)

	def test_init_list_o_sep_pairs_shape_1_is_2(self):

		list_o_sep_pairs = [[(1, 2)]] # shape = (1, 1, 2)
		data = np.tile(DATA, (NUM_TIMES, 1))
		self.assertRaises(AssertionError, DelaySolver, list_o_sep_pairs, data)

	def test_init_list_o_sep_pairs_shape_2_is_2(self):

		list_o_sep_pairs = [[(1, ), (2, )]] # shape = (1, 2, 1)
		data = np.tile(DATA, (NUM_TIMES, 1))
		self.assertRaises(AssertionError, DelaySolver, list_o_sep_pairs, data)

	def test_init_list_o_sep_pairs_dtype_is_int(self):

		list_o_sep_pairs = [[(1., 2), (3, 4)]]
		data = np.tile(DATA, (NUM_TIMES, 1))
		self.assertRaises(AssertionError, DelaySolver, list_o_sep_pairs, data)

if __name__ == '__main__':
	unittest.main()
