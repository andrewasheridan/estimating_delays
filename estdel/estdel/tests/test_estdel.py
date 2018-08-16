import unittest
import numpy as np

from estdel.estdel import _DelayPredict, DelaySolver, VratioDelayMagnitude


N_FREQS = 1024
MAX_EST_MAG = 0.0400
NUM_TIMES = 60
MAX_EST_MAG = 0.0400
ESTIMATE_WIDTH = 0.0001

DATA = np.exp(-2j * np.pi * np.arange(N_FREQS) * MAX_EST_MAG).reshape(-1, N_FREQS) # shape = (1,  1024)
V_DATA = np.tile(DATA, (NUM_TIMES, 1)) # shape = (60,  1024)


########################################################################################################
# test _DelayPredict
class test_DelayPredict(unittest.TestCase):

	# test __init__
	def test_init_data_is_list_or_array(self):

		data = set(range(N_FREQS))
		self.assertRaises(AssertionError, _DelayPredict, data)


	def test_init_data_capture(self):
		# ???: Why test this, is there any way it could fail?
		# ???: Should I be testing every single assignment on init?

		delayPredict = _DelayPredict(DATA)
		np.testing.assert_array_equal(delayPredict._data, DATA)


	def test_init_data_type_is_complex(self):

		data = np.real(DATA)
		self.assertRaises(AssertionError, _DelayPredict, data)


	def test_init_data_shape_last_is_N_FREQS(self):

		data = np.exp(-2j * np.pi * np.arange(N_FREQS + 1) * MAX_EST_MAG).reshape(-1, N_FREQS + 1)
		self.assertRaises(AssertionError, _DelayPredict, data)

	# test _angle_tx
	def test_angle_tx_scaling_lower_bound_is_zero(self):

		x = np.array([-4])
		self.assertRaises(AssertionError, _DelayPredict(DATA)._angle_tx, x)


	def test_angle_tx_scaling_upper_bound_is_one(self):

		x = np.array([4])
		self.assertRaises(AssertionError, _DelayPredict(DATA)._angle_tx, x)

	# test _predict 
	#def test_predict....
	# XXX: How do I test this?





########################################################################################################
# test VratioDelayMagnitude
class test_VratioDelayMagnitude(unittest.TestCase):

	# test _convert_precictions
	def test_convert_predictions_conversion_fn_correct_string(self):

		conversion_fn = 'Default'
		raw_predictions = np.arange(0, MAX_EST_MAG + ESTIMATE_WIDTH, ESTIMATE_WIDTH)[:NUM_TIMES]
		_VratioDelayMagnitude = VratioDelayMagnitude(V_DATA, conversion_fn)
		self.assertRaises(AssertionError, _VratioDelayMagnitude._convert_predictions, raw_predictions)

	def test_convert_predictions_conversion_fn_is_callable(self):

		conversion_fn = [1]
		raw_predictions = np.arange(0, MAX_EST_MAG + ESTIMATE_WIDTH, ESTIMATE_WIDTH)[:NUM_TIMES]
		_VratioDelayMagnitude = VratioDelayMagnitude(V_DATA, conversion_fn)
		self.assertRaises(AssertionError, _VratioDelayMagnitude._convert_predictions, raw_predictions)




########################################################################################################
# test DelaySolver
class test_DelaySolver(unittest.TestCase):

	# test __init__
	def test_init_list_o_sep_pairs_capture(self):

		list_o_sep_pairs = [[(0, 1), (2, 3)]]
		delaySolver = DelaySolver(list_o_sep_pairs, V_DATA)
		np.testing.assert_array_equal(delaySolver._list_o_sep_pairs, list_o_sep_pairs)

	def test_init_list_o_sep_pairs_shape_1_is_2(self):

		list_o_sep_pairs = [[(1, 2)]] # shape = (1, 1, 2)
		self.assertRaises(AssertionError, DelaySolver, list_o_sep_pairs, V_DATA)

	def test_init_list_o_sep_pairs_shape_2_is_2(self):

		list_o_sep_pairs = [[(1, ), (2, )]] # shape = (1, 2, 1)
		self.assertRaises(AssertionError, DelaySolver, list_o_sep_pairs, V_DATA)

	def test_init_list_o_sep_pairs_dtype_is_int(self):

		list_o_sep_pairs = [[(1., 2), (3, 4)]]
		self.assertRaises(AssertionError, DelaySolver, list_o_sep_pairs, V_DATA)

	# test true_b
	def test_true_b_true_delays_keys_equals_unique_ants(self):

		list_o_sep_pairs = [[(0, 1), (2, 3)]]
		delaySolver = DelaySolver(list_o_sep_pairs, V_DATA)
		true_ant_delays = {1: 1.2}
		self.assertRaises(AssertionError, delaySolver.true_b, true_ant_delays)




if __name__ == '__main__':
	unittest.main()
