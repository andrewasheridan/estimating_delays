import unittest
from estdel.estdel import _DelayPredict
#from estdel.estdel import N_FREQS
import numpy as np

N_FREQS = 1024
MAX_EST_MAG = 0.0400

DATA = np.exp(-2j * np.pi * np.arange(N_FREQS) * MAX_EST_MAG).reshape(-1, N_FREQS)

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

	# test _predict
	# XXX: How do I test this?


if __name__ == '__main__':
	unittest.main()
