import unittest
import numpy as np

from estdel.estdel import (
    _DelayPredict,
    DelaySolver,
    VratioDelayMagnitude,
    VratioDelaySign,
    VratioDelay,
)
import estdel.constants as constants


DATA = np.exp(
    -2j * np.pi * np.arange(constants.N_FREQS) * constants.MAX_EST_MAG
).reshape(
    -1, constants.N_FREQS
)  # shape = (1,  1024)
V_DATA = np.tile(DATA, (constants.N_TIMES, 1))  # shape = (60,  1024)

EXPECTED_PREDICTIONS = np.array(
    [[-0.04, -0.0321, -0.024, -0.016, -0.008, 0., 0.0081, 0.0161, 0.0241, 0.0322]]
)

########################################################################################################
# test _DelayPredict
class test_DelayPredict(unittest.TestCase):

    # test __init__
    def test_init_data_is_list_or_array(self):

        data = set(range(constants.N_FREQS))
        self.assertRaises(TypeError, _DelayPredict, data)

    def test_init_data_capture(self):
        # ???: Why test this, is there any way it could fail?
        # ???: Should I be testing every single assignment on init?

        _delayPredict = _DelayPredict(DATA)
        np.testing.assert_array_equal(_delayPredict._data, DATA)

    def test_init_data_type_is_complex(self):

        data = np.real(DATA)
        self.assertRaises(TypeError, _DelayPredict, data)

    def test_init_data_shape_last_is_N_FREQS(self):

        data = np.exp(
            -2j * np.pi * np.arange(constants.N_FREQS + 1) * constants.MAX_EST_MAG
        ).reshape(-1, constants.N_FREQS + 1)
        self.assertRaises(ValueError, _DelayPredict, data)

    # test _angle_tx
    def test_angle_tx_scaling_lower_bound_is_zero(self):

        x = np.array([-4])
        self.assertRaises(ValueError, _DelayPredict(DATA)._angle_tx, x)

    def test_angle_tx_scaling_upper_bound_is_one(self):

        x = np.array([4])
        self.assertRaises(ValueError, _DelayPredict(DATA)._angle_tx, x)


########################################################################################################
# test VratioDelaySign
class test_VratioDelaySign(unittest.TestCase):

    # test predict
    def test_predict_network_model_exists(self):

        _VratioDelaySign = VratioDelaySign(V_DATA)
        _VratioDelaySign._model_path = "not_a_model.pb"
        with self.assertRaises(IOError):
            _VratioDelaySign.predict()

    # ???: How can I test this better?
    def test_predict_predicts(self):
        try:
            _VratioDelaySign = VratioDelaySign(V_DATA)
            predictions = _VratioDelaySign.predict()
        except:
            self.fail("Prediction failed unexpectedly")

    def test_predict_produces_expected_output(self):

        expected_predictions = np.sign(EXPECTED_PREDICTIONS).reshape(10)
        expected_predictions[expected_predictions == 0] = 1
        delays = np.arange(-0.0400, 0.0400, 0.0001)[::80]
        freqs = np.arange(constants.N_FREQS)

        waterfalls = []
        for delay in delays:

            w = np.exp(-2j * np.pi * (freqs * delay))
            waterfalls.append(w)

        predictor = VratioDelaySign(waterfalls)
        predictions = predictor.predict()

        np.testing.assert_array_equal(
            expected_predictions, predictions.astype(np.float)
        )


########################################################################################################
# test VratioDelayMagnitude
class test_VratioDelayMagnitude(unittest.TestCase):

    # test _convert_precictions
    def test_convert_predictions_conversion_fn_correct_string(self):

        conversion_fn = "Default"
        raw_predictions = np.arange(
            0,
            constants.MAX_EST_MAG + constants.ESTIMATE_WIDTH,
            constants.ESTIMATE_WIDTH,
        )[: constants.N_TIMES]
        _VratioDelayMagnitude = VratioDelayMagnitude(V_DATA, conversion_fn)
        self.assertRaises(
            ValueError, _VratioDelayMagnitude._convert_predictions, raw_predictions
        )

    def test_convert_predictions_conversion_fn_is_callable(self):

        conversion_fn = [1]
        raw_predictions = np.arange(
            0,
            constants.MAX_EST_MAG + constants.ESTIMATE_WIDTH,
            constants.ESTIMATE_WIDTH,
        )[: constants.N_TIMES]
        _VratioDelayMagnitude = VratioDelayMagnitude(V_DATA, conversion_fn)
        self.assertRaises(
            ValueError, _VratioDelayMagnitude._convert_predictions, raw_predictions
        )

    # test predict
    def test_network_model_exists(self):

        _VratioDelayMagnitude = VratioDelayMagnitude(V_DATA)
        _VratioDelayMagnitude._model_path = "not_a_model.pb"
        with self.assertRaises(IOError):
            _VratioDelayMagnitude.predict()

    # ???: How can I test this better?
    def test_predict_operates_without_failure(self):
        try:
            _VratioDelayMagnitude = VratioDelayMagnitude(V_DATA)
            predictions = _VratioDelayMagnitude.predict()
        except:
            self.fail("Prediction failed unexpectedly")

    def test_predict_produces_expected_output(self):

        expected_predictions = np.abs(EXPECTED_PREDICTIONS.reshape(10))
        delays = np.arange(-0.0400, 0.0400, 0.0001)[::80]
        freqs = np.arange(constants.N_FREQS)

        waterfalls = []
        for delay in delays:

            w = np.exp(-2j * np.pi * (freqs * delay))
            waterfalls.append(w)

        predictor = VratioDelayMagnitude(waterfalls, conversion_fn=None)
        predictions = predictor.predict()

        np.testing.assert_allclose(expected_predictions, predictions)


# test VratioDelay
class test_VratioDelay(unittest.TestCase):

    # test predict
    def test_predict_operates_without_failure(self):
        try:
            _VratioDelay = VratioDelay(V_DATA)
            predictions = _VratioDelay.predict()
        except:
            self.fail("Prediction failed unexpectedly")

    def test_predict_produces_expected_output(self):

        expected_predictions = EXPECTED_PREDICTIONS.reshape(10)
        delays = np.arange(-0.0400, 0.0400, 0.0001)[::80]
        freqs = np.arange(constants.N_FREQS)

        waterfalls = []
        for delay in delays:

            v = np.exp(-2j * np.pi * (freqs * delay))
            waterfalls.append(v)

        predictor = VratioDelay(waterfalls, conversion_fn=None)
        predictions = predictor.predict()

        np.testing.assert_allclose(expected_predictions, predictions)


########################################################################################################
# test DelaySolver
class test_DelaySolver(unittest.TestCase):

    # test __init__
    def test_init_list_o_sep_pairs_capture(self):

        list_o_sep_pairs = [[(0, 1), (2, 3)]]
        _DelaySolver = DelaySolver(list_o_sep_pairs, V_DATA)
        np.testing.assert_array_equal(_DelaySolver._list_o_sep_pairs, list_o_sep_pairs)

    # test __init__
    def test_init_list_o_sep_is_list_or_array(self):

        list_o_sep_pairs = set((((0, 1), (2, 3))))
        self.assertRaises(TypeError, DelaySolver, list_o_sep_pairs, V_DATA)

    def test_init_list_o_sep_pairs_shape_1_is_2(self):

        list_o_sep_pairs = [[(1, 2)]]  # shape = (1, 1, 2)
        self.assertRaises(ValueError, DelaySolver, list_o_sep_pairs, V_DATA)

    def test_init_list_o_sep_pairs_shape_2_is_2(self):

        list_o_sep_pairs = [[(1,), (2,)]]  # shape = (1, 2, 1)
        self.assertRaises(ValueError, DelaySolver, list_o_sep_pairs, V_DATA)

    def test_init_list_o_sep_pairs_dtype_is_int(self):

        list_o_sep_pairs = [[(1., 2), (3, 4)]]
        self.assertRaises(TypeError, DelaySolver, list_o_sep_pairs, V_DATA)

    # test true_b
    def test_true_b_true_delays_keys_equals_unique_ants(self):

        list_o_sep_pairs = [[(0, 1), (2, 3)]]
        _DelaySolver = DelaySolver(list_o_sep_pairs, V_DATA)
        true_ant_delays = {1: 1.2}
        self.assertRaises(AssertionError, _DelaySolver.true_b, true_ant_delays)

    # test predict
    def test_predict_operates_without_failure(self):
        try:
            list_o_sep_pairs = [[(0, 1), (2, 3)]]
            _DelaySolver = DelaySolver(list_o_sep_pairs, V_DATA)
            predictions = _DelaySolver.predict()
        except:
            self.fail("Prediction failed unexpectedly")


########################################################################################################
# test constants
class test_constants(unittest.TestCase):

    # test description
    def test_description_each_constant_has_key_in_dict(self):

        _constants = [
            item
            for item in dir(constants)
            if not (
                item.startswith("__")
                or item.startswith("desc")
                or item.startswith("print")
            )
        ]
        _description = constants.description()
        np.testing.assert_array_equal(sorted(_description.keys()), sorted(_constants))


if __name__ == "__main__":
    unittest.main()
