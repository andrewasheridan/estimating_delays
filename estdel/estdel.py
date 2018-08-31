"""estdel - for estimating delays
 - Andrew Sheridan sheridan@berkeley.edu

Estimate the overall cable delay in waterfalls of visibility ratios.

For each list of waterfalls of 60 x 1024 complex visibility ratios produce 60 estimated cable delays.

Passes each row through one of (or both of) two trained neural networks. 

'Sign' network classifies the sign of the delay as being positive or negative.
'Magnitude' network classfies the magnitude of the delay as being in one of 401 classes.
    -   401 classes from 0.00 to 0.0400, each of width 0.0001
    -   default conversion to 401 classes from 0 to ~ 400 ns

DelaySign estimates the sign of the delay (< 0 or >= 0)
DelayMagnitude estimates the magnitude of the delay
Delay provides Delay_Sign * Delay_Magnitude

tau = # slope in range -0.0400 to 0.0400
nu = np.arange(1024) # unitless frequency channels
phi = # phase in range 0 to 1
data = np.exp(-2j*np.pi*(nu*tau + phi))

estimator = estdel.Delay(data)
prediction = estimator.predict()

# prediction should output tau
"""
# from __future__ import absolute_import
import numpy as np
import pkg_resources
import tensorflow as tf

try:
    import constants as constants
except:
    import estdel.constants as constants
# from . import constants as constants
# import estdel.constants as constants
# import constants
# suppress tensorflow INFO messages
tf.logging.set_verbosity(tf.logging.WARN)

# ???: Should i move these to constants.py?
_TRAINED_MODELS_DIR = "trained_models"

# fn for best postive-negative classifier
_SIGN_PATH = "sign_NN_frozen.pb"

# fn for best magnitude classifier
_MAG_PATH = "mag_NN_frozen.pb"


class _DelayPredict(object):
    """_DelayPredict
    
    Handles data processing and prediction.
    
    """

    def __init__(self, data):
        """__init__
        
        Args:
            data (numpy array of complex or list of complex): Visibliity data
        """

        if type(data) is np.ndarray:
            self._data = data
        elif type(data) is list:
            self._data = np.array(data)
        else:
            raise TypeError(
                "data should be list or numpy array not {}".format(type(data))
            )

        if np.iscomplexobj(self._data) is not True:
            raise TypeError("data must be complex")

        if self._data.shape[-1] == constants.N_FREQS:
            self.data = self._angle_tx(np.angle(self._data)).reshape(
                -1, 1, constants.N_FREQS, 1
            )
        else:
            raise ValueError(
                "last dim in data shape must be {} not {}".format(
                    constants.N_FREQS, self._data.shape[-1]
                )
            )

    def _angle_tx(self, x):
        """_angle_tx
        
        Scales (-pi, pi) to (0, 1)
        
        Args:
            x (numpy array of floats): Angle data in range -pi to pi
        
        Returns:
            numpy array of floats: scaled angle data
        """
        tx = (x + np.pi) / (2. * np.pi)

        # ???: I dont see how these checks could ever possibly fire
        # ... x here is passed data from np.angle()...
        if np.min(tx) < 0:
            raise ValueError(
                "Scaled angle data out of range, check that np.angle(data) is in range -pi to pi"
            )
        if np.max(tx) > 1:
            raise ValueError(
                "Scaled angle data out of range, check that np.angle(data) is in range -pi to pi"
            )

        return tx

    def _predict(self):
        """_predict
        
        Import frozen tensorflow network, activate graph, 
        feed data, and make prediction.
        
        """

        resource_package = __name__
        resource_path = "/".join((_TRAINED_MODELS_DIR, self._model_path))
        if pkg_resources.resource_exists(resource_package, resource_path) is True:
            path = pkg_resources.resource_filename(resource_package, resource_path)
        else:
            raise IOError("Network file '{}'' not found".format(self._model_path))

        with tf.gfile.GFile(path, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                restored_graph_def, input_map=None, return_elements=None, name=""
            )

            sample_keep_prob = graph.get_tensor_by_name("keep_probs/sample_keep_prob:0")
            conv_keep_prob = graph.get_tensor_by_name("keep_probs/conv_keep_prob:0")
            is_training = graph.get_tensor_by_name("is_training:0")
            X = graph.get_tensor_by_name("sample/X:0")

            # add hook to output operation
            pred_cls = graph.get_tensor_by_name("predictions/ArgMax:0")

        with tf.Session(graph=graph) as sess:
            feed_dict = {
                sample_keep_prob: 1.,
                conv_keep_prob: 1.,
                is_training: False,
                X: self.data,
            }

            # collect prediction
            self._pred_cls = sess.run(pred_cls, feed_dict=feed_dict)

            sess.close()


class DelaySign(_DelayPredict):
    """DelaySign
    
    Estimates waterfall cable delay sign by using two pretrained neural networks.
    
    Methods:
        predict()
            - call to make sign prediction for data
    
    Attributes:
        data (numpy array of floats): Input data of waterfalls of redundant visibility ratios is processed for predictions
        predictions (numpy array of floats): The converted raw magnitude predictions (see predict())
        raw_predictions (list of floats): The raw magnitude predictions from the network
    """

    def __init__(self, data):
        """__init__
        
        Preprocesses data for prediction.
        
            - converts complex data to angle
            - scales angles to range preferred by networks
            - reshapes 2D data to 4D tensor
        
        
        Args:
            data (list of complex or numpy array of complex): shape = (N, 1024)
                - Input data of waterfalls of redundant visibility ratios is processed for predictions
        
        """
        _DelayPredict.__init__(self, data=data)

        self._model_path = _SIGN_PATH

    def _pred_cls_to_sign(self):
        """_pred_cls_to_sign
        
            Convert index of predicted class index to value
              1 if class is postive
             -1 if class is negative
        
        Returns:
            list of ints: 
        """
        return [1 if x == 0 else -1 for x in self._pred_cls]

    def predict(self):
        """predict
        
        Returns:
            numpy array of floats: sign predictions
        """
        self._predict()
        # ???: Is it confusing to have raw_predcitions here when they are the same as predictions
        #       - it follows the same pattern as mag predictor
        self.raw_predictions = self._pred_cls_to_sign()
        self.predictions = np.array(self.raw_predictions)

        return self.predictions


def _default_conversion_fn(x):
    """_default_conversion_fn
    
    Convert unitless predictions to nanoseconds
    Based on 1024 channels and 
    0.100 GHz - 0.200 GHz range
    
    Args:
        x (numpy array of floats): Predicted values in range 0.000 to 0.040
    
    Returns:
        numpy array of floats: Converted predicted value
    """

    freqs = np.linspace(
        constants.MIN_FREQ_GHZ, constants.MAX_FREQ_GHZ, constants.N_FREQS
    )
    channel_width_in_GHz = np.mean(np.diff(freqs))

    return x / channel_width_in_GHz


class DelayMagnitude(_DelayPredict):
    """DelayMagnitude
    
    Estimates watefall total cable delay by using two pretrained neural networks.
    
    Methods:
        predict()
            - call to make prediction
    
    Attributes:
        data (numpy array of complex or list of complex): Input data of waterfalls of redundant visibility ratios is processed for predictions
        predictions (numpy array of floats): The converted raw magnitude predictions (see predict())
        raw_predictions (list of floats): The raw magnitude predictions from the network
    """

    def __init__(self, data, conversion_fn="default"):
        """Preprocesses data for prediction.
        
            - converts complex data to angle
            - scales angles to range preferred by networks
            - reshapes 2D data to 4D tensor
        
        
        Args:
            data (numpy array of complex or list of complex): shape = (N, 1024)
                - Input data of waterfalls of redundant visibility ratios is processed for predictions
            conversion_fn (None, str, or function):
                - None - Do no conversion, output predictions are the raw predictions
                - 'default' - convert raw predictions to ns by using frequencies 
                    with a 0.100 GHz range over 1024 channels
                - function - provide your own function to do the conversion
                    - one required argument, the raw predictions, one output, the predictions
        
        """
        _DelayPredict.__init__(self, data=data)

        self._model_path = _MAG_PATH
        self._conversion_fn = conversion_fn

    def _pred_cls_to_magnitude(self):
        """_pred_cls_to_magnitude
        
        Convert predicted label index to magnitude
        
        Returns:
            list of floats: Magnitudes
        """
        magnitudes = np.arange(
            constants.MIN_EST_MAG,
            constants.MAX_EST_MAG + constants.ESTIMATE_WIDTH,
            constants.ESTIMATE_WIDTH,
        )
        return [magnitudes[x] for x in self._pred_cls]

    def _convert_predictions(self, raw_predictions):
        # pass through if conversion_fn is none else convert

        if self._conversion_fn is None:
            return raw_predictions

        elif type(self._conversion_fn) is str:
            if self._conversion_fn is "default":
                return _default_conversion_fn(raw_predictions)
            else:
                raise ValueError(
                    "conversion_fn must be a callable function, None, or 'default' not '{}".format(
                        self._conversion_fn
                    )
                )

        elif callable(self._conversion_fn) is True:
            return self._conversion_fn(raw_predictions)
        else:
            raise ValueError(
                "conversion_fn must be a callable function, None, or 'default'"
            )

    def predict(self):
        """predict
        
        Returns:
            numpy array of floats or list of floats: predictions
        """

        self._predict()
        # ???: Is it desireable to have the raw_predictions available as well as the predictions?
        self.raw_predictions = self._pred_cls_to_magnitude()
        self.predictions = self._convert_predictions(self.raw_predictions)

        return np.array(self.predictions)


class Delay(object):
    """Delay
    
    Estimates waterfall total cable delay by using two pretrained neural networks.
    
    Methods:
        predict()
            - call to make prediction
    
    Attributes:
        raw_predictions (list of floats): The raw predictions from the network
        predictions (numpy array of floats or list of floats) = The converted raw predictions

    """

    def __init__(self, data, conversion_fn="default"):
        """__init__
        
        Preprocesses data for prediction.
        
            - converts complex data to angle
            - scales angles to range preferred by networks
            - reshapes 2D data to 4D tensor
        
        
        Args:
            data (list of complex floats): shape = (N, 1024)
                - Input data of waterfalls of redundant visibility ratios is processed for predictions
        
        """

        self._mag_evaluator = DelayMagnitude(data, conversion_fn=conversion_fn)
        self._sign_evaluator = DelaySign(data)

    def predict(self):
        """predict
        
            Make predictions
        
        Returns:
            numpy array of floats or list of floats: Predicted values
        """
        signs = self._sign_evaluator.predict()
        mags = self._mag_evaluator.predict()

        self.raw_predictions = [
            self._mag_evaluator.raw_predictions[i] * signs[i] for i in range(len(signs))
        ]
        self.predictions = signs * mags

        return self.predictions


class DelaySolver(object):
    """DelaySolver
    
    Args:
        list_o_sep_pairs (list of lists of tuples): shape = (N, 2, 2)
            ex: list_o_sep_pairs[0] = [(1, 2), (3, 4)]
            each length 2 sublist is made of two redundant separations, one in each tuple
    
        data (numpy array of complex or list of complex): shape = (N, 60, 1024) or (N * 60, 1024)
            # ???: should we optionally allow less or more than 60 rows per visibility?
            Watefalls made from the the corresponding redundant sep pairs in list_o_sep_pairs
    
        true_ant_delays (dict): dict of delays with antennas as keys,
            ex : true_ant_delays[143] = 1.2
            if conversion_fn == 'default', ant delays should be in ns
    
    Attributes:
        A (numpy array of ints): The matrix representing the watefalls
        b (numpy array of floats): A times x
        unique_ants (numpy array of ints): All the unique antennas in list_o_sep_pairs
        v_ratio_row_predictions (numpy array of floats or list of floats): Predicted values
        v_ratio_row_predictions_raw (list of floats): Predicted values with no conversion
        x (list floats): True delays in order of antenna
    """

    def __init__(
        self, list_o_sep_pairs, data, conversion_fn="default"  # shape = (N, 2, 2)
    ):
        """__init__

        Preprocess data, make predictions, covert data to ns, 
        construct A matrix.

        """
        if type(list_o_sep_pairs) is list:
            self._list_o_sep_pairs = np.array(list_o_sep_pairs)
        elif type(list_o_sep_pairs) is np.ndarray:
            self._list_o_sep_pairs = list_o_sep_pairs
        else:
            raise TypeError(
                "list_o_sep_pairs must be list or numpy array, not {}".format(
                    type(list_o_sep_pairs)
                )
            )

        if self._list_o_sep_pairs.shape[1] != 2:
            raise ValueError(
                "Each sublist must have len = 2 not {}".format(
                    self._list_o_sep_pairs.shape[1]
                )
            )
        if self._list_o_sep_pairs.shape[2] != 2:
            raise ValueError(
                "Each subsublist must have len = 2 not {}".format(
                    self._list_o_sep_pairs.shape[2]
                )
            )
        if np.issubdtype(self._list_o_sep_pairs.dtype, np.integer) is not True:
            raise TypeError(
                "Each subsublist must have elements with type like int not {}".format(
                    self._list_o_sep_pairs.dtype
                )
            )

        self.unique_ants = np.unique(self._list_o_sep_pairs)

        self._make_A_from_list_o_sep_pairs()
        # ???: Should the creation of A and b be its own object?

        self._predictor = Delay(data, conversion_fn=conversion_fn)

    def predict(self):

        self._predictor.predict()
        self.v_ratio_row_predictions = self._predictor.predictions
        self.v_ratio_row_predictions_raw = self._predictor.raw_predictions

    def _get_A_row(self, sep_pair):
        """_get_A_row

        constructs a single row of A from a sep pair

        Args:
            sep_pair (numpy array of numpy arrays ints): Antennas for this waterfall

        Returns;
            list of ints: a row of A
    
        """

        a = sep_pair.flatten()

        # construct the row
        # XXX: there must be a better way
        # https://stackoverflow.com/a/29831596

        # row is 4 x _max_ant_idx, all zeros
        # ???: a.size is always 4 (checked indirectly in init, can I just replace a.size with 4?
        row = np.zeros((a.size, self._max_ant_idx), dtype=int)

        # for each element in sep_pair, got to the corresponding row
        # and assign the corresponding antenna the value 1
        row[np.arange(a.size), a] = 1

        # flip the sign of the middle two rows
        row[1] *= -1
        row[2] *= -1

        # add the rows, row is now 1 x _max_ant_idx
        row = np.sum(row, axis=0)

        return row

    def _make_A_from_list_o_sep_pairs(self):
        """_make_A_from_list_o_sep_pairs

        Construct A row by row
        """

        # _max_ant_idx is used to set the numberof columns in the matrix A
        # There should be a column for each antenna
        self._max_ant_idx = np.max(self.unique_ants)

        # In case the antenna indexed zero is inlcuded in list_o_sep_pairs
        # if (0 in self.unique_ants) is True:
        self._max_ant_idx = self._max_ant_idx + 1

        self.A = []
        for sep_pair in self._list_o_sep_pairs:

            # each waterfall of height 60 has one sep_pair
            # so make 60 identical rows in A for each waterfall
            # so that A is the correct shape
            # (because the prediction will output a unique prediction
            # for each row in the waterfall)
            # ???: Is there a better way to do this
            self.A.append(np.tile(self._get_A_row(sep_pair), (constants.N_TIMES, 1)))

        self.A = np.asarray(self.A).reshape(-1, self._max_ant_idx)

    def true_b(self, true_ant_delays):
        """ true_b

        do A times x to find the true values for each waterall
        where x is a list of the true antenna delays in order of antenna

        Returns:
            numpy array of floats
        """

        # ???: Is there a better way to do this check?
        # Every antenna present in unique antennas has to also be oresent in true antenna delays
        assert (
            np.array(sorted(true_ant_delays.keys())).all() == self.unique_ants.all()
        ), "Each antenna present in a waterfall needs a true delay here"

        self.x = [true_ant_delays[ant] for ant in self.unique_ants]
        self.b = np.matmul(self.A[:, self.unique_ants], self.x)

        return self.b
