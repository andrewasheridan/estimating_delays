from __future__ import print_function

# number of rows (times) in visibility 
N_TIMES = 60

# number of frequency channels
N_FREQS = 1024

# For default conversion function
# minimum frequency
MIN_FREQ_GHZ = 0.100

# For default conversion function
# maximum frequency
MAX_FREQ_GHZ = 0.200

# minimum magnitude of estimate (scaled down)
MIN_EST_MAG = 0.0000

# maximum magnitude of estimate (scaled down)
MAX_EST_MAG = 0.0400

# width of magnitude classes (scaled down)
ESTIMATE_WIDTH = 0.0001 

# See README
def description():
	return {
		
		'N_FREQS' : "To provide the ability to make predictions for data with different frequency ranges, " \
		+ "during network training the frequency range was a unitless range of {} unit-width frequency channels:".format(N_FREQS) \
		+ " freqs = np.arange({}). ".format(N_FREQS),

		# ???: How can I phrase this better?
		'N_TIMES' : "Each visibility is expected to have {} unique measurements".format(N_TIMES),

		'MIN_EST_MAG' : "To accomodate the unitless unit-width frequency channels," \
		+ " raw predictions of the magnitude of delay have a minimum value of {}".format(MIN_EST_MAG),

		'MAX_EST_MAG' : "To accomodate the unitless unit-width frequency channels," \
		+ " raw predictions of the magnitude of delay have a maximum value of {}".format(MAX_EST_MAG),

		'ESTIMATE_WIDTH' : "The minimum diference between raw predictions is {}".format(ESTIMATE_WIDTH),

		'MIN_FREQ_GHZ' : "When using the default conversion function the minimum frequency is {}".format(MIN_FREQ_GHZ),

		'MAX_FREQ_GHZ' : "When using the default conversion function the maximum frequency is {}".format(MAX_FREQ_GHZ),	
	}	