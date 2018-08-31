# estdel
[![Build Status](https://travis-ci.org/andrewasheridan/estimating_delays.svg?branch=test_predict)](https://travis-ci.org/andrewasheridan/estimating_delays)
[![Coverage Status](https://coveralls.io/repos/github/andrewasheridan/estimating_delays/badge.svg?branch=test_predict)](https://coveralls.io/github/andrewasheridan/estimating_delays?branch=test_predict)

Estimating interferometer antenna cable delays
## Prereqs
```
numpy >= 1.14.5 # possible tensorflow bug with numpy 1.15.0
tensorflow >= 1.8.0 # developed on 1.8, no known issues with 1.9 or 1.10
```
### Python 2.7, 3.5

Developement and initial evaluation was on Python 2.7. Limited testing on python 3.5 has not revealed any issues thus far.

## Installation

Clone git repo and navigate to directory and run `python setup.py install`

## Description

estdel estimates the antenna cable delays in interferometer waterfall data. 

Complex waterfall data is converted to angle and fed into two pretrained neural networks. One network predicts if the slope of the phase angle is positive or negative. Another network predicts the magnitude of the phase angle. These predictions are multiplied together to produce an estimate of the delay. Predictions are provided as raw predictions, unitless predictions directly from the networks, or predictions with units.  

## Prediction Details

Cable delays can be represented as the slope of the phase angle of complex data. For some data that has the general form exp(-2&pi;i(&nu;&tau; + &phi;)), the value &tau; is the slope we seek, and the value &nu; is the range of frequency channels for the data.

To provide the ability to make predictions for data with different frequency ranges, the value &nu; used during network training was a unitless range of 1024 unit-width frequency channels: `freqs = np.arange(1024)`. 

To accomodate the unitless unit-width frequency channels, raw predictions of the magnitude of delay fall in the range of 0.0000 to 0.0400, in increments of 0.0001. When taking into account sign, `raw_predictions` will have values in the range -0.0400 to +0.0400.

### Conversion of unitless predictions to predictions with units

If `conversion_fn` is set to `None`, then `predictions` is the same as `raw_predictions`, as described above.

If `conversion_fn` is set to `'default'`, then the frequency range &nu; is taken to be 1024 equal width frequency channels over a total of 0.100 GHz:
`freqs = np.linspace(0.100, 0.200, 1024) # GHz`. 

Each channel then is approximately 0.098 MHz wide. This has the effect of changing the smallest increment of delay from a unitless value of 0.0001 to about 1.023 nanoseconds, and then `predictions` will contain delay values in the range of about -409 ns to +409 ns.

If you would like to use a different range of frequency channels of equal width, pass your own conversion function to `conversion_fn`, `predictions` will then have values in the commensurate range.

## Usage
In general:
 - Pass waterfall data to one of the predictor objects 
 - Call its `predict` method:

```
import estdel

# pass complex waterfalls to VratioDelay
predictor = estdel.VratioDelay(waterfalls)

# call predict
predictions = predictor.predict()
```

`predictions` is a list of floats of the delays. If each waterfall has `num_times` rows, then we can print the mean delay of each waterfall like so:

```
# mean value for each waterfall
for i in range(num_waterfalls):
    mean_value = np.mean(predictions[i * num_times : (i + 1) * num_times])
    print("{:7.2f} ns".format(mean_value))
```

Which produces the output:
```
   9.21 ns
-274.16 ns
 352.94 ns
 -69.56 ns
 360.10 ns
 247.57 ns
 -32.74 ns
 150.38 ns
 409.20 ns
 192.32 ns
```

## Processing Time

### GPU

If your system has more than one GPU make sure to specifiy a single GPU for tensorflow to use, otherwise tensorflow will claim all memory on all devices.
```
# do this before importing estdel
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
```

On a system with a 24 core Intel Xeon CPU  @ 2.30GHz (Turbo 2.8), 128GB RAM, and an Nvidia GeForce GTX TITAN w/ 6GB RAM:

Processing waterfalls that are 60x1024:
 -   10 waterfalls ~4 seconds
 -  100 waterfalls ~7 seconds
 -  500 waterfalls ~14 seconds
 - More than ~600 causes an out of memory exception :( 

Using an Nvidia GeForce GTX 1080 w/ 8GB RAM produces approximately the same results.

### CPU only
On the Xeon system with the GPUs disabled, processing 500 waterfalls takes ~50 seconds.

On a 2015 12" Macbook with a 2 core  Intel Core M  @ 1.1 GHz (Turbo 1.9), 8GB RAM, and with no GPU, processing 100 waterfalls takes ~1 minute.