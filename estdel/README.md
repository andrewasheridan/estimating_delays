# estdel
[![Build Status](https://travis-ci.org/andrewasheridan/estimating_delays.svg?branch=travis_coveralls)](https://travis-ci.org/andrewasheridan/estimating_delays)
[![Coverage Status](https://coveralls.io/repos/github/andrewasheridan/estimating_delays/badge.svg?branch=travis_coveralls)](https://coveralls.io/github/andrewasheridan/estimating_delays?branch=travis_coveralls)

Estimating interferometer antenna cable delays
## Prereqs
```
numpy >= 1.14.5 # possible tensorflow bug with numpy 1.15.0
tensorflow >= 1.8.0 # developed on 1.8, no known issues with 1.9 or 1.10
```
### Python 2.7, 3.5

Developement and initial evaluation was on Python 2.7. Limited testing on python 3.5 has not revealed any issues thus far.

## Installatio

Clone git repo and navigate to directory and run `python setup.py install`

## Description

estdel estimates the antenna cable delays in interferometer visibility data. 

Complex visibility data is converted to angle and fed into two pretrained neural networks. One network predicts if the slope of the phase angle is positive or negative. Another network predicts the magnitude of the phase angle. These predictions are multiplied together to produce an estimate of the delay. Predictions are provided as raw predictions, unitless predictions directly from the networks, or predictions with units.  

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




