# estdel
Estimating interferometer antenna cable delays
## Prereqs

```
numpy
tensorflow
``` 
## Description

estdel estimates the antenna cable delays in interferometer visibility data. 

Cable delays can be represented as the slope of the phase angle of complex data.
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\exp{\big(-2\pi&space;i&space;\cdot&space;(\nu\tau&space;&plus;&space;\phi)\big)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\fn_cm&space;\exp{\big(-2\pi&space;i&space;\cdot&space;(\nu\tau&space;&plus;&space;\phi)\big)}" title="\exp{\big(-2\pi i \cdot (\nu\tau + \phi)\big)}" /></a>

Complex visibility data is converted to angle and fed into two pretrained neural networks. One network predicts if the slope of the phase angle is positive or negative. Another network predicts the magnitude of the phase angle. These predictions are multiplied toether to produce an estimate of the delay. 
\text{data} = \exp{\big(-2\pi i \cdot (\nu\tau + \phi)\big)}

