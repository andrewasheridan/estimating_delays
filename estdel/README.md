# estdel
Estimating interferometer antenna cable delays
## Prereqs

```
numpy
tensorflow
``` 
## Description

estdel estimates the antenna cable delays in interferometer visibility data. 

Cable delays can be represented as the slope of the phase angle of complex data. For some data that has the general form exp(-2&pi;i(&nu;&tau; + &phi;)), the value &tau; is the slope we are looking for, and the value &nu; is the range of frequency channels for our data.

Complex visibility data is converted to angle and fed into two pretrained neural networks. One network predicts if the slope of the phase angle is positive or negative. Another network predicts the magnitude of the phase angle. These predictions are multiplied toether to produce an estimate of the delay. 


