# M.Sc. Thesis: Study and analysis of training strategies to improve the reliability of artificial neural networks.

In this work we present a study and analysis of different methods to train Deep Neural Networks to improve their reliability. Specifically, we train a residual network of 18 layers (ResNet18) with different training parameters and optimizers to see how much the accuracy of these trained models decreases in the presence of faults. We perform fault injections at a software level using the PytorchFI library which works over the framework Pytorch. The Resnet18 was trained on an image recognition task with the CIFAR-10 dataset.

The most important result in this work is obtained at the moment of comparing reliability between optimizers. After selecting the models which gave the best reliability from the fault injection campaigns, we observe from the experiments that the optimizer which performs better in terms of reliability is SGD, followed by Adagrad, then Rmsprop and finally Adam.

![alt text](https://github.com/gaboceron10/MSc_Thesis_v2/blob/master/optmizer-comparison.png?raw=true)

Another experiment was done with the SGD optimizer, we used the fault model bit-flip to see if there is a bit of the weight (float-64) which is more sensible to this kind of fault.
The results show that bits 54 and 62 are very sensible to this fault and, as these bits determine the exponent of the floating-point number, they drop the accuracy significantly after the bit-flip.


![alt text](https://github.com/gaboceron10/MSc_Thesis_v2/blob/master/bit-flip-fault-injection.png?raw=true)

More tests and variations on training parameters where performed to assess reliability, see the full work for more results:
https://webthesis.biblio.polito.it/secure/21314/1/tesi.pdf
