# Implementation Details

## class EncoderLSTM

```__init__(self, batch_size, units)```

* ```batch_size```: Input batch size

* ```units```: The number of units to use for the encoder state and hidden layers.

Description: Initializes the EncoderLSTM layer with the appropriate batch size and the number of units to use.

```build(self, input_shape)```

* ```input_shape```: The input shape of the inputs. In this instance, it is $[batch\_size, sequence\_length, embedding\_dim]$ where:
	* ```batch_size```: is the input batch size

