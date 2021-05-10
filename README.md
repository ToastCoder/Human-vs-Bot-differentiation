# Human vs Bot differentiation

### About:

This code differentiates if a user is a human or a bot based on various metrics. Implemented with TensorFlow.

The model is trained with some parameters which are helpful in identifying difference if a logged in user is human or a bot in train.py. The code in the file test.py collects various parameters from the user and predicts accordingly.

Model accuracy of 99% is achieved while testing.

### Supported Operating Systems: 
Runs on Windows, macOS and Linux

### Tested with: 
1. Python 3.8.6 64-bit
2. TensorFlow 2.4.1
3. Pop OS 20.10

### Developed by: 
[Vigneshwar Ravichandar](https://github.com/ToastCoder)

### List of Features: 
1. Check Status 
2. Captcha Attempts
3. Number of Login Attempts 
4. Average Time between the attempts 

### List of Classes:
1. Human (Class [0]) 
2. Bot (Class [1]) 

### Execution Instructions:  
Execute the following command in the terminal to run with default procedure.  
```python
python3 main.py --test=True
```

### Command Line Arguments:
* `-tr` (or) `--train` - Used to train the Neural Network.  
  * **Argument type:** bool  
  * **Parameter type:** Optional  
  * **Default value:** False

* `-t` (or) `--test` - Used to test the Neural Network with custom inputs.
  * **Argument type:** bool  
  * **Parameter type:** Mandatory 
  
* `-v` (or) `--visualize` - Used to vizualize the metrics.
  * **Argument type:** bool  
  * **Parameter type:** Optional
  * **Default value:** False
  
* `-req` (or) `--install_requirements` - Used to install the required dependancies.
  * **Argument type:** bool  
  * **Parameter type:** Optional  
  * **Default value:** False

* `-e` (or) `--epochs` - Used for mentioning the number of epochs for the model.
  * **Argument type:** int
  * **Parameter type:** Optional
  * **Default value:** 10

* `-bs` (or) `--batch_size` - Used for mentioning the batch size for the model.
  * **Argument type:** int
  * **Parameter type:** Optional
  * **Default value:** 5

* `-l` (or) `--loss` - Used for mentioning the loss function for the model.
  * **Argument type:** str
  * **Parameter type:** Optional
  * **Default value:** "sparse_categorical_crossentropy"

* `-op` (or) `--optimizer` - Used for mentioning the optimizer for the model.
  * **Argument type:** str
  * **Parameter type:** Optional
  * **Default value:** "adam"

