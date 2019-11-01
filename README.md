# L_layered-Deep-Neural-Network-Implementation
## 1. Description
  This notebook includes implementation of the DNN with L-layered architecture for binary classification and tested with iris dataset. You can use this notebook as a L-layered DNN model to train different datasets and different architectures. You have to fit train dataset of shape (dimensions, examples) and target output set of (dimensions,examples) to the model and parameters will be resulted. Hyperparameters are also there to tune as your preferrence and predictions can be made. 

![deep neural network](https://github.com/Christopher-K-Constantine/Deep-Neural-Network-from-scratch/blob/master/dnn.jpeg)

## 2. Dependencies
1. Numpy
2. Pandas
3. Scikit-learn
4. Scipy
5. Matplotlib

## 3. Usage
1. Download the repo
2. Open the 'Binary-classification DNN' ipython notebook with Jupyter notebook
3. Run the notebook cells step-by-step

## 4. Hyperparameters to tune
1. layer_dims: Dimensions of your network (according to your dataset)
2. num_epochs: Number of epochs to train
3. learning_rate: Learning rate alpha for gradient descent
4. optimizer: Optimization method ("gd"- minibatch gradient descent, "adam"- Adam optimizer)
5. lambd: Regularization parameter
6. mini_batch_size: Size of minibatch for minibatch gradient descent
7. beta1, beta2: Hyperparameters to adjust Adam optimizer's momentum
8. epsilon: Hyperparameter to include in the updating function of the model parameters

## Reference: Deeplearning.ai's deep learning specialization course on coursera

 
  

