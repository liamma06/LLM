Auto [[Differentiate]] is the process that [[Pytorch]] has 

[[Pytorch]] can internally builds [[Computational Graphs]] to compute gradients. These gradients would be used to train [[neural network]]s  with backpropagation. 

__________________________________

![[Pasted image 20260315135114.png|480]]

Backpropagation is an implementation of the [[chain rule]]. To know the loss effect you would have to multiply through every step in the chain which backpropagation does (right to left) .   

Summary: 
*starts from output layer/loss and work backwards through the chain/network to the input* 

_____________________________________________________________________ 

#### Gradient is a collection of slope. 

Partial Derivative seen is the slope for **one single variable** where all other variables is frozen. *how much does this w1/b alone effect lose*

The collection of all the partial derivative gives us the gradient. 

**Autograd** ties it all together: 
- it watches every operation on tensors
- builds the computation graph 
- When you call .grad it computes the gradient using chain rule
- we know know the gradient of loss concerning w1 

example: pytorch\ComputingWithAutoGrad.ipynb



