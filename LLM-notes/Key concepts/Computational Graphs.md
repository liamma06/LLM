This helps us understand how [[Pytorch]]'s **Automatic [[Differentiate]] Engine** actually does it job. 

*A sequence of calculations needed to compute the output of a [[neural network]]* 
which is mainly done though backpropagation (figuring out those gradients) which then uses gradient descent to update. 

References the BasicComputationalGraph.ipynb: 
![[Pasted image 20260315132642.png]]
Pytorch in the background helps build maps like these so the model can rewind and see how much w1 and b (the training units) need to be changed to reduce loss. 


