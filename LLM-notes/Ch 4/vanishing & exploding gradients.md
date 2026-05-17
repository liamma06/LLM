understanding that gradients tells the model how much and what direction to adjust the weights when training. 

however if the gradients become extremely *small* doing backpropagation it causes the layers to learn very slowly. Due to the fact of chain rule and repeated multiplication though layers the gradient vanishes.  

if the gradient grows too large, weights become unstable with chaotic updates and in return breaks 