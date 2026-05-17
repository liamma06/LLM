GELU(gaussian error linear unit) like ReLU is an [[activation function]].

*Why not stick with RELU?*

Relu is a simple sharp filter -> if negative it become 0
however by "killing" by returning output of zero the gradients fail to flow during backpropagation and weights are not updated for that neuron meaning part of the network become inactive therefore unable to learn further. 

$$GELU(x) \approx 0.5 \times x \times \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \times (x + 0.044715 \times x^3)\right)\right)$$
By using GELU, it allows a small amount of negative information to leak keeping the gradients alive and allow model to learn more complex patterns. 
![[Pasted image 20260517114116.png]]