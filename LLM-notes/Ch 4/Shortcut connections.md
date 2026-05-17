![[Pasted image 20260517123355.png]]

#### *Why?*
currently: 
- forward Layer 1 $\rightarrow$ Layer 2 $\rightarrow$ Layer 3 $\rightarrow$ Layer 4 $\rightarrow$ Layer 5 $\rightarrow$ Prediction
- model looks at prediction, compares to actual answer, calculate error scores 
- back propagation 

In back propagation, the model sends gradient backwards from end to front
```
[Layer 1] ◄── [Layer 2] ◄── [Layer 3] ◄── [Layer 4] ◄── [Layer 5] ◄── [Loss/Error]
```
gradient of early layer requires multiplying the changes of the layers that came after it (chain rule).

*Example :
- **Layer 5** receives the original error signal: `1.0`
    
- **Layer 4** multiplies it: $1.0 \times 0.2 = \mathbf{0.2}$
    
- **Layer 3** multiplies it again: $0.2 \times 0.2 = \mathbf{0.04}$
    
- **Layer 2** multiplies it again: $0.04 \times 0.2 = \mathbf{0.008}$
    
- **Layer 1** multiplies it a final time: $0.008 \times 0.2 = \mathbf{0.0016}$

By the time gradient reaches layer 1 it is almost zero. With such a low gradient the weights for layer 1 barely changes where it is essentially stuck in an untrained state. 

#### Solution
The gradient where given a eq Y = x + F(x) and you look at how a change in Y affects x and add it on:

$$\text{Gradient} = 1 + \text{Gradient of Layer}$$
*Backpropagation with short cut*:
1. gradient travel backwards and hit the exit gate of layer.

2.  the addition operation (+), the error signal splits :
	- path 1: it goes through the layer's complex math where a tiny decimal (same as without shortcut)
	- path 2: it completely bypasses layer because addition node, it passes with full strength
3. at the layer's entry gate the shrunken signal from path 1 and full-strength signal from path 2 are added 

Basically total gradient is gradient from that layer plus perfect copy of incoming gradient 