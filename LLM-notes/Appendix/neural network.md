
![[Pasted image 20260316093230.png]]

## Overview
#### Neutrons
*neurons don't think or really make any decisions*

rather: 
>neuron_output= activation(weighted_sum(inputs) + bias)
- based of the inputs and bias the activation would fire or not. *"is the total high enough to matter?"* 
- it is the cause of apply the **same** equation again and again all the while adjusting on the feedback. 

-------------
#### Layers 
*Zooming out to layers*

Neurons on the **same** layer receive the **same** inputs however each has their **own weights** and personal bias. 

*similar to asking a question to 6 neurons in parallel which passes 6 answers down to next layer*

there are also bias units that are contributed to the next layer seen without inputs

----
#### Network
*layers combined together*

The chaining of layers and exchange of inputs and outputs is the result of a Neural Network. Each neuron slowly nudging towards the correct answer. 

The natural question is how to know what those neuron values should be: 
*Backpropagation*

[[Pytorch]]'s [[Automatic Differentiation]] calculates the gradient for every single neuron's weight and bias. This gives us how much that specific neuron should adjust to decrease loss. 

**Forward -> Loss -> Backward -> Optimize** 

## Code understanding 





