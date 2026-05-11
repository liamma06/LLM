A single attention head has one set of weights (Wq, Wk, Wv) which mean it can really just focus on **one** type of relationship. By having multiple heads, the model can capture all diverse patterns at once become much better for crucial/sophisticated relationship recognition. 

### Methodology 

##### *Simple*
![[Pasted image 20260511102513.png]]
Every head would process independently to create it's own context vectors which would be concatenated into a final context vector. 

##### *Optimized: weight splitting*
In the simple implementation with the for loops this is a sequential process (Head 1 has to finish before Head 2 and ...)
Where instead of individual proj for every head it creates a larger proj that is further split to produce those heads which allows parallel computation.

1. Large projection
A single large input embedding which can be divisible to fit heads x head_dim

2. reshape
The tensor is reshaped with view and transpose to separate into the heads

3. attention calculation 
like previously but now  parallelly 

4. concatenate 
output from head are combined ![[Pasted image 20260511120911.png]]

I often had to reference this video! as it was super difficult to visualize what was going on 
[Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) 