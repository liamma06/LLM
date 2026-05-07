![[Pasted image 20260501214415.png|517]]
![[Pasted image 20260501214541.png|515]]

### Why is attention mechanism needed? 
consider if we make a language translation model you can't just translate word by word as it missing the grammatical context
-previous text influences the meaning of other text where was embedding would be the same regardless of positioning therefore with context form other words it allows a better representation of the text .
![[Pasted image 20260501225409.png|491]]![[Pasted image 20260502113007.png|530]]
RNNS have limitations. They create a summary "hidden state" as it moves to second word previous hidden state is used to create a new one. This squeezes entire information causing potential information loss. 

![[Pasted image 20260502152358.png]]
The Bahdanau attention mechanism selectively access the encoder state for each word. It also assigns a weight to each previous word for the more "relevant" information

### What does "self" mean in self-attention
refers to the sequence looking at itself to compute weights by relating different positions within itself. Essentially finding relationships within itself versus against another sequence.  What is a relationship? well there are the weights/scores that tell the model how much attention to pay to other parts of the input. 

### Mechanics of attention
![[Pasted image 20260504163511.png]]

1. Calculating the attention score 
This model computes the dot products between token embeddings. This tells how related these token are and quantifies the relevance score between the embeddings. 
![[Pasted image 20260506074335.png]]

2. Convert previous attention score to attention weight
Once the attention score is computed via dot product we would normalize using softmax. This converts the raw numbers into probabilities better represents the impact on chosen query/word. 

3. Compute context vector 
Essentially doing a weighted sum of the token embedding with the attention weight and getting a final vector that includes the context information. ![[Pasted image 20260507102643.png]]

### Computing attention weights for all input tokens 