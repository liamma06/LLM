In a typical self-attention mechanism each token gather info from all the other tokens. However allow the token to take in consideration future tokens breaks the idea of training a model for word predictions where it should only rely on previous words. 

![[Pasted image 20260509230112.png]]

### Methodology 
![[Pasted image 20260510012905.png]]
- Zeroing out the element ahead of token(above the diagonal)
- normalize again so attention weight sums up to zero again. 

##### *the negative infinity implementation*

Softmax converts negative infinity value into a zero probability. By masking above the diagonal with negative infinity (first 1s then replace with -inf values) before softmax it would be a more efficient way to implement causal attention. 

### Additional Masking with dropout

dropout is the process of randomly zero out attention weights after softmax. During training this prevents overfitting on particular attention/contextual patterns.
![[Pasted image 20260510212449.png]]



