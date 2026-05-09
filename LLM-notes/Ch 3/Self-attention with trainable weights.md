![[Pasted image 20260507181222.png]]
Simplified self-attention the attention was hardcoded(simply the dot products of embeddings). With the addition of weight matrices the model can adjust these values and understand which features of a word are actually important for understanding context. 

### The different matrices projections. 
Instead of the basic dot product there are the 3 new transformations query, key, and value vectors 
![[Pasted image 20260507182745.png]]

#### Query
When you take the embedding from the original input multiplied by learned weight matrix (Wq). It transforms a static definition into a active question to rest of sentence.  By comparing to the Keys of every other word the model calculates exactly how much influence each neighbour should have on the word's final context vectors
#### Keys
Also when you take the input token embedding multiplied by weight matrix(Wk) it provides mathematically comparison on how the Query to these Keys are relevant.
#### Value 
Value like the query and key is a transformed embedding(using Wv). However these contained the actual informational content a word contributes to the final context vector. Query and key determine *how much* the words listen to each other the value represents *what* they are actually saying. It is multiplied and merged into a new representation of the query word chosen. 

![[Pasted image 20260509032620.png]]
The [[attention weights]](unnormalized -> attention score) are generated for each token with respect to the query vector.

![[Pasted image 20260509043148.png]]

##### *why scaled dot product attention score before [[softmax]]?*

$$ \text{Attention Weights} = \text{softmax}\left(\frac{\text{Attention Scores}}{\sqrt{d_k}}\right) $$
Notice we divide to keep the scores at a manageable magnitude before calling softmax. This is because as embedding dimensions grow (could be over 1000 in professional LLMs) dot product can become very large. However due to the nature of softmax with large input numbers the softmax function pushes the highest values to 1 and everything else to 0 (Step function). The gradient in tern nears zero. Remembering that the gradient is used to figure out how to adjust the weights of the model, with a nearly zero gradient, the information on how to improve is very small causing training to slow down or even stop. So by dividing you prevent this. 

![[Pasted image 20260509043127.png]]
Once the attention weights are computed you calculate the weighted sum  with the value vector. Value vector represents the actual information/content of each token so with the context(from attention weights) we can calculate the final context vector built from the values considering the context. 

