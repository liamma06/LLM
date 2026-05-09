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
Value like the query and key is a transformed embedding(using Wv). However these contained the actual informational content a word contributes to the final context vector. Query and key determine *how much* the words listen to each other the key represents *what* they are actually saying. It is multiplied and merged into a new representation of the query word chosen. 


