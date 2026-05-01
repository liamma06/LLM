Is the process of turning raw test into units/value that could be processed as a value. 

### Regex
In the example regex was used to simply break apart a sentence separating words and punction. Where every entity was assigned a value and mapped to the sentence.
![[Pasted image 20260430232956.png|294]]
However this doesn't scale properly as there would be unknown words when tokenizing a new sentence not part of the initial one that built the vocab map.  

### BPE 
BPE is able to avoid unknown by breaking words down being able to tokenizer all sentences to get token IDS of all characters 

Now that we have converted text into token IDs we need to separate them between input and target done through [[data sampling]]