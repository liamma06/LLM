![[Pasted image 20260519221045.png]]

*Why finetuning is needed?*
We are able to "predict" the next word but fine tuning teaches the model how to behave ("Answer this question"). 

![[Pasted image 20260519221405.png]]

#### 2 Types of fine tuning

- **[[Classification Fine Tuning]]:**
In the final output layer of the model instead of prediction the word it outputs a probability for specific label (eg: Spam, not spam). The model becomes completely restricted for spam detection you can't ask further questions (eg: why is this spam?)
![[Pasted image 20260519222213.png|430]]

- **Instruction Fine Tuning:**
It trained to understand the intent of the prompt