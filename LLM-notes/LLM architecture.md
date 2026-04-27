SUPER HIGH LEVEL UNDERSTANDING 
1. **Data loading and Model preparation** 
 diverse, comprehensive text is gathered

![[Pasted image 20260427101202.png|457]]

- GPT models use the decoder half of the original transformer( left to right text generation) ( unidirectional)

- BERT models use the encoder half (context from both directions) ( directional both ways ) Predicts masked words 

We will be using the GPT model

2. **Pretraining (self-supervised)**
- helps model learn general patterns 
- The model preforms next-word prediction on text, the next word on the completed sentence is used as the label. Does it match? -> loss function is calculated to see how off the model is and adjusts it weights. 
- NO manual labels are needed. simply the input text is used 

Original: The stove is hot. 
The __ -> mouse (not matching adjust weights)

2. **Fine-tuning (supervised)**
- pretrain model from the previous step goes through further training with more specific datasets 
- for a variety of use cases

![[Pasted image 20260427110632.png|458]]

