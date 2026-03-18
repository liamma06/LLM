![[Pasted image 20260316220809.png]]

Pytorch custom dataset (input data) splits into training and test dataset:
- Training-> the [[neural network]] learns from this and adjusts weights 
- Test-> used to check real-world performance 

### "What is the point of the dataloader?"

*How smartly you feed data into the network during training.*

1) **Problem 1 - Batching** 
*without*: feed 1 at a time and update then feed next. 
*with*: the groups the data in batches and process in parallel. Helps reduce memory load and easier to train larger datasets. 

2) **Problem 2 -  Shuffling** 
*without:* Network sees same example in same order every epoch. 
*with:* model encounters data points in different combination forcing it to learn from a more diverse set of data combination for each epoch. 

3)**Problem 3 - scale/processing**
*without*: load all data into memory at once. 
*with*: loads only 1 batch at a time and frees batch afterwards. Also tools such as [[num_workers]]







 