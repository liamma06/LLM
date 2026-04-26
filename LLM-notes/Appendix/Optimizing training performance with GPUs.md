
For processes to occur, math with 2 [[tensors]] they have to be in the same "room" (CPU vs GPU)

Using the 
```
       .to()
       tensor.to("cuda")
       tensor.to("cpu")
```
you move the data from CPU to GPU and visa versa

#### Single-CPU training 

```
model.to(device)

features, labels = feature.to(device), labels.to(device)
```
- moves entire model into GPU at the start
- Happening inside the loop it would move those feat and labels into the GPU for calculations to occur 

Moving RAM(CPU) to VRAM(GPU) isn't instant and take time where doing all the processes on CPU might be faster but with LLMs (super math very) the time to transfer data might be worth it .

#### Multiple GPU training 

