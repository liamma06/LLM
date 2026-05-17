*Why ?*

When training neural networks with many layers a major problem is [[vanishing & exploding gradients]].  With the help of layer normalization it prevent this issue by normalizing each output of the layer to remain within a stable range.

It does this by taking each row of outputs and standard them using 

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$ Where $\mu$ is the mean, $\sigma^2$ is the variance, and $\epsilon$ is a tiny number to prevent dividing by zero.

- It shifts the numbers so the average is **0**. Data is now balanced around center  (**Mean = 0**)
- It scales numbers of the spread has a standard deviation of **1**. Data is not too tight or blown far apart (**Variance = 1**)![[68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830345f636f6d707265737365642f30352e77656270.webp]]
