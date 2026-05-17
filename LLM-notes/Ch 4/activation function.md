An activation function acts at a filter at the exit of a neural network layer. It takes the number produced in the layer and determines what to do with them before passing to the next layer. 

*Why is needed?*

standard linear layer(nn.Linear) only does basic algebra. So be stacking linear layers it essentially just acts as a single linear layer which represented real-world pattern in a straight line which is not helpful. 

By having an activation function it allows non-linearity. For every layer , it adds bends better "shaping" the mathematical model to predict for complex data.