![[Pasted image 20260318114737.png]]

Typically without the workers the training loop has to stop and wait for each batch. *running in a queue*

With mulitple workers, background processes are queuing up the next batches while GPU trains the current one. *GPU training has less idle time* 

This comes with tradeoffs as it takes time to start up workers processes too and could be longer than actual data loading if set is small .