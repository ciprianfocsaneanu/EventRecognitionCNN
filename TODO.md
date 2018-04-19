### Scene recognition CNN
#### Cloud training tasks:

##### -> research and find out how to make a dataset suitable for Google ML Engine (maybe H5)
##### -> run training on Google Cloud ML Engine
##### -> run distributed training on Google Compute Engine
##### ...

#### Local training tasks:

##### -> fix ResNet50 local training (at the moment a full model epoch training is inserted at the beginning)
##### -> train local models on CPU and note duration per epoch
##### -> compare local CPU vs GPU (duration, convergence time)
##### ...

---
### Object recognition CNN

#### -> choose CNN trained on ImageNet (comparing all the available options)

---
### Object-Scene Recognition CNN

#### -> merge the two CNNs
#### -> load best weights to each of them
#### -> add extra layers on top of them
#### -> train only extra layers to predict

---
### Dataset tasks:

#### -> find/choose activity dataset