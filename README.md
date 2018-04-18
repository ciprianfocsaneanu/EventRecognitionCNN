## Local training

### Train GoogleNet InceptionV3 model

https://arxiv.org/abs/1512.00567

`python local_cnn_train.py --cnn_model=inception --data_directory=places205-subset27 --batch_size=25 --img_size=224 --first_training_epochs=5 --second_training_epochs=20`

### Train DenseNet121 model

https://arxiv.org/abs/1608.06993

`python local_cnn_train.py --cnn_model=densenet --data_directory=places205-subset27 --batch_size=25 --img_size=224 --first_training_epochs=5 --second_training_epochs=20`

<del>
### Train ResNet50 model

https://arxiv.org/abs/1512.03385

`python local_cnn_train.py --cnn_model=resnet --data_directory=places205-subset27 --batch_size=25 --img_size=224 --first_training_epochs=5 --second_training_epochs=20`
</del>
