# Train GoogleNet InceptionV3 model
https://arxiv.org/abs/1512.00567
python train_cnn.py --model=inception --data_directory=subset27x2000 --first_training_epochs=5 --second_training_epochs=20

# Train ResNet50 model
https://arxiv.org/abs/1512.03385
python train_cnn.py --model=resnet --data_directory=subset27x2000 --first_training_epochs=5 --second_training_epochs=20

# Train DenseNet121 model
https://arxiv.org/abs/1608.06993
python train_cnn.py --model=densenet --data_directory=subset27x2000 --first_training_epochs=5 --second_training_epochs=20