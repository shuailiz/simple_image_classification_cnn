import os
from train import TrainCNN

model_path = os.path.join(os.getcwd(), '../models/')
data_path = os.path.join(os.getcwd(), '../data/training_set/')
trainer = TrainCNN(128, 3, ['cats', 'dogs'], model_path, [3, 3, 3], [32, 32, 64], [128], 32)
trainer.load_data(data_path, 0.85)

trainer.train(10000)
