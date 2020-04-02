import Lenet5
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import time

batch_size = 256
train_dataset = mnist.MNIST('./data/mnist', train=True,download = False, transform=ToTensor())
test_dataset = mnist.MNIST('./data/mnist', train=False,download = False, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
models = [Lenet5.NetOriginal(),Lenet5.NetD(),Lenet5.NetBN(),Lenet5.NetOriginal()]
models_name = ['original','dropout','batch normalization','weight decay']
epoch = 15
origres,dropoutres,bnres,weight_decayres = [],[],[],[]
ResVsName = {'original':origres,'dropout':dropoutres,'batch normalization':bnres,'weight decay':weight_decayres}
trainOrig,trainDrop,trainBn,trainWd = [],[],[],[]
TrainResVsName = {'original':trainOrig,'dropout':trainDrop,'batch normalization':trainBn,'weight decay':trainWd}
numOfSamplesForTrainData = train_loader.dataset.train_data.shape[0]
for name,model in zip(models_name,models):
	if name != 'weight decay':
		sgd = SGD(model.parameters(), lr=1e-1)
	else:
		sgd = SGD(model.parameters(), lr=1e-1,weight_decay = 0.001)
		print('weight decay start with {}'.format(.1))
	cross_error = CrossEntropyLoss()
	print('start trainig model: ' + name)
	for _epoch in range(epoch):
		start_time = time.time()
		model.train(True)
		sum_of_errors_in_epoch = 0
		for idx, (train_x, train_label) in enumerate(train_loader):
			label_np = np.zeros((train_label.shape[0], 10))
			sgd.zero_grad()
			predict_y = model(train_x.float())
			_error = cross_error(predict_y, train_label.long())
			sum_of_errors_in_epoch += _error.item()*train_label.shape[0]
			if idx % 10 == 0:
				print('idx: {}, _error: {}'.format(idx, _error))
			_error.backward()
			sgd.step()
		TrainResVsName[name].append(1-(sum_of_errors_in_epoch/numOfSamplesForTrainData)/100)
		correct = 0
		_sum = 0
		model.train(False) # Change to eval mode for Dropout to clear out
		for idx, (test_x, test_label) in enumerate(test_loader):
			predict_y = model(test_x.float()).detach()
			predict_ys = np.argmax(predict_y, axis=-1)
			label_np = test_label.numpy()
			_ = predict_ys == test_label
			correct += np.sum(_.numpy(), axis=-1)
			_sum += _.shape[0]
		print('accuracy: {:.2f}'.format(correct / _sum) + ' epoch : {}'.format(_epoch+1))
		print('********************************************************************************************')
		end_time = time.time()
		print('time elapsed {}'.format(end_time-start_time))
		ResVsName[name].append(correct / _sum)
		
	#torch.save(model, 'models/mnist_{:.2f}.pkl'.format(correct / _sum)) 
print('Results for training stage')
for key in TrainResVsName:
	print('Results for' + key)
	print(TrainResVsName[key])
print('**************************************************************************************************************')
print('Results for test stage')
for key in ResVsName:
	print('Results for' + key)
	print(ResVsName[key])