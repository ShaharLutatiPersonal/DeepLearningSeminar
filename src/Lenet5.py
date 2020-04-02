import torch.nn as nn

class NetOriginal(nn.Module):
	def __init__(self):
		super(NetOriginal,self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5,padding = 2)
		self.Tanh1 = nn.Tanh()
		self.pool1 = nn.AvgPool2d(2,stride=2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.Tanh2 = nn.Tanh()
		self.pool2 = nn.AvgPool2d(2, stride=2)
		self.conv3 = nn.Conv2d(16, 120, 5)
		self.Tanh3 = nn.Tanh()
		self.fc1 = nn.Linear(120, 84)
		self.Tanh4 = nn.Tanh()
		self.fc2 = nn.Linear(84, 10)
		self.Tanh5 = nn.Tanh()
	def forward(self, input):
		output = self.conv1(input)
		output = self.Tanh1(output)
		output = self.pool1(output)
		output = self.conv2(output)
		output = self.Tanh2(output)
		output = self.pool2(output)
		output = self.conv3(output)
		output = self.Tanh3(output)
		output = output.view(-1,self.num_flat_features(output))
		output = self.fc1(output)
		output = self.Tanh4(output)
		output = self.fc2(output)
		output = self.Tanh5(output)
		return output
	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
		
class NetD(nn.Module):
	def __init__(self):
		super(NetD,self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5,padding = 2)
		self.dropout1 = nn.Dropout2d(p=1.0)
		self.Tanh1 = nn.Tanh()
		self.pool1 = nn.AvgPool2d(2,stride=2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.Tanh2 = nn.Tanh()
		self.pool2 = nn.AvgPool2d(2, stride=2)
		self.conv3 = nn.Conv2d(16, 120, 5)
		self.dropout2 = nn.Dropout2d(p=0.5)
		self.Tanh3 = nn.Tanh()
		self.fc1 = nn.Linear(120, 84)
		self.dropout3 = nn.Dropout(p=0.5)
		self.Tanh4 = nn.Tanh()
		self.fc2 = nn.Linear(84, 10)
		self.Tanh5 = nn.Tanh()
	def forward(self, input):
		output = self.conv1(input)
		#output = self.dropout1(output)
		output = self.Tanh1(output)
		output = self.pool1(output)
		output = self.conv2(output)
		#output = self.dropout2(output)
		output = self.Tanh2(output)
		output = self.pool2(output)
		output = self.conv3(output)
		output = self.Tanh3(output)
		output = output.view(-1,self.num_flat_features(output))
		output = self.fc1(output)
		output = self.dropout3(output)
		output = self.Tanh4(output)
		output = self.fc2(output)
		output = self.Tanh5(output)
		return output
	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
		


class NetBN(nn.Module):
	def __init__(self):
		super(NetBN,self).__init__()
		self.BN = nn.BatchNorm2d(1)
		self.conv1 = nn.Conv2d(1, 6, 5,padding = 2)
		self.Tanh1 = nn.Tanh()
		self.pool1 = nn.AvgPool2d(2,stride=2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.Tanh2 = nn.Tanh()
		self.pool2 = nn.AvgPool2d(2, stride=2)
		self.conv3 = nn.Conv2d(16, 120, 5)
		self.Tanh3 = nn.Tanh()
		self.fc1 = nn.Linear(120, 84)
		self.Tanh4 = nn.Tanh()
		self.fc2 = nn.Linear(84, 10)
		self.Tanh5 = nn.Tanh()
	def forward(self, input):
		output = self.BN(input)
		output = self.conv1(output)
		output = self.Tanh1(output)
		output = self.pool1(output)
		output = self.conv2(output)
		output = self.Tanh2(output)
		output = self.pool2(output)
		output = self.conv3(output)
		output = self.Tanh3(output)
		output = output.view(-1,self.num_flat_features(output))
		output = self.fc1(output)
		output = self.Tanh4(output)
		output = self.fc2(output)
		output = self.Tanh5(output)
		return output
	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features




		
#original paper - after each one need tanh, we'll use relu cause cooler.
#conv2d - stride 1,kernel 5*5,size 28x28,channels 6
#average polling stride 2, kernel 2x2,size 14x14, channels 6
#conv2d stride 1, kernel 5x5,size 10x10,channels 16
#average polling stride 2, kernel 2x2 , size 5x5 ,channels 16
#conv2d stride 1 , kernel 5x5, size 1x1, channels 120
#FC1 size 84
#FC size 10
