require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'image'

matio = require 'matio'

-- Command Line Arguments

opt = lapp[[
	
	--pixel_size		(default  11)		input pixel size
	--kernel_size		(default  3)		convolutional kernel size
	--learning_rate		(default  0.01)		learning rate hyperparameter
	--max_epochs		(default  4000)		maximum number of iterations
	--conv1				(default  500)		number of filters in the first convolutional layer
	--conv2				(default  100)		number of filters in the second convolutional layer
	--fc1				(default  200)		number of units in the first fully connected layer
	--fc2				(default  84)		number of units in the second fully connected layer
	--batch_size		(default  100)		size of the mini batch 
	--dropout			(default  1.0)		probability of dropout layer
]]



-- Loading the dataset 

function load_dataset(pixel_size)
	
	train = matio.load("../Data/Train" .. tostring(pixel_size) .. "_0_" .. tostring(1) .. ".mat")
	test = matio.load("../Data/Train" .. tostring(pixel_size) .. "_90_" .. tostring(1) .. ".mat")
	x_train, y_train, x_test, y_test = train.train_patch, train.train_labels:transpose(1,2) , test.test_patch, test.test_labels:transpose(1,2)
	for i = 2, 8 do
		temp = matio.load("../Data/Train" .. tostring(pixel_size) .. "_0_" .. tostring(i) .. ".mat")
		x_train = torch.cat(x_train, temp.train_patch, 1)
		y_train = torch.cat(y_train, temp.train_patch:transpose(1,2), 1)
	end

	for i = 2, 6 do 
		temp = matio.load("../Data/Test" .. tostring(pixel_size) .. "_0_" .. tostring(i) .. ".mat")
		x_test = torch.cat(x_test, temp.test_patch, 1)
		y_test = torch.cat(y_test, temp.test_labels:transpose(1,2), 1)
	end

   local x_train = x_train:type(torch.getdefaulttensortype())
   local x_test = x_test:type(torch.getdefaulttensortype())

   print("Training Data Size:")
   print(data:size())
   print("Training Labels Size:")
   print(labels:size())

   local train_nExample = x_train:size(1)

   local train = {}
   train.data = x_train
   train.labels = y_train

   function train:size()
      return train_nExample
   end

   local test = {}
   test.data = x_test
   test.labels = y_test

   local labelvector = torch.zeros(16)

  setmetatable(train, {__index = function(self, index)
		     local input = self.data[index]
		     local class = self.labels[index]
		     local label = labelvector:zero()
		     label[class[1]] = 1
		     local example = {input, label}
                                   return example
	end})

    setmetatable(test, {__index = function(self, index)
		     local input = self.data[index]
		     local class = self.labels[index]
		     local label = labelvector:zero()
		     label[class[1]] = 1
		     local example = {input, label}
                                   return example
	end})
  
    return train, test

end









