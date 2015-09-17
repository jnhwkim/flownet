require 'loadcaffe'
require 'rgbd'
require 'cutorch'
require 'cunn'

--[[load the first convolutional layer of VGGNet]]--
prototxt = '/Users/Calvin/Github/caffe/models/VGG_ILSVRC_16_layers/deploy_conv4_features.prototxt'
caffemodel = '/Users/Calvin/Github/caffe/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel'
model = loadcaffe.load(prototxt, caffemodel)

for i=model:size(),2,-1 do
	model:remove(i)
end

-- load data set
dataPath = '/Volumes/Oculus/data/rgbd/rgbd_dataset_freiburg2_pioneer_360'
ds = rgbd(dataPath, false, true)[1]
-- td = rgbd(dataPath, true, true)[1]

conv1 = model:get(1):float()  -- THTensor's default type is float
res = conv1:forward(ds:narrow(1,1,2))

res1 = torch.Tensor(res[1]:size()):copy(res[1]):float()
res1:pow(-1)
res1:cmul(res[1] - res[2])