require 'dp'
require 'cunn'
require 'cutorch'
require 'utils'
require 'rgbd'

dat = '/home/jhkim/save/mlc-Z97X-UD5H:1441208366:1.dat'
xp = torch.load(dat)

-- default dataset loading
dataType = 2
tensor = false
dataPath = rgbdPath('360')
ds = rgbd(dataPath, dataType, tensor)

model = xp:model()
data = ds:validSet():inputs():input():narrow(1,1,5)
target = ds:validSet():targets():input():narrow(1,1,5)

print(model:forward(data))
print(target)