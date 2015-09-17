-- require("mobdebug").start()
require 'rgbd'
require 'dp'
require 'utils'

--[[command line arguments]]--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using MLP Training/Optimization')
cmd:text('Example:')
cmd:text('$> th neuralnetwork.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('--xp', '', 'reload Experiment')
--[[learning parameters]]--
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--lrDecay', 'linear', 'type of learning rate decay : adaptive | linear | schedule | none')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 300, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--schedule', '{}', 'learning rate schedule')
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.001, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
--[[early stopping]]--
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
--[[display]]--
cmd:option('--progress', true, 'display progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
--[[advanced learning parameters]]--
cmd:option('--batchNorm', true, 'use batch normalization. dropout is mostly redundant with this')
cmd:option('--dropout', false, 'apply dropout on hidden neurons')
cmd:option('--dropoutProb', '{0.2,0.5,0.5}', 'dropout probabilities')
--[[convolution parameters]]--
cmd:option('--channelSize', '{48,128,192,192,128}', 'Number of output channels for each convolution layer.')
cmd:option('--kernelSize', '{11,5,5,3,3,3}', 'kernel size of each convolution layer. Height = Width')
cmd:option('--kernelStride', '{4,1,1,1,1}', 'kernel stride of each convolution layer. Height = Width')
cmd:option('--poolSize', '{3,3,1,1,3}', 'size of the max pooling of each convolution layer. Height = Width')
cmd:option('--poolStride', '{2,2,1,1,2}', 'stride of the max pooling of each convolution layer. Height = Width')
cmd:option('--padding', true, 'add math.floor(kernelSize/2) padding to the input of each convolution')
--[[neural networks parameters]]--
cmd:option('--activation', 'Tanh', 'transfer function like ReLU, Tanh, Sigmoid')
cmd:option('--hiddenSize', '{4096,4096}', 'number of hidden units per layer after convolutional layers.')
--[[cuda options]]--
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
--[[data & preprocessing]]--
cmd:option('--dataset', 'Rgbd', 'which dataset to use : Rgbd | Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--loadSize', '', 'Image size')
cmd:option('--sampleSize', '3,224,224', 'The size to use for cropped images')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:text()
opt = cmd:parse(arg or {})
opt.schedule = dp.returnString(opt.schedule)
opt.hiddenSize = dp.returnString(opt.hiddenSize)
if not opt.silent then
   table.print(opt)
end

opt.channelSize = table.fromString(opt.channelSize)
opt.kernelSize = table.fromString(opt.kernelSize)
opt.kernelStride = table.fromString(opt.kernelStride)
opt.poolSize = table.fromString(opt.poolSize)
opt.poolStride = table.fromString(opt.poolStride)
opt.dropoutProb = table.fromString(opt.dropoutProb)
opt.hiddenSize = table.fromString(opt.hiddenSize)
opt.loadSize = opt.loadSize:split(',')
for i = 1, #opt.loadSize do
   opt.loadSize[i] = tonumber(opt.loadSize[i])
end
opt.sampleSize = opt.sampleSize:split(',')
for i = 1, #opt.sampleSize do
   opt.sampleSize[i] = tonumber(opt.sampleSize[i])
end

--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())
end
if opt.lecunlcn then
   table.insert(input_preprocess, dp.GCN())
   table.insert(input_preprocess, dp.LeCunLCN{progress=true})
end

--[[data]]--
dataType = 2
tensor = false
volumetric = (2 == dataType)
dataPath = rgbdPath('360', opt.silent) -- default dataset loading
ds = rgbd(dataPath, dataType, tensor, opt.sampleSize)  -- get time-diff dataset

if opt.xp == '' then
   --[[Model]]--
   model = nn.Sequential()

   -- convolutional and pooling layers
   depth = 1
   inputSize = ds:imageSize('c') or opt.loadSize[1]
   for i=1,#opt.channelSize do
      if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
         -- dropout can be useful for regularization
         model:add(nn.SpatialDropout(opt.dropoutProb[depth]))
      end
      if not volumetric then
         model:add(nn.SpatialConvolution(
            inputSize, opt.channelSize[i], 
            opt.kernelSize[i], opt.kernelSize[i], 
            opt.kernelStride[i], opt.kernelStride[i],
            opt.padding and 2 or math.floor(opt.kernelSize[i]/2) or 0
         ))
      else
         if 1 == i then kT = 2 else kT = 1 end
         model:add(nn.VolumetricConvolution(
            inputSize, opt.channelSize[i], kT,
            opt.kernelSize[i]+1-kT*2, opt.kernelSize[i]+1-kT*2,
            1, opt.kernelStride[i]+2-kT*2, opt.kernelStride[i]+2-kT*2
         ))
      end
      if opt.batchNorm and not volumetric then
         -- batch normalization can be awesome
         model:add(nn.SpatialBatchNormalization(opt.channelSize[i]))
      end
      -- model:add(nn[opt.activation]())
      if opt.poolSize[i] and opt.poolSize[i] > 0 then
         if volumetric then
            model:add(nn.VolumetricMaxPooling(
               1, opt.poolSize[i], opt.poolSize[i],
               1, opt.poolStride[i] or opt.poolSize[i],
               opt.poolStride[i] or opt.poolSize[i]
            ))
         else
            model:add(nn.SpatialMaxPooling(
               opt.poolSize[i], opt.poolSize[i],
               opt.poolStride[i] or opt.poolSize[i],
               opt.poolStride[i] or opt.poolSize[i]
            ))
         end
      end
      inputSize = opt.channelSize[i]
      depth = depth + 1
   end
   -- get output size of convolutional layers
   if volumetric then 
      outsize = model:outside{1,ds:imageSize('c'),ds:imageSize('t'),ds:imageSize('h'),ds:imageSize('w')}
      model:insert(nn.Convert(ds:ioShapes(), 'bcthw'), 1)
   else
      outsize = model:outside{1,ds:imageSize('c'),ds:imageSize('h'),ds:imageSize('w')}
      model:insert(nn.Convert(ds:ioShapes(), 'bchw'), 1)
   end
   inputSize = outsize[2]*outsize[3]*outsize[4]
   if volumetric then
      inputSize = inputSize*outsize[5]
   end
   dp.vprint(not opt.silent, "input to dense layers has: "..inputSize.." neurons")

   -- dense hidden layers
   model:add(nn.Collapse(3))
   for i,hiddenSize in ipairs(opt.hiddenSize) do
      if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
         model:add(nn.Dropout(opt.dropoutProb[depth]))
      end
      model:add(nn.Linear(inputSize, hiddenSize))
      if opt.batchNorm then
         model:add(nn.BatchNormalization(hiddenSize))
      end
      model:add(nn[opt.activation]())
      inputSize = hiddenSize
      depth = depth + 1
   end

   -- output layer
   if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
      model:add(nn.Dropout(opt.dropoutProb[depth]))
   end
   model:add(nn.Linear(inputSize, #(ds:classes())))
   -- model:add(nn.LogSoftMax())

   --[[Propagators]]--
   if opt.lrDecay == 'adaptive' then
      ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
   elseif opt.lrDecay == 'linear' then
      opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
   end

   train = dp.Optimizer{
      acc_update = opt.accUpdate,
      loss = nn.ModuleCriterion(nn.MSECriterion(), nil, nn.Convert()),
      epoch_callback = function(model, report) -- called every epoch
         -- learning rate decay
         if report.epoch > 0 then
            if opt.lrDecay == 'adaptive' then
               opt.learningRate = opt.learningRate*ad.decay
               ad.decay = 1
            elseif opt.lrDecay == 'schedule' and opt.schedule[report.epoch] then
               opt.learningRate = opt.schedule[report.epoch]
            elseif opt.lrDecay == 'linear' then 
               opt.learningRate = opt.learningRate + opt.decayFactor
            end
            opt.learningRate = math.max(opt.minLR, opt.learningRate)
            if not opt.silent then
               print("learningRate", opt.learningRate)
            end
         end
      end,
      callback = function(model, report) -- called every batch
         if opt.accUpdate then
            model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
         else
            model:updateGradParameters(opt.momentum) -- affects gradParams
            model:updateParameters(opt.learningRate) -- affects params
         end
         model:maxParamNorm(opt.maxOutNorm) -- affects params
         model:zeroGradParameters() -- affects gradParams 
      end,
      --feedback = dp.Confusion(),
      sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
      progress = opt.progress
   }
   valid = dp.Evaluator{
      loss = nn.ModuleCriterion(nn.MSECriterion(), nil, nn.Convert()),
      --feedback = dp.Confusion(),  
      sampler = dp.Sampler{batch_size = opt.batchSize}
   }
   test = dp.Evaluator{
      --feedback = dp.Confusion(),
      sampler = dp.Sampler{batch_size = opt.batchSize}
   }

   --[[Experiment]]--
   xp = dp.Experiment{
      model = model,
      optimizer = train,
      validator = valid,
      --tester = test,
      observer = {
         dp.FileLogger(),
         dp.EarlyStopper{
            --error_report = {'validator','feedback','confusion','accuracy'},
            --error_report = {'validator','accuracy'},
            maximize = false,
            max_epochs = opt.maxTries
         }
      },
      random_seed = os.time(),
      max_epoch = opt.maxEpoch
   }
else
   print('Reloading previously trained network and experiment settings')
   require 'cutorch'
   require 'cunn'
   xp = torch.load(opt.xp)
   -- model = xp:model().module
   -- xp:model(model)
end

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Model :"
   print(model)
end

xp:run(ds)
