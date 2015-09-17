--require("mobdebug").start()
require 'dp'
require 'torchx'  -- for paths.indexdir
require 'csvigo'  -- for parsing formatted text
-- require 'qt'

function rgbd(dataPath, dataType, tensor, sampleSize, validRatio)
    validRatio = validRatio or 0.15  -- last part of seq.
    height = 256
    width = 256
    dataType = dataType or 0
    tensor = tensor or false
    sampleSize = sampleSize or {3, 224, 224}

    -- 1. load images into input and target Tensors
    local trj = csvigo.load{path=dataPath..'/groundtruth.txt', verbose=false,
                            mode='query', separator=' ', skip=3, header=false}
    local rgb = csvigo.load{path=dataPath..'/rgb.txt', verbose=false,
                            mode='tidy', separator=' ', skip=3, header=false}
    local size = #(rgb.var_1)
    local _input = torch.FloatTensor(size, 3, sampleSize[2] or height, sampleSize[3] or width):fill(0)
    local _target = torch.FloatTensor(size, #(trj('vars'))-1)
    local cropped = {}
    local cached = false
    if os.execute('ls '..dataPath..'/cropped.dat 2> /dev/null') then
        print('cached!')
        cropped = torch.load(dataPath..'/cropped.dat')
        cached = true
    end
    
    local trj_idx = 1
    local rgb_idx = 1
    local ts_threshold = 1/20  -- in seconds, if it is large, consistency is compromised.
    for i=1,size do  -- # of rgb images
        xlua.progress(rgb_idx, size)
        local rgb_ts = rgb.var_1[i]
        local filename = rgb.var_2[i]
        local rgb_tsn = tonumber(rgb_ts)
        local trj_ts = trj('all').var_1[trj_idx]
        local trj_tsn = tonumber(trj_ts)

        -- note that some trj has the same ts.
        while math.abs(rgb_tsn - trj_tsn) >= math.abs(rgb_tsn - tonumber(trj('all').var_1[trj_idx+1])) do
            trj_idx = trj_idx + 1
            trj_ts = trj('all').var_1[trj_idx]
            trj_tsn = tonumber(trj_ts)
        end

        -- post-while assertion. if it failed, trj data is not dense enough.
        if math.abs(rgb_tsn - trj_tsn) >= ts_threshold then
            print('No sync: '..rgb_tsn..' - '..trj_tsn..' = '..rgb_tsn - trj_tsn)
        end
        --assert(math.abs(rgb_tsn - trj_tsn) < ts_threshold)
        if math.abs(rgb_tsn - trj_tsn) < ts_threshold then
            if cached then
                out = cropped[i]
            else
                local img = image.load(dataPath..'/'..filename)
                img = image.scale(img, height, width)
                -- crop
                local oW = sampleSize[3] or width
                local oH = sampleSize[2] or height
                local h1 = math.ceil((height - oH)/2)
                local w1 = math.ceil((width - oW)/2)
                out = image.crop(img, w1, h1, w1+oW, h1+oH)
                cropped[i] = out
            end
            -- image.display(out)
            _input[rgb_idx]:copy(torch.Tensor(out))
            for j=2,#(trj('vars')) do
                _target[rgb_idx][j-1] = torch.Tensor(trj('union', {var_1=trj_ts})['var_'..j])
            end
            rgb_idx = rgb_idx + 1
            -- collectgarbage()
        end
    end
    -- caching
    if not cached then
        torch.save(dataPath..'/cropped.dat', cropped)
    end

    -- narrow down
    size = rgb_idx - 1
    local input = _input:narrow(1,1,size):clone()
    local target = _target:narrow(1,1,size):clone()
    -- collectgarbage()

    -- so far we have input and target
    T = 4 -- 30,40,50,60
    if 0 ~= dataType then
        local td_size = size * T - T * (30+10*(T-1)+30) / 2;  --T_diff from 1 to T
        local td_input;
        if 1 == dataType then  -- time difference dataset
             td_input = torch.FloatTensor(td_size, 3, sampleSize[2] or height, sampleSize[3] or width)
        elseif 2 == dataType then  -- volumetric dataset, see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.VolumetricModules
            td_input = torch.FloatTensor(td_size, 3, 2, sampleSize[2] or height, sampleSize[3] or width)
        end
        local td_target = torch.FloatTensor(td_size, #(trj('vars'))-1)
        local td_i = 1
        for i = 1, size do
            for t = 30,math.min(30+(T-1)*10, size-i),10 do
                if 1 == dataType then
                    td_input[td_i] = input[i+t] - input[i]
                elseif 2 == dataType then
                    td_input[td_i]:select(2,1):copy(input[i])
                    td_input[td_i]:select(2,2):copy(input[i+t])
                end
                td_target[td_i] = target[i+t] - target[i]
                td_i = td_i + 1
            end
        end
        assert(td_i-1==td_size)
        size = td_size
        input = td_input
        target = td_target
    end
    collectgarbage()
    
    -- 2. divide into train and valid set and wrap into dp.Views
    local nValid = math.floor(size * validRatio)
    local nTrain = size - nValid

    if 2 == dataType then
        input_desc = 'bcthw' 
    else
        input_desc = 'bchw'
    end

    local trainInput = dp.ImageView(input_desc, input:narrow(1, 1, nTrain))
    local trainTarget = dp.ClassView('bc', target:narrow(1, 1, nTrain))
    local validInput = dp.ImageView(input_desc, input:narrow(1, nTrain+1, nValid))
    local validTarget = dp.ClassView('bc', target:narrow(1, nTrain+1, nValid))

    --@TODO how to deal with multi-out?
    trainTarget:setClasses({'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'})
    validTarget:setClasses({'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'})

    if tensor then
        return {input:narrow(1, 1, nTrain), 
                target:narrow(1, 1, nTrain),
                input:narrow(1, nTrain+1, nValid), 
                target:narrow(1, nTrain+1, nValid)}
    end

    -- 3. wrap view into datasets
    local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}   
    local valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}

    -- 4. wrap datasets into datasource
    local ds = dp.DataSource{train_set=train,valid_set=valid}
    ds:classes{'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'}
    return ds
end