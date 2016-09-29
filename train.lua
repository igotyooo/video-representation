--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local numModules = #model.modules
local optimState = {}
for m = 1, numModules do
	optimState[m] = {
	learningRate = opt.LR[m],
	learningRateDecay = 0.0,
	momentum = opt.momentum,
	dampening = 0.0,
	weightDecay = opt.weightDecay}
end
assert(numModules == #opt.LR)

if startEpoch > 1 then
	local optimPath = opt.pathOptim:format(startEpoch - 1)
    assert(paths.filep(optimPath), 'File not found: ' .. optimPath)
    print('Loading optimState from file: ' .. optimPath)
    optimState = torch.load(optimPath)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(opt.pathTrainLog:format(startEpoch))
local batchNumber
local eval_epoch, loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   eval_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
				-- DOYOO 1-) Call sampleVideos() instead of sample()
            local inputs, labels = trainLoader:sampleVideos(opt.batchSize, opt.seqLength)
				-- DOYOO) End.
            return inputs, labels, evaluateBatch
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   eval_epoch = eval_epoch / opt.epochSize
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['EvalMetric'] = eval_epoch,
      ['Loss'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'evaluation: %.2f\t',
                       epoch, tm:time().real, loss_epoch, eval_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   saveDataParallel(opt.pathModel:format(epoch), model) -- defined in util.lua
   torch.save(opt.pathOptim:format(epoch), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = {}, {}
for m, module in pairs(model.modules) do
	local w, gw = module:getParameters()
	parameters[ m ] = w
	gradParameters[ m ] = gw
end

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU, evaluateBatch)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

	model:zeroGradParameters()

	-- Forward pass.
   local outputs = {}
	for m, module in pairs(model.modules) do
		if m == 1 then
			outputs[m] = module:forward(inputs)
		else
			outputs[m] = module:forward(outputs[m - 1])
		end
	end
	local err = criterion:forward(outputs[numModules], labels)
	-- Backward pass.
	local gradOutputs = criterion:backward(outputs[numModules], labels)
	local gradBuff
	for m = numModules,1,-1 do
		if optimState[m].learningRate == 0 then break end
		if m == numModules then
			gradBuff = model.modules[m]:backward(outputs[m - 1], gradOutputs)
		elseif m == 1 then
			gradBuff = model.modules[m]:backward(inputs, gradBuff)
		else
			gradBuff = model.modules[m]:backward(outputs[m - 1], gradBuff)
		end
	end
	-- Update weights.
	for m = numModules,1,-1 do
		if optimState[m].learningRate == 0 then break end
		optim.sgd(function(x) return _, gradParameters[m] end, parameters[m], optimState[m])
	end

   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end
   

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   -- Task-specific evaluation.
	local eval = evaluateBatch(outputs[numModules], labelsCPU, opt.seqLength)
	eval_epoch = eval_epoch + eval
   -- Print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Eval %.2f DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, eval, dataLoadingTime))

   dataTimer:reset()
end
