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
local parameters, gradParameters, optimState = groupParams(model)
assert(#parameters == #gradParameters and #parameters == #optimState)
if startEpoch > 1 then
	local optimPath = opt.pathOptim:format(startEpoch - 1)
    assert(paths.filep(optimPath), 'File not found: ' .. optimPath)
    print('Loading optimState from file: ' .. optimPath)
    optimState = torch.load(optimPath)
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
	local outputs = model:forward(inputs)
	local err = criterion:forward(outputs, labels)

	-- Backward pass.
	local gradOutputs = criterion:backward(outputs, labels)
	model:backward(inputs, gradOutputs)

	-- Update weights.
	for m = 1,#parameters do
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
	local eval = evaluateBatch(outputs, labelsCPU, opt.seqLength)
	eval_epoch = eval_epoch + eval
	local fwdbwdTime = timer:time().real
	local totalTime = dataLoadingTime + fwdbwdTime
	local speed = opt.batchSize / totalTime
   -- Print information
   print(('Epoch %d) %d/%d, %dim/s (load %.2fs, fwdbwd %.2fs), err %.4f, eval %.2f'):format(
          epoch, batchNumber, opt.epochSize, speed, dataLoadingTime, fwdbwdTime, err, eval))

   dataTimer:reset()
end
