--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'optim'
require 'LSTM'

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
paths.dofile('models/' .. opt.netType .. '.lua')
startEpoch = 1
for i=1,opt.nEpochs do
	local modelPath = opt.pathModel:format(i)
	local optimPath = opt.pathOptim:format(i)
	if not (paths.filep(modelPath) and paths.filep(optimPath)) then startEpoch = i break end 
end
if opt.videoLevelTest ~= 0 then
	assert( startEpoch > opt.videoLevelTest )
	startEpoch = opt.videoLevelTest + 1
end
if startEpoch ~= 1 then
   print('Loading model from epoch ' .. startEpoch - 1);
   model = loadDataParallel(opt.pathModel:format(startEpoch - 1), opt.nGPU) -- defined in util.lua
elseif opt.startFrom ~= '' then
   print('Loading model of ' .. opt.startFrom);
	model = loadDataParallel(opt.startFrom, opt.nGPU)
else
	assert(opt.startFrom == '')
	print('No model to continue.')
	print('=> Creating model from file: models/' .. opt.netType .. '.lua')
	model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
	if opt.backend == 'cudnn' then
		require 'cudnn'
		cudnn.convert(model, cudnn)
	elseif opt.backend ~= 'nn' then
		error'Unsupported backend'
	end
end
-- 2. Create Criterion
if opt.numLoss == 1 then
	if opt.task == 'slcls' then
		criterion = nn.ClassNLLCriterion()
	elseif opt.task == 'mlcls' then
		criterion = nn.MultiLabelMarginCriterion()
	end
else
	criterion = nn.ParallelCriterion(true)
	for n = 1, opt.numLoss do
		if opt.task == 'slcls' then
			criterion:add(nn.ClassNLLCriterion(), 1 / opt.numLoss)
		elseif opt.task == 'mlcls' then
			criterion:add(nn.MultiLabelMarginCriterion(), 1 / opt.numLoss)
		end
	end
end

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
model = model:cuda()
criterion:cuda()

collectgarbage()
