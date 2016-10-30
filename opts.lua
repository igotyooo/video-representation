--
--Copyright (c) 2014, Facebook, Inc.
--All rights reserved.
--
--This source code is licensed under the BSD-style license found in the
--LICENSE file in the root directory of this source tree. An additional grant
--of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }
function M.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Torch-7 Video Classification Training script')
	cmd:text()
	cmd:text('Options:')
	------------ General options --------------------
	cmd:option('-cache', gpath.dataout, 'subdirectory in which to save/log experiments')
	cmd:option('-data', 'UCF101', 'Name of dataset defined in "./db/"')
	cmd:option('-GPU', 1, 'Default preferred GPU')
	cmd:option('-nGPU', 4, 'Number of GPUs to use by default')
	cmd:option('-backend', 'cudnn', 'Options: cudnn | nn')
	------------- Data options ------------------------
	cmd:option('-nDonkeys', 4, 'number of donkeys to initialize (data loading threads)')
	cmd:option('-imageSize', 240, 'Smallest side of the resized image')
	cmd:option('-cropSize', 224, 'Height and Width of image crop to be used as input layer')
	cmd:option('-keepAspect', 0, '1 for keep, 0 for no')
	cmd:option('-nClasses', 101, 'number of classes in the dataset')
	cmd:option('-normalizeStd', 0, 'Normalize piexel std to 1. 1 for yes, 0 for no.')
	------------- Training options --------------------
	cmd:option('-nEpochs', 55, 'Number of total epochs to run')
	cmd:option('-epochSize', 10000, 'Number of batches per epoch')
	cmd:option('-batchSize', 128, 'Frame-level mini-batch size (1 = pure stochastic)')
	cmd:option('-videoLevelTest', 0, 'To do video-level test, give a target epoch number of which model should be evaluated.')
	cmd:option('-saveFeature', 2, 'Module ID of model from which features are saved.')
	---------- Optimization options ----------------------
	cmd:option('-lrFeature', 0, 'Learning rates for conv feature layer.')
	cmd:option('-lrLstm', 16e-5, 'Learning rates for LSTM layer.')
	cmd:option('-lrFc', 1e-3, 'Learning rates for FC layer.')
	cmd:option('-lrClassifier', 1e-3, 'Learning rates for classifier layer.')
	cmd:option('-lrClassWeight', 1e-3, 'Learning rates for class-dependent weight layer.')
	cmd:option('-lrAgent', 1e-4, 'Learning rates for agent layer.')
	cmd:option('-momentum', 0.9, 'momentum')
	cmd:option('-weightDecay', 5e-4, 'weight decay')
	---------- Model options ----------------------------------
	cmd:option('-netType', 'alexpt_lstm256', 'Specify a network.')
	cmd:option('-inputIsCaffe', 1, 'Network taking input is loaded from Caffe? 1 for yes, 0 for no.')
	cmd:option('-seqLength', 16, 'Number of frames per input video')
	cmd:option('-task', 'slcls', 'slcls | mlcls')
	cmd:option('-numLoss', 1, 'Number of losses.')
	cmd:option('-startFrom', '', 'Path to the initial model. Using it for LR decay is recommended only.')
	cmd:text()

	local opt = cmd:parse(arg or {})
	-- Set dst path for 1) root.
	local dirRoot = paths.concat(opt.cache, opt.data)
	-- Set dst path for 2) dataset meta.
	local pathDbTrain = paths.concat(dirRoot, 'dbTrain.t7')
	local pathDbTest = paths.concat(dirRoot, 'dbTest.t7')
	-- Set dst path for 3) image statistics.
	local ignore = {} for k,v in pairs(opt) do ignore[k] = true end
	ignore['inputIsCaffe'] = nil
	ignore['imageSize'] = nil
	ignore['cropSize'] = nil
	ignore['keepAspect'] = nil
	local pathImStat = cmd:string('meanstd', opt, ignore) .. '.t7'
	pathImStat = paths.concat(dirRoot, pathImStat)
	-- Set dst path for 4) models and other informations.
	local ignore = {cache=true, data=true, GPU=true, nGPU=true, backend=true, videoLevelTest=true, saveFeature=true, netType=true, nDonkeys=true, startFrom=true}
	local dirModel = paths.concat(dirRoot, cmd:string(opt.netType, opt, ignore))
	if opt.startFrom ~= '' then
		local baseDir, epoch = opt.startFrom:match('(.+)/model_(%d+).t7')
		dirModel = paths.concat(baseDir, cmd:string('model_' .. epoch, opt, ignore))
	end
	opt.dirModel = dirModel
	opt.pathModel = paths.concat(opt.dirModel, 'model_%d.t7')
	opt.pathOptim = paths.concat(opt.dirModel, 'optimState_%d.t7')
	opt.pathTrainLog = paths.concat(opt.dirModel, 'train_start_from_%d.log')
	opt.pathTestLog = paths.concat(opt.dirModel, 'test_start_from_%d.log')
	opt.pathVideoLevelTestLog = paths.concat(opt.dirModel, 'video_level_test_of_%d')
	-- Put paths into option.
	opt.pathDbTrain = pathDbTrain
	opt.pathDbTest = pathDbTest
	opt.pathImStat = pathImStat
	-- Value processing.
	opt.normalizeStd = opt.normalizeStd > 0
	opt.inputIsCaffe = opt.inputIsCaffe > 0
	opt.keepAspect = opt.keepAspect > 0
	return opt
end

return M
