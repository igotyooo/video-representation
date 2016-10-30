--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'nngraph'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

gpath = paths.dofile('setgpath.lua')
local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

nClasses = opt.nClasses

paths.dofile('util.lua')
paths.dofile('model.lua')
opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

print(opt)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(startEpoch)

print('Saving everything to: ' .. opt.dirModel)
os.execute('mkdir -p ' .. opt.dirModel)

paths.dofile('data.lua')
if opt.videoLevelTest == 0 then
	paths.dofile('train.lua')
	paths.dofile('test.lua')
	epoch = startEpoch
	for i=startEpoch,opt.nEpochs do
		train()
		test()
		epoch = epoch + 1
	end
else
	paths.dofile('testvideo.lua')
	videoLevelTest()
end
