--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
      local options = opt -- make an upvalue to serialize over to donkey threads
      local globalPath = gpath -- make an upvalue to serialize over to donkey threads
		local se = startEpoch
      donkeys = Threads(
         opt.nDonkeys,
         function()
            require 'torch'
         end,
         function(idx)
				torch.setnumthreads(1)
            opt = options -- pass to all donkeys via upvalue
            gpath = globalPath -- pass to all donkeys via upvalue
            tid = idx
            local seed = se + idx
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            paths.dofile('donkey.lua')
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile('donkey.lua')
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

nClasses = nil
classes = nil
donkeys:addjob(function() return trainLoader.cid2name end, function(c) classes = c end) -- DOYOO) trainLoader.classes -> trainloader.cid2name
donkeys:synchronize()
nClasses = classes:size( 1 )
assert(nClasses, "Failed to get nClasses")
assert(nClasses == opt.nClasses,
       "nClasses is reported different in the data loader, and in the commandline options")
print('nClasses: ', nClasses)

nTrain, nTest = 0
donkeys:addjob(function() return trainLoader.vid2path:size( 1 ), testLoader.vid2path:size( 1 ) end, function(ntr, nte) nTrain = ntr nTest = nte end)
donkeys:synchronize()
assert(nTrain > 0, "Failed to get nTrain")
assert(nTest > 0, "Failed to get nTest")
print('nTrain: ', nTrain)
print('nTest: ', nTest)

testVideoPaths = nil
donkeys:addjob(function() return testLoader.vid2path end, function(v) testVideoPaths = v end)
donkeys:synchronize()
