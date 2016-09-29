--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(opt.pathTestLog:format(startEpoch))

local batchNumber
local eval_center, loss
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   eval_center = 0
   loss = 0
   for i=1,nTest*opt.seqLength/opt.batchSize do -- DOYOO) Multiplied by seqLength.
      local indexStart = (i-1) * opt.batchSize / opt.seqLength + 1 -- DOYOO) Video index, not image index. indexEnd removed.
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, opt.batchSize, opt.seqLength) -- DOYOO) Get video func. 
            return inputs, labels, evaluateBatch
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   eval_center = eval_center / (nTest*opt.seqLength/opt.batchSize)
   loss = loss / (nTest*opt.seqLength/opt.batchSize) -- because loss is calculated per batch DOYOO) Divided by seqLength.
   testLogger:add{
		['EvalMetric'] = eval_center,
		['Loss'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'evaluation: %.2f\t ',
                       epoch, timer:time().real, loss, eval_center))

   print('\n')

end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU, evaluateBatch)
   batchNumber = batchNumber + opt.batchSize

   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, labels)
   cutorch.synchronize()

   loss = loss + err	
	local eval = evaluateBatch(outputs, labelsCPU, opt.seqLength)
	eval_center = eval_center + eval

   if batchNumber % 1024 == 0 then
      print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest*opt.seqLength)) -- DOYOO) seqLength multiplied.
   end
end
