-- This code evaluates trained model in video-level, not in frame-level.
-- Center crop, full-frame pooling will be done to do that.
local ffi = require 'ffi'
local eval_center
local batchSize = opt.batchSize
local seqLength = opt.seqLength
local numSeqPerBatch = batchSize / seqLength
local inputBatchGPU = torch.CudaTensor(  )
local loggerDir = opt.pathVideoLevelTestLog:format( opt.videoLevelTest )
local saveFeature = opt.saveFeature
os.execute( 'mkdir -p ' .. loggerDir )
function videoLevelTest(  )
   print( 'Doing video-level evaluation over test set.' )
   cutorch.synchronize(  )
   model:evaluate(  )
   eval_center = 0
	for vid = 1, nTest do
		donkeys:addjob( 
			function(  )
				local video, labels = testLoader:getVideo( vid )
				return vid, video, labels, evaluateVideo
			end,
			testVideo
		 )
	end
   donkeys:synchronize(  )
   cutorch.synchronize(  )
   eval_center = eval_center / nTest
	local msg = ( 'Average evaluation score: %.4f\n' ):format( eval_center )
	print( msg )
	local evalLogger = io.open( paths.concat( loggerDir, 'evaluation.txt' ), 'w' )
	evalLogger:write( msg )
	io.close( evalLogger )
end
function testVideo( vid, video, labels, evaluateVideo )
	collectgarbage(  )
	-- Check if input is shorter than seq length.
	local s1, s2, s3 = video:size( 2 ), video:size( 3 ), video:size( 4 )
	local numFrames = video:size( 1 )
	if numFrames < seqLength then
		local numAdd = seqLength - numFrames
		local video_ = torch.Tensor( seqLength, s1, s2, s3 )
		for i = 1, seqLength do
			if i <= numAdd then
				video_[ i ]:copy( video[ 1 ] )
			else
				video_[ i ]:copy( video[ i - numAdd ] )
			end
		end
		video = video_
	end
	numFrames = video:size( 1 )
	assert( numFrames >= seqLength )
	-- Do the job.
	local stride = math.max( 1, seqLength / 2 )
	local numSeq = math.floor( ( numFrames - seqLength ) / stride + 1 )
	local numBatch = math.ceil( numSeq * seqLength / batchSize )
	local inputs = torch.FloatTensor( numSeq * seqLength, s1, s2, s3 )
	local outputs = torch.FloatTensor( numSeq * seqLength, opt.nClasses )
	local features = nil
	assert( stride % 1 == 0 )
	for s = 1, numSeq do
		local fstart = ( s - 1 ) * stride + 1
		local fend = fstart + seqLength - 1
		local istart = ( s - 1 ) * seqLength + 1
		local iend = istart + seqLength - 1
		inputs[ { { istart, iend } } ] = video[ { { fstart, fend } } ]
	end
	for b = 1, numBatch do
		local istart = ( b - 1 ) * batchSize + 1
		local iend = math.min( numSeq * seqLength, istart + batchSize - 1 )
		local inputBatch = inputs[ { { istart, iend } } ]
		inputBatchGPU:resize( inputBatch:size(  ) ):copy( inputBatch )
		local buff = inputBatchGPU
		for m, module in pairs( model.modules ) do
			buff = module:forward( buff )
			if m == saveFeature then
				if features == nil then
					local sz = buff:size(  )
					sz[ 1 ] = numSeq * seqLength
					features = torch.FloatTensor( sz )
				end
				features[ { { istart, iend } } ]:copy( buff )
			end
			if m == #model.modules then
				outputs[ { { istart, iend } } ]:copy( buff )
			end
		end
		cutorch.synchronize(  )
	end
	local evalScore, predictions, cid2score, feature = 
		evaluateVideo( outputs, labels, features, seqLength )
	eval_center = eval_center + evalScore
	-- Logging.
	local vpath = ffi.string( torch.data( testVideoPaths[ vid ] ) )
	local evalLogger = io.open( paths.concat( loggerDir, ( '%06d.txt' ):format( vid ) ), 'w' )
	evalLogger:write( vpath, '\n' )
	evalLogger:write( ( '%.2f\n' ):format( evalScore ) )
	for c = 1, labels:size( 1 ) do
		evalLogger:write( ffi.string( torch.data( classes[ labels[ c ] ] ) ), '; ' )
	end
	evalLogger:write( '\n' )
	for c = 1, predictions:size( 1 ) do
		evalLogger:write( ffi.string( torch.data( classes[ predictions[ c ] ] ) ), '; ' )
	end
	evalLogger:write( '\n\n' )
	io.close( evalLogger )
	print( ( 'Video %06d(/%d): %.4f' ):format( vid, nTest, evalScore ) )
	local featPath = paths.concat( loggerDir, ( '%06d.t7' ):format( vid ) )
	torch.save( featPath, feature )	
end
