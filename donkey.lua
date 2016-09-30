--------------------------------------------------------
-- This file was customized from Soumith's "donkey.lua".
--------------------------------------------------------

require 'image'
paths.dofile( string.format( 'dataset_%s.lua', opt.task ) )
paths.dofile( 'util.lua' )
local mean, std
local function resizeImage( im )
	local imSize = opt.imageSize
	if opt.keepAspect then
		if input:size( 3 ) < input:size( 2 ) then
			im = image.scale( im, imSize, imSize * im:size( 2 ) / im:size( 3 ) )
		else
			im = image.scale( im, imSize * im:size( 3 ) / im:size( 2 ), opt.imSize )
		end
	else
		im = image.scale( im, imSize, imSize )
	end
	return im
end
local function loadImage( path )
	-- Load image.
	local im = image.load( path, 3, 'float' )
	-- Spatial rescale.
	im = resizeImage( im )
	-- Pixel rescale and permut channels.
	if opt.inputIsCaffe then
		im = im * 255
		im = im:index( 1, torch.LongTensor{ 3, 2, 1 } )
	end
	return im
end
local function normalizeImage( im )
	for i = 1, 3 do
		if mean then im[ { { i }, {  }, {  } } ]:add( -mean[ i ] ) end
		if std and opt.normalizeStd then im[ { { i }, {  }, {  } } ]:div( std[ i ] ) end
	end
	return im
end
do assert( opt.batchSize % opt.seqLength == 0 ) end

------------------------------
-- Create a train data loader.
------------------------------
local trainHook = function( self, path, rw, rh, rf )
	collectgarbage(  )
	local input = loadImage( path )
	local iW = input:size( 3 )
	local iH = input:size( 2 )
	-- Do random crop.
	local oW = opt.cropSize
	local oH = opt.cropSize
	local h1 = math.ceil( ( iH - oH ) * rh )
	local w1 = math.ceil( ( iW - oW ) * rw )
	if iH == oH then h1 = 0 end
	if iW == oW then w1 = 0 end
	local out = image.crop( input, w1, h1, w1 + oW, h1 + oH )
	assert( out:size( 3 ) == oW )
	assert( out:size( 2 ) == oH )
	-- Do horz-flip.
	if rf > 0.5 then out = image.hflip( out ) end
	-- Normalize.
	out = normalizeImage( out )
	return out
end
if paths.filep( opt.pathDbTrain ) then
	print( 'Loading train metadata from cache' )
	trainLoader = torch.load( opt.pathDbTrain )
	trainLoader.sampleHookTrain = trainHook
else
	print( 'Creating train metadata.' )
	trainLoader = dataLoader( genDb, 'train', opt.imageSize, opt.cropSize )
	torch.save( opt.pathDbTrain, trainLoader )
	trainLoader.sampleHookTrain = trainHook
end
collectgarbage(  )
print( string.format( 'Train loader copes with %d videos of %d classes.',
trainLoader.vid2path:size( 1 ), trainLoader.cid2name:size( 1 ) ) )

-----------------------------
-- Create a test data loader.
-----------------------------
local testHook = function( self, path )
	collectgarbage(  )
	local input = loadImage( path )
	local iW = input:size( 3 )
	local iH = input:size( 2 )
	-- Do central crop.
	local oW = opt.cropSize
	local oH = opt.cropSize
	local h1 = math.ceil( ( iH - oH ) / 2 )
	local w1 = math.ceil( ( iW - oW ) / 2 )
	if iH == oH then h1 = 0 end
	if iW == oW then w1 = 0 end
	local out = image.crop( input, w1, h1, w1 + oW, h1 + oH )
	assert( out:size( 3 ) == oW )
	assert( out:size( 2 ) == oH )
	-- Normalize.
	out = normalizeImage( out )
	return out
end
if paths.filep( opt.pathDbTest ) then
	print( 'Loading test metadata from cache' )
	testLoader = torch.load( opt.pathDbTest )
	testLoader.sampleHookTest = testHook
else
	print( 'Creating validation metadata.' )
	testLoader = dataLoader( genDb, 'val', opt.imageSize, opt.cropSize )
	torch.save( opt.pathDbTest, testLoader )
	testLoader.sampleHookTest = testHook
end
collectgarbage(  )
do
	print( string.format( 'Test loader copes with %d videos of %d classes.',
	testLoader.vid2path:size( 1 ), testLoader.cid2name:size( 1 ) ) )
end

-----------------------------------------
-- Estimate the per-channel mean and std.
-----------------------------------------
if paths.filep( opt.pathImStat ) then
	local meanstd = torch.load( opt.pathImStat )
	mean = meanstd.mean
	std = meanstd.std
	print( 'Loaded mean and std from cache.' )
else
	local tm = torch.Timer(  )
	local numVideos = 10000
	print( 'Estimating the RGB mean over ' .. numVideos .. ' videos.' )
	local meanEstimate = { 0, 0, 0 }
	for v = 1, numVideos do
		local img = trainLoader:sampleVideos( 1, 1 )[ 1 ]
		for ch = 1, 3 do
			meanEstimate[ ch ] = meanEstimate[ ch ] + img[ ch ]:mean(  )
		end
	end
	for ch = 1, 3 do
		meanEstimate[ ch ] = meanEstimate[ ch ] / numVideos
	end
	mean = meanEstimate
	print( 'Estimating the RGB std over ' .. numVideos .. ' videos.' )
	local stdEstimate = { 0, 0, 0 }
	for i = 1, numVideos do
		local img = trainLoader:sampleVideos( 1, 1 )[ 1 ]
		for ch = 1, 3 do
			stdEstimate[ ch ] = stdEstimate[ ch ] + img[ ch ]:std(  )
		end
	end
	for ch = 1, 3 do
		stdEstimate[ ch ] = stdEstimate[ ch ] / numVideos
	end
	std = stdEstimate
	local cache = {  }
	cache.mean = mean
	cache.std = std
	torch.save( opt.pathImStat, cache )
	print( 'Time to estimate:', tm:time( ).real )
end
