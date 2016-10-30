---------------------------------------------------------
-- This file was customized from Soumith's "dataset.lua".
---------------------------------------------------------

require 'torch'
require 'image'
local ffi = require 'ffi'
paths.dofile( string.format( './db/%s.lua', opt.data ) )
torch.setdefaulttensortype( 'torch.FloatTensor' )

local dataset = torch.class( 'dataLoader' )
function dataset:__init( gendb, setName, loadSize, sampleSize )
	self.setName = setName
	self.loadSize = loadSize
	self.sampleSize = sampleSize
	self.sampleHookTrain = self.defaultSampleHook
	self.sampleHookTest = self.defaultSampleHook
	self.vid2path, self.vid2numim, self.vid2cid, self.cid2name, self.frameFormat = genDb( setName )
end
-- Converts a table of samples and labels to a clean tensor.
local function tableToOutput( self, dataTable, scalarTable )
	local data, scalarLabels
	local quantity = #scalarTable
	assert( dataTable[ 1 ]:dim(  ) == 3 )
	data = torch.Tensor( quantity, 3, self.sampleSize, self.sampleSize )
	scalarLabels = torch.LongTensor( quantity ):fill( -1111 )
	for i = 1, #dataTable do
		data[ i ]:copy( dataTable[ i ] )
		scalarLabels[ i ] = scalarTable[ i ]
	end
	return data, scalarLabels
end
function dataset:sampleVideos( quantity, seqLength )
	assert( quantity > 0 )
	assert( ( quantity % seqLength ) == 0 )
	local dataTable = {  }
	local scalarTable = {  }
	local numVideoToSample = quantity / seqLength
	local numVideo = self.vid2path:size( 1 )
	for v = 1, numVideoToSample do
		local vid = torch.random( 1, numVideo )
		local vpath = ffi.string( torch.data( self.vid2path[ vid ] ) )
		local numFrame = self.vid2numim[ vid ]
		local cid = self.vid2cid[ vid ]
		local startFrame = torch.random( 1, math.max( 1, numFrame - seqLength + 1 ) )
		local rw, rh, rf
		for f = 1, seqLength do
			local fid = math.min( numFrame, startFrame + f - 1 )
			local fpath = paths.concat( vpath, string.format( self.frameFormat, fid ) )
			if f == 1 then
				rw = torch.uniform(  )
				rh = torch.uniform(  )
				rf = torch.uniform(  )
			end
			local out = self:sampleHookTrain( fpath, rw, rh, rf )
			table.insert( dataTable, out )
			table.insert( scalarTable, cid )
		end
	end
	local data, scalarLabels = tableToOutput( self, dataTable, scalarTable )
	return data, scalarLabels
end
function dataset:get( vidStart, quantity, seqLength )
	assert( quantity > 0 )
	assert( ( quantity % seqLength ) == 0 )
	local dataTable = {  }
	local scalarTable = {  }
	local numVideoToSample = quantity / seqLength
	local numVideo = self.vid2path:size( 1 )
	for v = 1, numVideoToSample do
		local vid = vidStart + v - 1
		local vpath = ffi.string( torch.data( self.vid2path[ vid ] ) )
		local numFrame = self.vid2numim[ vid ]
		local cid = self.vid2cid[ vid ]
		local startFrame = torch.random( 1, math.max( 1, numFrame - seqLength + 1 ) )
		for f = 1, seqLength do
			local fid = math.min( numFrame, startFrame + f - 1 )
			local fpath = paths.concat( vpath, string.format( self.frameFormat, fid ) )
			local out = self:sampleHookTest( fpath )
			table.insert( dataTable, out )
			table.insert( scalarTable, cid )
		end
	end
	local data, scalarLabels = tableToOutput( self, dataTable, scalarTable )
	return data, scalarLabels
end
function dataset:getVideo( vid )
	local numFrame = self.vid2numim[ vid ]
	local video = torch.Tensor( numFrame, 3, self.sampleSize, self.sampleSize )
	local vpath = ffi.string( torch.data( self.vid2path[ vid ] ) )
	for f = 1, numFrame do
		local fpath = paths.concat( vpath, string.format( self.frameFormat, f ) )
		local out = self:sampleHookTest( fpath )
		video[ f ]:copy( out )
	end
	local label = torch.Tensor( { self.vid2cid[ vid ] } )
	return video, label
end
evaluateBatch = function( fid2out, fid2gt, seqLength )
	if type( fid2out ) == 'table' then
		for l = 2, #fid2out do fid2out[ 1 ]:add( fid2out[ l ] ) end
		fid2out = fid2out[ 1 ] / #fid2out
	end
	local _, fid2pcid = fid2out:float(  ):sort( 2, true )
	local batchSize = fid2out:size( 1 )
	local numVideo = batchSize / seqLength
	local vid2true = torch.zeros( numVideo, 1 )
	local top1 = 0
	for v = 1, numVideo do
		local fbias = ( v - 1 ) * seqLength
		local pcid2num = torch.zeros( fid2out:size( 2 ) )
		local cid = fid2gt[ fbias + 1 ]
		for f = 1, seqLength do
			local fid = fbias + f
			local pcid = fid2pcid[ fid ][ 1 ]
			pcid2num[ pcid ] = pcid2num[ pcid ] + 1
		end
		local _, rank2pcid = pcid2num:sort( true )
		if cid == rank2pcid[ 1 ] then top1 = top1 + 1 end
	end
	top1 = top1 * 100 / numVideo
	return top1
end
evaluateVideo = function( outputs, label, seqLength )
	if type( outputs ) == 'table' then
		for l = 2, #outputs do outputs[ 1 ]:add( outputs[ l ] ) end
		outputs = outputs[ 1 ] / #outputs
	end
	local numFrame = outputs:size( 1 )
	local numCls = outputs:size( 2 )
	local numSeq = numFrame / seqLength
	local topn = math.min( 10, numCls )
	local seqPool = 'sum'
	local vidPool = 'sum'
	assert( numSeq % 1 == 0 )
	-- Sequence-level pooling.
	local sid2out = torch.Tensor( numSeq, numCls )
	for s = 1, numSeq do
		local fstart = ( s - 1 ) * seqLength + 1
		local fend = fstart + seqLength - 1
		if seqPool == 'sum' then
			sid2out[ s ] = outputs[ { { fstart, fend } } ]:sum( 1 )
		elseif seqPool == 'max' then
			sid2out[ s ] = outputs[ { { fstart, fend } } ]:max( 1 )
		elseif seqPool == 'last' then
			sid2out[ s ] = outputs[ fend ]
		end
	end
	-- Video-level pooling.
	if vidPool == 'sum' then
		cid2score = sid2out:sum( 1 )
	elseif vidPool == 'max' then
		cid2score = sid2out:max( 1 )
	end
	-- Make and evaluate predictions.
	local _, rank2pcid = cid2score:sort( true )
	local topPreds = rank2pcid[ 1 ][ { { 1, topn } } ]
	local pcid = rank2pcid[ 1 ][ 1 ]
	local top1 = 0
	if pcid == label[ 1 ] then top1 = 1 end
	return top1, topPreds, cid2score
end
return dataset
