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
	self.vid2path, self.vid2numim, self.vid2cids, self.cid2name, self.frameFormat = genDb( setName )
end
-- Converts a table of samples and labels to a clean tensor.
local function tableToOutput( self, dataTable, labelTable )
	local data, labels
	local quantity = #labelTable
	local numDim = self.cid2name:size( 1 )
	assert( dataTable[ 1 ]:dim(  ) == 3 )
	assert( #dataTable == #labelTable )
	data = torch.Tensor( quantity, 3, self.sampleSize, self.sampleSize )
	labels = torch.LongTensor( quantity, numDim ):zero(  )
	for i = 1, quantity do
		data[ i ]:copy( dataTable[ i ] )
		local cids = labelTable[ i ][ labelTable[ i ]:gt( 0 ) ]
		labels[ i ][ { { 1, cids:size( 1 ) } } ]:copy( cids )
	end
	return data, labels
end
function dataset:sampleVideos( quantity, seqLength )
	assert( quantity > 0 )
	assert( ( quantity % seqLength ) == 0 )
	local dataTable = {  }
	local labelTable = {  }
	local numVideoToSample = quantity / seqLength
	local numVideo = self.vid2path:size( 1 )
	for v = 1, numVideoToSample do
		local vid = torch.random( 1, numVideo )
		local vpath = ffi.string( torch.data( self.vid2path[ vid ] ) )
		local numFrame = self.vid2numim[ vid ]
		local cids = self.vid2cids[ vid ]
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
			table.insert( labelTable, cids )
		end
	end
	local data, labels = tableToOutput( self, dataTable, labelTable )
	return data, labels
end
function dataset:get( vidStart, quantity, seqLength )
	assert( quantity > 0 )
	assert( ( quantity % seqLength ) == 0 )
	local dataTable = {  }
	local labelTable = {  }
	local numVideoToSample = quantity / seqLength
	local numVideo = self.vid2path:size( 1 )
	for v = 1, numVideoToSample do
		local vid = vidStart + v - 1
		local vpath = ffi.string( torch.data( self.vid2path[ vid ] ) )
		local numFrame = self.vid2numim[ vid ]
		local cids = self.vid2cids[ vid ]
		local startFrame = torch.random( 1, math.max( 1, numFrame - seqLength + 1 ) )
		for f = 1, seqLength do
			local fid = math.min( numFrame, startFrame + f - 1 )
			local fpath = paths.concat( vpath, string.format( self.frameFormat, fid ) )
			local out = self:sampleHookTest( fpath )
			table.insert( dataTable, out )
			table.insert( labelTable, cids )
		end
	end
	local data, labels = tableToOutput( self, dataTable, labelTable )
	return data, labels
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
	local labels = self.vid2cids[ vid ][ self.vid2cids[ vid ]:gt( 0 ) ]
	return video, labels
end
evaluateBatch = function( fid2out, fid2cids, seqLength )
	local numVideo = fid2out:size( 1 ) / seqLength
	local numDim = fid2out:size( 2 )
	local map = 0
	for v = 1, numVideo do
		local fstart = ( v - 1 ) * seqLength + 1
		local fend = fstart + seqLength - 1
		local cids = fid2cids[ fstart ][ fid2cids[ fstart ]:gt( 0 ) ]
		local _, rank2pcid = fid2out[ { { fstart, fend } } ]:sum( 1 ):sort( true )
		local ap = 0
		local numTrue = 0
		local rank = 0
		for r = 1, numDim do
			numTrue = numTrue + cids:eq( rank2pcid[ 1 ][ r ] ):sum(  )
			ap = ap + numTrue / r
			if cids:size( 1 ) == numTrue then rank = r break end
		end
		map = map + ap / rank
	end
	map = map * 100 / numVideo
	return map
end
evaluateVideo = function( outputs, labels, features, seqLength )
	local numFrame = outputs:size( 1 )
	local numCls = outputs:size( 2 )
	local dimFeat = features[ 1 ]:numel(  )
	local numSeq = numFrame / seqLength
	local labels = labels[ labels:gt( 0 ) ]
	local topn = math.min( 10, numCls )
	local seqPool = 'sum'
	local vidPool = 'sum'
	assert( numSeq % 1 == 0 )
	-- Sequence-level pooling.
	local sid2out = torch.Tensor( numSeq, numCls )
	local sid2feat = torch.Tensor( numSeq, dimFeat )
	for s = 1, numSeq do
		local fstart = ( s - 1 ) * seqLength + 1
		local fend = fstart + seqLength - 1
		if seqPool == 'sum' then
			sid2out[ s ] = outputs[ { { fstart, fend } } ]:sum( 1 )
			sid2feat[ s ] = features[ { { fstart, fend } } ]:view( -1, dimFeat ):sum( 1 )
		elseif seqPool == 'max' then
			sid2out[ s ] = outputs[ { { fstart, fend } } ]:max( 1 )
			sid2feat[ s ] = features[ { { fstart, fend } } ]:view( -1, dimFeat ):max( 1 )
		elseif seqPool == 'last' then
			sid2out[ s ] = outputs[ fend ]
			sid2feat[ s ] = features[ fend ]:view( -1, dimFeat )
		end
	end
	-- Video-level pooling.
	local cid2score = nil
	local feature = nil
	if vidPool == 'sum' then
		cid2score = sid2out:sum( 1 )
		feature = sid2feat:sum( 1 )
	elseif vidPool == 'max' then
		cid2score = sid2out:max( 1 )
		feature = sid2feat:max( 1 )
	end
	-- Make and evaluate predictions.
	local _, rank2pcid = cid2score:sort( true )
	local topPreds = rank2pcid[ 1 ][ { { 1, topn } } ]
	local ap = 0
	local numTrue = 0
	local rank = 0
	for r = 1, numCls do
		numTrue = numTrue + labels:eq( rank2pcid[ 1 ][ r ] ):sum(  )
		ap = ap + numTrue / r
		if labels:size( 1 ) == numTrue then rank = r break end
	end
	ap = ap / rank
	return ap, topPreds, cid2score, feature
end
return dataset
