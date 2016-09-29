require 'paths'
require 'torch'
require 'xlua'
local ffi = require 'ffi'
-- Set paths and params.
local srcVideoLabelListPath = '/home/doyoo/workspace/src/tag-processing/data_ml50/video-labels.txt'
local srcLabelListPath = '/home/doyoo/workspace/src/tag-processing/data_ml50/classes.txt'
local videoRoot = '/home/doyoo/ssd/stock-frames/00/'
local frameFormat = 'frame-%05d.jpg'
local numVal = 20000

local function printl( str )
	print( 'STOCK250K_NEW) ' .. str )
end

local function strTableToTensor( strTable )
	local maxStrLen = 0
	local numStr = #strTable
	for _, path in pairs( strTable ) do
		if maxStrLen < path:len(  ) then maxStrLen = path:len(  ) end
	end
	maxStrLen = maxStrLen + 1
	local charTensor = torch.CharTensor( numStr, maxStrLen ):fill( 0 )
	local pt = charTensor:data(  )
	for _, path in pairs( strTable ) do
		ffi.copy( pt, path )
		pt = pt + maxStrLen
	end
	-- Manual clean up. Fuck lua gargabe collection!!!
	for i = 1, #strTable do strTable[ i ] = nil end strTable = nil
	collectgarbage(  )
	return charTensor
end

local function readVideoAndLabelList(  )
	printl( 'Read video-label list.' )
	local vid = 0
	local vid2path = {  }
	local vid2numim = {  }
	local vid2cids = {  }
	for line in io.lines( srcVideoLabelListPath ) do
		nums = line:split( ' ' )
		local code = nums[ 1 ]
		local numim = tonumber( nums[ 2 ] )
		local numcls = tonumber( nums[ 3 ] )
		assert( numcls == #nums - 3 )
		assert( code:len(  ) == 8 )
		local p1, p2, p3 = code:match( '^(%d%d)(%d%d)(%d%d)%d%d$' )
		local path = paths.concat( videoRoot, p1, p2, p3, code )
		local cids = {  }
		for ind = 4, #nums do
			cids[ #cids + 1 ] = tonumber( nums[ ind ] )
		end
		vid = vid + 1
		vid2path[ vid ] = path
		vid2numim[ vid ] = numim
		vid2cids[ vid ] = torch.LongTensor( cids )
		if vid % 50000 == 0 then 
			printl( ( '%d videos found.' ):format( vid ) )
		end
	end
	printl( ( '%d videos found in total.' ):format( vid ) )
	collectgarbage(  )
	return vid2path, vid2numim, vid2cids
end

local function readLabelList(  )
	printl( 'Read label names.' )
	local cid, cid2name  = 0, {  }
	for line in io.lines( srcLabelListPath ) do
		cid = cid + 1
		cid2name[ cid ] = line
	end
	printl( 'Done.' )
	collectgarbage(  )
	return cid2name
end

function genDb( setName )
	local setid = nil
	if setName == 'train' then 
		setid = 1 
	elseif setName == 'val' then
		setid = 2 
	end
	-- Read data from text.
	local vid2path, vid2numim, vid2cids = 
		readVideoAndLabelList(  )
	local cid2name = readLabelList(  )
	-- Separating train/val.
	local numVideo = #vid2path
	local svid2path = {  }
	local svid2numim = {  }
	local svid2cids = {  }
	local svid = 0
	local valSampleStep = math.floor( numVideo / numVal )
	for vid, path in pairs( vid2path ) do
		if setid == 1 and vid % valSampleStep ~= 0 then
			svid = svid + 1
			svid2path[ svid ] = vid2path[ vid ]
			svid2numim[ svid ] = vid2numim[ vid ]
			svid2cids[ svid ] = vid2cids[ vid ]
		elseif setid == 2 and vid % valSampleStep == 0 then
			svid = svid + 1
			svid2path[ svid ] = vid2path[ vid ]
			svid2numim[ svid ] = vid2numim[ vid ]
			svid2cids[ svid ] = vid2cids[ vid ]
		end
	end
	collectgarbage(  )
	printl( ( '%d videos chosen for %s.' ):format( svid, setName ) )
	-- Convert string tables to tensors. 
	printl( 'IMPORTANT! Convert tables to tensors.' )
	local svid2path = strTableToTensor( svid2path )
	local cid2name = strTableToTensor( cid2name )
	local numMaxCls = 0
	local numVid = #svid2cids
	for _, cids in pairs( svid2cids ) do
		if numMaxCls < cids:size( 1 ) then numMaxCls = cids:size( 1 ) end
	end
	local svid2cids_ = torch.LongTensor( numVid, numMaxCls ):fill( 0 )
	for svid, cids in pairs( svid2cids ) do
		svid2cids_[ svid ][ { { 1, cids:size( 1 ) } } ]:copy( cids )
		svid2cids[ svid ] = nil
	end
	svid2cids = svid2cids_
	svid2numim = torch.LongTensor( svid2numim )
	collectgarbage(  )
	printl( 'IMPORTANT! Done.' )
	return svid2path, svid2numim, svid2cids, cid2name, frameFormat
end
