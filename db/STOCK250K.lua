require 'paths'
require 'torch'
require 'xlua'
local ffi = require 'ffi'

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

genDb = function( setName )
	local setid
	if setName == 'train' then setid = 1 else setid = 2 end
	local videoRoot = '/home/doyoo/ssd/stock-frames/00/'
	local metaPath = '/home/doyoo/workspace/src/data-transfer/videos_from_40_to_90.txt'
	local classPath = '/home/doyoo/workspace/datain/STOCK/category.csv'
	local frameFormat = 'frame-%05d.jpg'
	local minNumVidPerCls = 100
	local minNumImPerVid = 16
	local numVal = 20000
	
	print( 'Find videos and read meta.' )
	local vid = 0
	local vid2path, vid2numim, vid2ccodes, vid2numc = {  }, {  }, {  }, {  }
	local numVideoTotal = 0
	for line in io.lines( metaPath ) do
		numVideoTotal = numVideoTotal + 1
		local meta = line:split( ',' )
		for k,v in pairs( meta ) do meta[ k ] = tonumber( v ) end
		meta = torch.LongTensor( meta )
		local vcode = meta[ 1 ]
		local numim = meta[ 2 ]
		local ccodes = meta[ { { 3, 11 } } ]
		-- Basic filtering.
		if vcode < 1e7 then goto continue end
		if numim < minNumImPerVid then goto continue end
		if ccodes:gt( 0 ):sum(  ) == 0 then goto continue end
		-- Check if video exists.
		local vpath = videoRoot
		local vcode_pre = tostring( math.floor( vcode / 100 ) )
		vcode_pre:gsub( '..', function( c ) vpath = paths.concat( vpath, c ) end )
		vpath = paths.concat( vpath, tostring( vcode ) )
		local ipath = paths.concat( vpath, string.format( frameFormat, numim ) )
		--if not paths.filep( ipath ) then print( 'Warning: No ' .. ipath ) goto continue end
		-- Now, store information.
		vid = vid + 1
		vid2path[ vid ] = vpath
		vid2numim[ vid ] = numim
		vid2ccodes[ vid ] = ccodes[ ccodes:gt( 0 ) ]
		if vid % 1e4 == 0 then print( vid .. ' videos found.' ) end
		::continue::
	end
	collectgarbage(  )
	print( string.format( '%d videos found in total. %d/%d videos skipped.', #vid2path, numVideoTotal - #vid2path, numVideoTotal ) )

	print( 'Define class ids and associate videos with them.' )
	local code2numv = {  }
	local cid2code = {  }
	local code2cid = {  }
	for vid, codes in pairs( vid2ccodes ) do
		for c = 1, codes:size( 1 ) do 
			local code = codes[ c ]
			if code2numv[ code ] then
				code2numv[ code ] = code2numv[ code ] + 1
			else
				code2numv[ code ] = 1
			end
		end
	end
	local cid = 0
	for code, numv in pairs( code2numv ) do
		if numv >= minNumVidPerCls then
			cid = cid + 1
			code2cid[ code ] = cid
			cid2code[ cid ] = code
		else
			code2cid[ code ] = 0
		end
	end
	local vid2cids = {  }
	for vid, codes in pairs( vid2ccodes ) do
		vid2cids[ vid ] = torch.LongTensor( codes:size( 1 ) ):zero(  )
		for c = 1, codes:size( 1 ) do 
			vid2cids[ vid ][ c ] = code2cid[ codes[ c ] ] 
		end
	end
	local numReject = 0
	for vid, cids in pairs( vid2cids ) do
		if cids:gt( 0 ):sum(  ) == 0 then
			numReject = numReject + 1
			vid2cids[ vid ] = nil
			vid2path[ vid ] = nil
			vid2numim[ vid ] = nil
		else
			vid2cids[ vid ] = cids[ cids:gt( 0 ) ]
		end
	end
	collectgarbage(  )
	assert( #vid2path == #vid2cids )
	assert( #vid2path == #vid2numim )
	print( string.format( '%d classes defined in total.', #cid2code ) )
	print( string.format( '%d none-class videos rejected.', numReject ) )

	print( string.format( 'Sample videos for %s.', setName ) )
	local numVideo = #vid2path
	local valSampleStep = math.floor( numVideo / numVal )
	local svid2path, svid2numim, svid2cids = {  }, {  }, {  }
	local svid = 0
	for vid, _ in pairs( vid2path ) do
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
	assert( #vid2path == #vid2cids )
	assert( #vid2path == #vid2numim )
	print( string.format( '%d videos Sampled for %s.', #svid2path, setName ) )
	collectgarbage(  )

	print( 'Read class name.' )
	local code2name = {  }
	for line in io.lines( classPath ) do
		local meta = line:split( ',' )
		local code, name
		if #meta == 5 or #meta == 4 then
			code = meta[ 1 ]
			name = meta[ 2 ]
		elseif #meta > 5 then
			code, name = line:match( '(%d+),"(.-)",".-",%d+,%d+' )
		else
			print( 'Skip: ' .. line ) goto continue
		end
		code = tonumber( code )
		-- Basic filtering.
		if code < 1 then goto continue end
		if #name < 1 then goto continue end
		if type( code ) ~= 'number' then print( code ) goto continue end
		if type( name ) ~= 'string' then print( code ) goto continue end
		-- Store information.
		code2name[ code ] = name
		::continue::
	end
	local numCls = 0
	for _ in pairs( code2name ) do numCls = numCls + 1 end -- <-- ....
	local cid2name = {  }
	for cid, code in pairs( cid2code ) do
		local name = code2name[ code ]
		assert( type( name ) == 'string' )
		cid2name[ cid ] = code2name[ code ]
	end
	collectgarbage(  )
	print( 'Done' )

	-- Convert string tables to tensors. Fuck lua table!!! 
	local svid2path_ = strTableToTensor( svid2path )
	local cid2name_ = strTableToTensor( cid2name )
	local numMaxCls = 0
	local numVid = #svid2cids
	for _, cids in pairs( svid2cids ) do
		if numMaxCls < cids:size( 1 ) then numMaxCls = cids:size( 1 ) end
	end
	local svid2cids_ = torch.LongTensor( numVid, numMaxCls ):fill( 0 )
	for vid, cids in pairs( svid2cids ) do
		svid2cids_[ vid ][ { { 1, cids:size( 1 ) } } ]:copy( cids )
	end
	svid2numim = torch.LongTensor( svid2numim )

	-- Manual clean up. Fuck lua gargabe collection!!!
	for i = 1, #svid2path do svid2path[ i ] = nil end svid2path = svid2path_
	for i = 1, #cid2name do cid2name[ i ] = nil end cid2name = cid2name_
	for i = 1, #svid2cids do svid2cids[ i ] = nil end svid2cids = svid2cids_
	collectgarbage(  )
	return svid2path, svid2numim, svid2cids, cid2name, frameFormat
end
