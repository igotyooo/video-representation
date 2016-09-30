require 'paths'
require 'sys'
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

local function readLinesFrom( fpath )
	lines = {}
	for line in io.lines( fpath ) do
		lines[#lines + 1] = line
	end
	return lines
end

function genDb( setName )
	local setid
	if setName == 'train' then setid = 1 else setid = 2 end
	local rootDir = gpath.db.ucf101
	local frameFormat = '%04d.jpg'
	local imFormat = frameFormat:match( '.+%.(.+)' )
	local cid2name  = readLinesFrom( paths.concat( rootDir, 'db_cid2name.txt' ), 'r' )
	local vid2path = readLinesFrom( paths.concat( rootDir, 'db_vid2path.txt' ), 'r' )
	local vid2cid = readLinesFrom( paths.concat( rootDir, 'db_vid2cid.txt' ), 'r' )
	local vid2setid = readLinesFrom( paths.concat( rootDir, 'db_vid2setid.txt' ), 'r' )
	assert( #vid2path == #vid2cid )
	assert( #vid2path == #vid2setid )
	local svid2path, svid2numim, svid2cid = {  }, {  }, {  }
	for vid, vpath in pairs( vid2path ) do
		if tonumber( vid2setid[ vid ] ) == setid then
			vpath_ = paths.concat( rootDir, vpath )
			svid2path[ #svid2path + 1 ] = vpath_
			svid2cid[ #svid2cid + 1 ] = tonumber( vid2cid[ vid ] )
			local numim = tonumber( sys.fexecute( ( 'find %s | wc -l' ):format( paths.concat( vpath_, '*.' .. imFormat ) ) ) )
			assert( numim > 0 )
			svid2numim[ #svid2numim + 1 ] = numim
		end
	end
	-- Convert string tables to tensors. Fuck lua table!!! 
	local svid2path_ = strTableToTensor( svid2path )
	local cid2name_ = strTableToTensor( cid2name )
	svid2numim = torch.LongTensor( svid2numim )
	svid2cid = torch.LongTensor( svid2cid )
	-- Manual clean up. Fuck lua gargabe collection!!!
	for i = 1, #svid2path do svid2path[ i ] = nil end svid2path = svid2path_
	for i = 1, #cid2name do cid2name[ i ] = nil end cid2name = cid2name_
	collectgarbage(  )
	return svid2path, svid2numim, svid2cid, cid2name, frameFormat
end
