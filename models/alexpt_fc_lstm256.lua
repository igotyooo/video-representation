require 'loadcaffe'
function createModel( nGPU )
	-- Set params.
	local fcSize = 4096
	local lstmSize = 256
	local numCls = opt.nClasses

	-- Load pre-trained CNN.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), 4096
	local proto = gpath.net.alex_caffe_proto
	local caffemodel = gpath.net.alex_caffe_model
	local features = loadcaffe.load( proto, caffemodel, opt.backend )
	features:remove(  )
	features:remove(  )
	features:remove(  )
	features:remove(  )
	features:remove(  )
	features:cuda(  )
	features = makeDataParallel( features, nGPU )

	-- Create FC tower.
	-- In:  ( numVideo X seqLength ), 4096
	-- Out: ( numVideo X seqLength ), numCls, 1
	local fc = nn.Sequential(  )
	fc:add( nn.Linear( 4096, fcSize ) )
	fc:add( nn.Threshold( 0, 1e-6 ) )
   fc:add( nn.Dropout( 0.5 ) )
	fc:add( nn.Linear( fcSize, numCls ) )
	fc:add( nn.View( -1, numCls, 1 ) )
	fc:add( nn.Copy( nil, nil, true ) )
	fc:cuda(  )

	-- Create LSTM tower.
	-- In:  ( numVideo X seqLength ), 4096
	-- Out: ( numVideo X seqLength ), numCls, 1
	local lstm = nn.Sequential(  )
	lstm:add( nn.View( -1, opt.seqLength, 4096 ) ) -- LSTM input is numVideo, SeqLength, vectorDim
	lstm:add( nn.LSTM( 4096, lstmSize ) )
	lstm:add( nn.View( -1, lstmSize ) )
	lstm:add( nn.Dropout( 0.5 ) )
	lstm:add( nn.Linear( lstmSize, numCls ) )
	lstm:add( nn.View( -1, numCls, 1 ) )
	lstm:add( nn.Copy( nil, nil, true ) )
	lstm:cuda(  )
	
	-- Parallelize FC, LSTM.
	-- In:  ( numVideo X seqLength ), 4096
	-- Out: ( numVideo X seqLength ), numCls, 2
	local fclstm = nn.Concat( 3 )
	fclstm:add( fc )
	fclstm:add( lstm )
	fclstm:cuda(  )

	-- Fusion.
	-- In:  ( numVideo X seqLength ), numCls, 2
	-- Out: ( numVideo X seqLength ), numCls
	local fusion = nn.Sequential(  )
	fusion:add( nn.CMul( 1, numCls, 2 ) )
	fusion:add( nn.Mean( 3 ) )
	fusion:add( nn.LogSoftMax(  ) ) -- Log??
	fusion:cuda(  )
	
	-- Combine sub models.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), numCls
   local model = nn.Sequential(  )
	model:add( features )
	model:add( fclstm )
	model:add( fusion )
	model:cuda(  )

	-- Check options.
	assert( opt.inputIsCaffe )
	assert( not opt.normalizeStd )
	assert( not opt.keepAspect )
   return model
end
