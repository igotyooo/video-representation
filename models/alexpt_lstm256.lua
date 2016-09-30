require 'loadcaffe'
function createModel( nGPU )
	-- Load pre-trained CNN.
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
	-- Create LSTM
	local lstmNumHidden = 256
	local seqLength = opt.seqLength
	local nClasses = opt.nClasses
	local lstm = nn.Sequential(  )
	lstm:add( nn.View( -1, seqLength, 4096 ) ) -- LSTM input is NumVideo X SeqLength X vectorDim
	lstm:add( nn.LSTM( 4096, lstmNumHidden ) )
	lstm:add( nn.View( -1, lstmNumHidden ) )
	lstm:add( nn.Dropout( 0.5 ) )
	lstm:cuda(  )
	-- Create classifier.
	local classifier = nn.Sequential(  )
	classifier:add( nn.Linear( lstmNumHidden, nClasses ) )
	classifier:add( nn.LogSoftMax(  ) )
	classifier:cuda(  )
	-- Combine sub models.
	local model = nn.Sequential(  ):add( features ):add( lstm ):add( classifier )
	-- Check options.
	assert( opt.inputIsCaffe )
	assert( not opt.normalizeStd )
	assert( not opt.keepAspect )
   return model
end
