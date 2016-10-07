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

	-- Create LSTM tower.
	-- In:  ( numVideo X seqLength ), 4096
	-- Out: ( numVideo X seqLength ), lstmSize
	local lstm = nn.Sequential(  )
	lstm:add( nn.View( -1, opt.seqLength, 4096 ) ) -- LSTM input is numVideo, SeqLength, vectorDim
	lstm:add( nn.LSTM( 4096, lstmSize ) )
	lstm:add( nn.View( -1, lstmSize ) )
	lstm:add( nn.Dropout( 0.5 ) )
	lstm:cuda(  )
	
	-- Create classifier.
	-- In:  ( numVideo * seqLength ), lstmSize
	-- out: ( numVideo * seqLength ), numCls
	local classifier = nn.Sequential(  )
	classifier:add( nn.Linear( lstmSize, numCls ) )
	classifier:add( nn.LogSoftMax(  ) )
	classifier:cuda(  )
	
	-- Combine sub models.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), numCls
	local model = nn.Sequential(  )
	model:add( features )
	model:add( lstm )
	model:add( classifier )
	model:cuda(  )

	-- Check options.
	assert( opt.inputIsCaffe )
	assert( not opt.normalizeStd )
	assert( not opt.keepAspect )
   return model
end

function groupParams( model )
	local params, grads, optims = {  }, {  }, {  }
	params[ 1 ], grads[ 1 ] = model.modules[ 1 ]:getParameters(  )
	params[ 2 ], grads[ 2 ] = model.modules[ 2 ]:getParameters(  )
	params[ 3 ], grads[ 3 ] = model.modules[ 3 ]:getParameters(  )
	optims[ 1 ] = {
		learningRate = opt.lrFeature,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 2 ] = {
		learningRate = opt.lrLstm,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 3 ] = {
		learningRate = opt.lrLinear,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	return params, grads, optims
end
