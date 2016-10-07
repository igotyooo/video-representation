require 'loadcaffe'
function createModel( nGPU )
	-- Set params.
	local fcSize = 4096
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
	-- Out: ( numVideo X seqLength ), numCls
	local fc = nn.Sequential(  )
	fc:add( nn.Linear( 4096, fcSize ) )
	fc:add( nn.Threshold( 0, 1e-6 ) )
  fc:add( nn.Dropout( 0.5 ) )
	fc:add( nn.Linear( fcSize, numCls ) )
  fc:add( nn.LogSoftMax(  ) ) -- Log??
	fc:cuda(  )

	-- Combine sub models.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), numCls
   local model = nn.Sequential(  )
	model:add( features )
	model:add( fc )
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
	optims[ 1 ] = {
		learningRate = opt.lrFeature,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 2 ] = {
		learningRate = opt.lrLinear,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	return params, grads, optims
end
