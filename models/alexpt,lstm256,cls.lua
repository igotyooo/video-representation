require 'loadcaffe'
function createModel( nGPU )
	-- Set params.
	local featSize = 4096
	local hiddenSize = 256
	local numCls = opt.nClasses

	-- Load pre-trained CNN.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), featSize
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
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), hiddenSize
	local lstm = nn.Sequential(  )
	lstm:add( nn.View( -1, opt.seqLength, featSize ) ) -- LSTM input is numVideo, SeqLength, vectorDim
	lstm:add( nn.LSTM( featSize, hiddenSize ) )
	lstm:add( nn.Tanh(  ) )
	lstm:add( nn.View( -1, hiddenSize ) )
	lstm:add( nn.Dropout( 0.5 ) )
	lstm:cuda(  )
	
	-- Create LSTM classifier.
	-- In:  ( numVideo * seqLength ), hiddenSize
	-- out: ( numVideo * seqLength ), numCls
	local classifierLstm = nn.Sequential(  )
	classifierLstm:add( nn.Linear( hiddenSize, numCls ) )
	classifierLstm:add( nn.LogSoftMax(  ) )
	classifierLstm:cuda(  )
	
	-- Combine sub models.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), numCls
	local model = nn.Sequential(  )
	model:add( features )
	model:add( lstm )
	model:add( classifierLstm )
	model:cuda(  )

	-- Check options.
	assert( opt.inputIsCaffe )
	assert( not opt.normalizeStd )
	assert( not opt.keepAspect )
	collectgarbage(  )
   return model
end

function loadModel( modelPath )
	local model0 = createModel( opt.nGPU )
	local model = loadDataParallel( modelPath, opt.nGPU )
	model0:getParameters(  ):copy( model:getParameters(  ) )
	return model0
end

function groupParams( model )
	local params, grads, optims = {  }, {  }, {  }
	params[ 1 ], grads[ 1 ] = model.modules[ 1 ]:getParameters(  ) -- Features.
	params[ 2 ], grads[ 2 ] = model.modules[ 2 ]:getParameters(  ) -- LSTM.
	params[ 3 ], grads[ 3 ] = model.modules[ 3 ]:getParameters(  ) -- Classifier.
	optims[ 1 ] = { -- Features.
		learningRate = opt.lrFeature,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 2 ] = { -- LSTM.
		learningRate = opt.lrLstm,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 3 ] = { -- Classifier.
		learningRate = opt.lrClassifier,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	return params, grads, optims
end
