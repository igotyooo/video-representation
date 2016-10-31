require 'loadcaffe'
function createModel( nGPU )
	-- Set params.
	local featSize = 4096
	local fcSize = 4096
	local lstmSize = 256
	local numCls = opt.nClasses
	local seqLength = opt.seqLength
	local numVideo = opt.batchSize / seqLength

	-- Load pre-trained CNN.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), featSize
	print( 'Load pre-trained Caffe feature.' )
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
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), lstmSize
	local fc = nn.Sequential(  )
	fc:add( nn.Linear( featSize, lstmSize ) )
	fc:add( nn.ReLU(  ) )
	fc:add( nn.Dropout( 0.5 ) )
	fc:add( nn.Copy( nil, nil, true ) )
	fc:cuda(  )

	-- Create LSTM tower.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), lstmSize
	local lstm = nn.Sequential(  )
	lstm:add( nn.View( -1, seqLength, featSize ) ) -- LSTM input is numVideo, SeqLength, vectorDim
	lstm:add( nn.LSTM( featSize, lstmSize ) )
	lstm:add( nn.Tanh(  ) )
	lstm:add( nn.Dropout( 0.5 ) )
	lstm:add( nn.View( -1, lstmSize ) )
	lstm:add( nn.Copy( nil, nil, true ) )
	lstm:cuda(  )

	-- Parallelize FC and LSTM.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), ( lstmSize X 2 )
	local fclstm = nn.Concat( 2 )
	fclstm:add( fc )
	fclstm:add( lstm )
	fclstm:cuda(  )

	-- Create agent.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), ( lstmSize X 2 )
	local agent = nn.Sequential(  )
	agent:add( nn.View( -1, seqLength * featSize ) ) -- numVideo, ( seqLength X featSize )
	agent:add( nn.Linear( seqLength * featSize, 2 ) ) -- numVideo, 2
	agent:add( nn.Sigmoid(  ) )
	agent:add( nn.Reshape( 2 * numVideo ) )
	agent:add( nn.Replicate( lstmSize, 2 ) )
	agent:add( nn.Reshape( numVideo, lstmSize * 2 ) )
	agent:add( nn.Replicate( seqLength, 1, 1 ) )
	agent:add( nn.Reshape( seqLength * numVideo, lstmSize * 2 ) )
	agent:cuda(  )

	-- Parallelize FC-LSTM and agent.
	-- In:    ( numVideo X seqLength ), featSize
	-- Out 1: ( numVideo X seqLength ), ( lstmSize X 2 )
	-- Out 2: ( numVideo X seqLength ), ( lstmSize X 2 )
	local fclstmagent = nn.ConcatTable(  )
	fclstmagent:add( fclstm )
	fclstmagent:add( agent )
	fclstmagent:cuda(  )

	-- Create classifier.
	-- In 1: ( numVideo X seqLength ), ( lstmSize X 2 )
	-- In 2: ( numVideo X seqLength ), ( lstmSize X 2 )
	-- Out:  ( numVideo X seqLength ), numCls
	local classifier = nn.Sequential(  )
	classifier:add( nn.CMulTable(  ) )
	classifier:add( nn.Linear( lstmSize * 2, numCls ) ) -- numVideo, 2
	classifier:add( nn.LogSoftMax(  ) ) -- Log??
	classifier:cuda(  )

	-- Combine sub models.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), numCls
   local model = nn.Sequential(  )
	model:add( features )
	model:add( fclstmagent )
	model:add( classifier )
	model:cuda(  )
	
	-- Check options.
	assert( opt.inputIsCaffe )
	assert( not opt.normalizeStd )
	assert( not opt.keepAspect )
	collectgarbage(  )
   return model
end

function groupParams( model )
	local params, grads, optims = {  }, {  }, {  }
	params[ 1 ], grads[ 1 ] = model.modules[ 1 ]:getParameters(  ) -- Features.
	params[ 2 ], grads[ 2 ] = model.modules[ 2 ].modules[ 1 ].modules[ 1 ]:getParameters(  ) -- FC.
	params[ 3 ], grads[ 3 ] = model.modules[ 2 ].modules[ 1 ].modules[ 2 ]:getParameters(  ) -- LSTM.
	params[ 4 ], grads[ 4 ] = model.modules[ 2 ].modules[ 2 ]:getParameters(  ) -- Agent.
	params[ 5 ], grads[ 5 ] = model.modules[ 3 ]:getParameters(  ) -- Final classifier.
	optims[ 1 ] = { -- Features.
		learningRate = opt.lrFeature,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 2 ] = { -- FC body.
		learningRate = opt.lrFc,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 3 ] = { -- LSTM body.
		learningRate = opt.lrLstm,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 4 ] = { -- Agent.
		learningRate = opt.lrAgent,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 5 ] = { -- Final classifier.
		learningRate = opt.lrClassifier,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	return params, grads, optims
end
