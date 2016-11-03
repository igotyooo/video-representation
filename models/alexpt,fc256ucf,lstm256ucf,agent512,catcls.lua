require 'loadcaffe'
function createModel( nGPU )
	-- Set params.
	local featSize = 4096
	local hiddenSize = 256
	local numCls = opt.nClasses
	local seqLength = opt.seqLength
	local numVideo = opt.batchSize / seqLength
	local agentBound = opt.agentBound

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
	-- Out: ( numVideo X seqLength ), hiddenSize
	local fc = torch.load( '/home/dgyoo/workspace/dataout/video-representation-cvpr/UCF101/alexpt,fc256,cls,batchSize=256,epochSize=2384,lrAgent=0.00016,lrClassifier=0.00016,lrFc=0.00016,nEpochs=120/model_2.t7' )
	fc = fc.modules[ 2 ]
	fc:cuda(  )

	-- Create LSTM tower.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), hiddenSize
	local lstm = torch.load( '/home/dgyoo/workspace/dataout/video-representation-cvpr/UCF101/alexpt,lstm256,cls,batchSize=256,epochSize=2384,lrAgent=0.00016,lrClassifier=0.00016,lrFc=0.00016,nEpochs=120/model_2.t7' )
	lstm = lstm.modules[ 2 ]
	lstm:cuda(  )
	
	-- Concat FC-LSTM.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), ( hiddenSize X 2 )
	local fclstm = nn.Sequential(  )
	fclstm:add( nn.ConcatTable(  ):add( fc ):add( lstm ) )
	fclstm:add( nn.JoinTable( 2 ) )
	fclstm:cuda(  )

	-- Create agent tower.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), ( hiddenSize X 2 )
	local agent = nn.Sequential(  )
	agent:add( nn.Reshape( numVideo, seqLength * featSize ) ) -- numVideo, ( seqLength X featSize )
	agent:add( nn.Linear( seqLength * featSize, hiddenSize * 2 ) ) -- numVideo, hiddenSize X 2
	agent:add( nn.Sigmoid(  ) )
	agent:add( nn.Replicate( seqLength, 2 ) )
	agent:add( nn.Reshape( numVideo * seqLength, hiddenSize * 2 ) )
	agent:cuda(  )

	-- Apply agent.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), ( hiddenSize X 2 )
	local fclstmagent = nn.Sequential(  )
	fclstmagent:add( nn.ConcatTable(  ):add( fclstm ):add( agent ) )
	fclstmagent:add( nn.CMulTable(  ) )
	fclstmagent:cuda(  )

	-- Create agent classifier.
	-- In:  ( numVideo X seqLength ), ( hiddenSize X 2 )
	-- Out: ( numVideo X seqLength ), numClass
	local classifierAgent = nn.Sequential(  )
	classifierAgent:add( nn.Linear( hiddenSize * 2, numCls ) )
	classifierAgent:add( nn.LogSoftMax(  ) )
	classifierAgent:cuda(  )

	-- Combine sub models.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), numClass
   local model = nn.Sequential(  )
	model:add( features )
	model:add( fclstmagent )
	model:add( classifierAgent )
	model:cuda(  )

	-- Check options.
	assert( opt.numLoss == 1 )
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
	params[ 2 ], grads[ 2 ] = model.modules[ 2 ].modules[ 1 ]:modules[ 1 ].modules[ 1 ].modules[ 1 ]:getParameters(  ) -- FC.
	params[ 3 ], grads[ 3 ] = model.modules[ 2 ].modules[ 1 ]:modules[ 1 ].modules[ 1 ].modules[ 2 ]:getParameters(  ) -- LSTM.
	params[ 4 ], grads[ 4 ] = model.modules[ 2 ].modules[ 1 ]:modules[ 2 ]:getParameters(  ) -- Agent.
	params[ 5 ], grads[ 5 ] = model.modules[ 3 ]:getParameters(  ) -- Classifier.
	local tdb = require( 'fb.debugger' ) tdb.enter(  )
	optims[ 1 ] = { -- Features.
		learningRate = opt.lrFeature,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 2 ] = { -- FC.
		learningRate = opt.lrFc,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 3 ] = { -- LSTM.
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
	optims[ 5 ] = { -- Classifier.
		learningRate = opt.lrClassifier,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	return params, grads, optims
end