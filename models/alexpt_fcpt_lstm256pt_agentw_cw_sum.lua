require 'loadcaffe'
function createModel( nGPU )
	-- Set params.
	local featSize = 4096
	local fcSize = 4096
	local lstmSize = 256
	local numCls = opt.nClasses

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
	-- Out: ( numVideo X seqLength ), numCls, 1
	print( 'Load pre-trained FC.' )
	local ptFcPath = '/home/dgyoo/workspace/dataout/video-representation-cvpr/UCF101/alexpt_fc,batchSize=256,epochSize=2384,nEpochs=20/model_20.t7'
	local ptFc = torch.load( ptFcPath )
	local fc = ptFc.modules[ 2 ]:clone(  )	
	fc:add( ptFc.modules[ 3 ].modules[ 1 ]:clone(  ) )
	fc:add( nn.Copy( nil, nil, true ) )
	fc:cuda(  )

	-- Create LSTM tower.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), numCls, 1
	print( 'Load pre-trained LSTM.' )
	local ptLstmPath = '/home/dgyoo/workspace/dataout/video-representation-cvpr/UCF101/alexpt_lstm256,batchSize=256,epochSize=2384,nEpochs=20/model_20.t7'
	local ptLstm = torch.load( ptLstmPath )
	local lstm = ptLstm.modules[ 2 ]:clone(  )
	lstm:add( ptLstm.modules[ 3 ].modules[ 1 ]:clone(  ) )
	lstm:add( nn.Copy( nil, nil, true ) )
	lstm:cuda(  )

	-- Parallelize FC and LSTM.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), numCls, 2
	local fclstm_ = nn.Concat( 2 )
	fclstm_:add( fc )
	fclstm_:add( lstm )
	local fclstm = nn.Sequential(  )
	fclstm:add( fclstm_ )
	fclstm:add( nn.Add( numCls * 2 ) ) -- Class-dependent weighting.
	fclstm:add( nn.Reshape( opt.batchSize, 2, numCls ) )
	fclstm:add( nn.Transpose( { 2, 3 } ) )
	fclstm:cuda(  )

	-- Create agent.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), numCls, 2
	local agent = nn.Sequential(  )
	agent:add( nn.View( -1, opt.seqLength * featSize ) ) -- numVideo, ( seqLength X featSize )
	agent:add( nn.Linear( opt.seqLength * featSize, 2 ) ) -- numVideo, 2
	agent:add( nn.Mul(  ) ) -- Scaling input-dependent weights.
	agent:add( nn.Replicate( opt.seqLength * numCls, 1, 1 ) ) -- numVideo X seqLength X numClass, 2
	agent:add( nn.Reshape( opt.batchSize, numCls, 2 ) ) -- ( numVideo X seqLength ), numClass, 2
	agent:cuda(  )

	-- Parallelize FC-LSTM and agent.
	-- In:    ( numVideo X seqLength ), featSize
	-- Out 1: ( numVideo X seqLength ), numCls, 2
	-- Out 2: ( numVideo X seqLength ), numCls, 2
	local fclstmagent = nn.ConcatTable(  )
	fclstmagent:add( fclstm )
	fclstmagent:add( agent )
	fclstmagent:cuda(  )

	-- Create classifier.
	-- In 1: ( numVideo X seqLength ), numCls, 2
	-- In 2: ( numVideo X seqLength ), numCls, 2
	-- Out:  ( numVideo X seqLength ), numCls
	local classifier = nn.Sequential(  )
	classifier:add( nn.CAddTable(  ) )
	classifier:add( nn.Sum( 3 ) )
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
	params[ 2 ], grads[ 2 ] = model.modules[ 2 ].modules[ 1 ].modules[ 1 ].modules[ 1 ].modules[ 1 ]:getParameters(  ) -- FC body.
	params[ 3 ], grads[ 3 ] = model.modules[ 2 ].modules[ 1 ].modules[ 1 ].modules[ 1 ].modules[ 4 ]:getParameters(  ) -- FC classifier.
	params[ 4 ], grads[ 4 ] = model.modules[ 2 ].modules[ 1 ].modules[ 1 ].modules[ 2 ].modules[ 2 ]:getParameters(  ) -- LSTM body.
	params[ 5 ], grads[ 5 ] = model.modules[ 2 ].modules[ 1 ].modules[ 1 ].modules[ 2 ].modules[ 5 ]:getParameters(  ) -- LSTM classifier.
	params[ 6 ], grads[ 6 ] = model.modules[ 2 ].modules[ 1 ].modules[ 2 ]:getParameters(  ) -- Class-dependent weight.
	params[ 7 ], grads[ 7 ] = model.modules[ 2 ].modules[ 2 ]:getParameters(  ) -- Agent.
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
	optims[ 3 ] = { -- FC classifier.
		learningRate = opt.lrClassifier,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 4 ] = { -- LSTM body.
		learningRate = opt.lrLstm,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 5 ] = { -- LSTM classifier.
		learningRate = opt.lrClassifier,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 6 ] = { -- Class-dependent weight.
		learningRate = opt.lrClassWeight,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 7 ] = { -- Agent.
		learningRate = opt.lrAgent,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	return params, grads, optims
end
