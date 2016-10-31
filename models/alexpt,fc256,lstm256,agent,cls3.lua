require 'loadcaffe'
function createModel( nGPU )
	-- Set params.
	local featSize = 4096
	local hiddenSize = 256
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
	-- Out: ( numVideo X seqLength ), hiddenSize
	local fc = nn.Sequential(  )
	fc:add( nn.Linear( featSize, hiddenSize ) )
	fc:add( nn.ReLU(  ) )
	fc:add( nn.Dropout( 0.5 ) )
	fc:cuda(  )

	-- Create LSTM tower.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), hiddenSize
	local lstm = nn.Sequential(  )
	lstm:add( nn.View( -1, seqLength, featSize ) ) -- LSTM input is numVideo, SeqLength, vectorDim
	lstm:add( nn.LSTM( featSize, hiddenSize ) )
	lstm:add( nn.Tanh(  ) )
	lstm:add( nn.Dropout( 0.5 ) )
	lstm:add( nn.View( -1, hiddenSize ) )
	lstm:cuda(  )

	-- Create agent tower.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), ( hiddenSize X 2 )
	local agent = nn.Sequential(  )
	agent:add( nn.View( -1, seqLength * featSize ) ) -- numVideo, ( seqLength X featSize )
	agent:add( nn.Linear( seqLength * featSize, 2 ) ) -- numVideo, 2
	agent:add( nn.Sigmoid(  ) )
	agent:add( nn.Reshape( 2 * numVideo ) )
	agent:add( nn.Replicate( hiddenSize, 2 ) )
	agent:add( nn.Reshape( numVideo, hiddenSize * 2 ) )
	agent:add( nn.Replicate( seqLength, 1, 1 ) )
	agent:add( nn.Reshape( seqLength * numVideo, hiddenSize * 2 ) )
	agent:cuda(  )
	
	-- Create FC classifier.
	-- In:  ( numVideo X seqLength ), hiddenSize
	-- Out: ( numVideo X seqLength ), numClass
	local classifierFc = nn.Sequential(  )
	classifierFc:add( nn.Linear( hiddenSize, numCls ) )
	classifierFc:add( nn.LogSoftMax(  ) )
	classifierFc:cuda(  )

	-- Create LSTM classifier.
	-- In:  ( numVideo X seqLength ), hiddenSize
	-- Out: ( numVideo X seqLength ), numClass
	local classifierLstm = nn.Sequential(  )
	classifierLstm:add( nn.Linear( hiddenSize, numCls ) )
	classifierLstm:add( nn.LogSoftMax(  ) )
	classifierLstm:cuda(  )

	-- Create agent classifier.
	-- In:  ( numVideo X seqLength ), ( hiddenSize X 2 )
	-- Out: ( numVideo X seqLength ), numClass
	local classifierAgent = nn.Sequential(  )
	classifierAgent:add( nn.Linear( hiddenSize * 2, numCls ) ) -- numVideo, 2
	classifierAgent:add( nn.LogSoftMax(  ) )
	classifierAgent:cuda(  )

	-- Make module after feature.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), numClass
	local x = nn.Identity(  )(  )
	local h_fc = fc( x )
	local h_fccls = classifierFc( h_fc )
	local h_lstm = lstm( x )
	local h_lstmcls = classifierLstm( h_lstm )
	local h_agent = agent( x )
	local h_fclstm = nn.JoinTable( 2 )( { h_fc, h_lstm } )
	local h_fclstmagent = nn.CMulTable(  )( { h_fclstm, h_agent } )
	local h_fclstmagentcls = classifierAgent( h_fclstmagent )
	local m = nn.gModule( { x }, { h_fccls, h_lstmcls, h_fclstmagentcls } )
	m:cuda(  )
	
	-- Combine sub models.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), numClass
   local model = nn.Sequential(  )
	model:add( features )
	model:add( m )
	model:cuda(  )

	-- Check options.
	assert( opt.numLoss == 3 )
	assert( opt.inputIsCaffe )
	assert( not opt.normalizeStd )
	assert( not opt.keepAspect )
	collectgarbage(  )
	return model
end

function groupParams( model )
	assert( opt.lrFc == opt.lrLstm )
	assert( opt.lrFc == opt.lrAgent )
	assert( opt.lrFc == opt.lrClassifier )
	local params, grads, optims = {  }, {  }, {  }
	params[ 1 ], grads[ 1 ] = model.modules[ 1 ]:getParameters(  ) -- Features.
	params[ 2 ], grads[ 2 ] = model.modules[ 2 ]:getParameters(  ) -- Remaining.
	optims[ 1 ] = { -- Features.
		learningRate = opt.lrFeature,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	optims[ 2 ] = { -- Remaining.
		learningRate = opt.lrFc,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		dampening = 0.0,
		weightDecay = opt.weightDecay 
	}
	return params, grads, optims
end