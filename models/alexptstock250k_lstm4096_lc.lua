local modelPath = '/home/doyoo/workspace/net/torch/alexptstock250k.t7'
local lstmNumHidden = 4096

function createModel( nGPU )
	-- Load pre-trained feature model.
	print( 'Load pre-trained model: ' .. modelPath )
	local features = loadDataParallel( modelPath, nGPU )
	features:remove(  ) -- removes classifier.
	features:cuda(  )
	-- Create LSTM
	local lstm = nn.Sequential(  )
	lstm:add( nn.View( -1, opt.seqLength, 4096 ) ) -- LSTM input is NumVideo X SeqLength X vectorDim
	lstm:add( nn.LSTM( 4096, lstmNumHidden ) )
	lstm:add( nn.Dropout( 0.5 ) )
	lstm:add( nn.View( -1, lstmNumHidden ) )
	lstm:cuda(  )
	-- Create classifier. 
	-- (does not add a normalization layers because we use MultiLabelMarginCriterion.)
	local classifier = nn.Sequential(  )
	classifier:add( nn.Linear( lstmNumHidden, opt.nClasses ) )
	classifier:cuda(  )
	-- Combine modules.
   local model = nn.Sequential(  ):add( features ):add( lstm ):add( classifier )
	-- Check options.
	assert( opt.inputIsCaffe )
	assert( not opt.normalizeStd )
	assert( not opt.keepAspect )
   return model
end
