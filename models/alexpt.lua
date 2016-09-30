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
	-- Create classifier.
	local classifier = nn.Sequential(  )
	classifier:add( nn.Linear( 4096, opt.nClasses ) )
	classifier:add( nn.LogSoftMax(  ) )
	classifier:cuda(  )
	-- Combine sub models.
   local model = nn.Sequential(  ):add( features ):add( classifier )
	-- Check options.
	assert( opt.inputIsCaffe )
	assert( not opt.normalizeStd )
	assert( not opt.keepAspect )
   return model
end
