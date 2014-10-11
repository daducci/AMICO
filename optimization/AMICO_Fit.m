function AMICO_Fit()

	global CONFIG

	% dispatch to the right handler for each model
	f = str2func( ['AMICO_Fit_' CONFIG.kernels.model] );
	if ( exist([func2str(f) '.m'],'file') )
		f();
	else
		error( '[AMICO_Fit] Model "%s" not recognized', CONFIG.kernels.model )
	end
	
	% save the configuration
	save( fullfile(CONFIG.OUTPUT_path,'CONFIG.mat'), '-v6', 'CONFIG' )
