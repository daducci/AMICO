function AMICO_fit()

	global CONFIG

	% dispatch to the right handler for each model
	switch ( CONFIG.kernels.model )

		case 'NODDI'
			AMICO_fit_NODDI();

		case 'ACTIVEAX'
			AMICO_fit_ACTIVEAX();

		otherwise
			error( '\t[AMICO_fit] Model "%s" not recognized', CONFIG.kernels.model )

	end
	
	% save the configuration
	save( fullfile(CONFIG.DATA_path,'CONFIG.mat'), '-v6', 'CONFIG' )
