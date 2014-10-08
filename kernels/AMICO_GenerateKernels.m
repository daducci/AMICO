%
% Generete the high-resolution atoms in the dictionary.
% Dispatch to the proper function, depending on the model.
%
function AMICO_GenerateKernels()

	global CONFIG AMICO_data_path

	% create folder for temporary files
	output_path = fullfile(AMICO_data_path,CONFIG.protocol,'tmp');
	[~,~,~] = mkdir( output_path );

	switch ( CONFIG.kernels.model )
		case 'NODDI'
			AMICO_GenerateKernels_NODDI();
		otherwise
			error( '\t[AMICO_GenerateKernels] Model "%s" not recognized', CONFIG.kernels.model )
	end

end
