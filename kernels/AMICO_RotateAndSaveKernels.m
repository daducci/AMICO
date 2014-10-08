%
% Precompute rotated versions (180x180 directions) of each simulated atom and
% store them matching the acquisition protocol used for the subject (specific scheme file)
%
% Parameters
% ----------
% lmax : unsigned int
%   Maximum spherical harmonics order to use for the rotation phase
%
function AMICO_RotateAndSaveKernels( lmax )
	if nargin < 1, lmax = 12; end
	global CONFIG AMICO_data_path KERNELS

	TIME = tic();
	fprintf( '\n-> Rotating kernels to 180x180 directions for subject "%s":\n', CONFIG.subject );

	% check if original scheme exists
	if ~exist( CONFIG.schemeFilename, 'file' )
		error( '[AMICO_RotateAndSaveKernels] File "%s" not found', CONFIG.schemeFilename )
	else
		scheme = AMICO_LoadScheme( CONFIG.schemeFilename );
	end

	% check if auxiliary matrices have been precomputed
	auxFilename = fullfile(AMICO_data_path,sprintf('AUX_matrices__lmax=%d.mat',lmax) );
	if ~exist( auxFilename, 'file' )
		error( '[AMICO_RotateAndSaveKernels] Auxiliary matrices "%s" not found', auxFilename )
	else
		AUX = load( auxFilename );
	end


	% precompute aux data structures
	% ------------------------------
	idx_IN  = [];
	idx_OUT = [];
	Ylm_OUT = [];
	row = 1;
	for i = 1:numel(scheme.shells)
		idx_IN{end+1}  = row : row+500-1;
		idx_OUT{end+1} = scheme.shells{i}.idx;
		[colatitude, longitude] = AMICO_Cart2sphere( scheme.shells{i}.grad(:,1), scheme.shells{i}.grad(:,2), scheme.shells{i}.grad(:,3) );
		Ylm_OUT{end+1} = AMICO_CreateYlm( AUX.lmax, colatitude, longitude ); % matrix from SH to real space
		row = row+500;
	end
	
	
	% rotate kernel in all direction and sample according to subject's acquisition scheme
	% -----------------------------------------------------------------------------------
	switch ( CONFIG.kernels.model )

		case 'NODDI'
			KERNELS = AMICO_RotateAndSaveKernels_NODDI( AUX, idx_IN, idx_OUT, Ylm_OUT );

		otherwise
			error( '\t[AMICO_RotateAndSaveKernels] Model "%s" not recognized', CONFIG.kernels.model )
	end


	% save to file
	% ------------
	fprintf( '\t- saving...    ' );
	TIME2 = tic();
	save( fullfile( CONFIG.DATA_path, sprintf('kernels_%s.mat',CONFIG.kernels.model) ), '-v6', 'KERNELS' )
	fprintf( '[%.1f seconds]\n', toc(TIME2) );


	fprintf( '   [ %.1f seconds ]\n', toc(TIME) );
end
