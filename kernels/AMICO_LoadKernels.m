% Load precomputed kernels
%
% Returns
% -------
% KERNELS : struct
% 	Atoms for each compartment rotated along 181x181 directions
function [ KERNELS ] = AMICO_LoadKernels()

	global CONFIG

	dictFilename = fullfile( CONFIG.OUTPUT_path, sprintf('kernels_%s.mat',CONFIG.kernels.model) );

	fprintf( '\n-> Loading precomputed kernels:\n' );

	% check if kernels file exists
	if ~exist( dictFilename, 'file' )
		error( '[AMICO_LoadKernels] File "%s" not found', dictFilename )
	end

	load( dictFilename );

	% check if the model used for these kernels is the same used here
	if strcmp( KERNELS.model, CONFIG.kernels.model ) ~= 1
		error( '[AMICO_LoadKernels] The model does not match' )
	end

	fprintf( '   [ %s, %d atoms, %d samples ]\n', KERNELS.model, KERNELS.nA, KERNELS.nS );
end
