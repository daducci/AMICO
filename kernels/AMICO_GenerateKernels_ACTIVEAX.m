%
% Simulate data with the CAMINO toolkit (its path must be set in SYSTEM path)
%
function AMICO_GenerateKernels_ACTIVEAX()

	global CONFIG AMICO_data_path CAMINO_path

	TIME = tic();

	fprintf( '\n-> Simulating high-resolution "ActiveAx" kernels:\n' );

	% check if original scheme exists
	if ~exist( CONFIG.schemeFilename, 'file' )
		error( '\t[AMICO_GenerateKernels_ACTIVEAX] File "%s" not found', CONFIG.schemeFilename )
	end

	% check if high-resolution scheme has been created
	schemeHrFilename = fullfile(AMICO_data_path,CONFIG.protocol,'protocol_HR.scheme');
	if ~exist( schemeHrFilename, 'file' )
		error( '[AMICO_GenerateKernels_ACTIVEAX] File "protocol_HR.scheme" not found in folder "%s"', tmp )
	end


	% restricted
	idx = 1;
	for R = CONFIG.kernels.IC_Rs
		filename = fullfile( AMICO_data_path, CONFIG.protocol, 'tmp', sprintf('A_%03d.Bfloat',idx) );
		if exist( filename, 'file' ), delete( filename ); end
		CMD = sprintf( '%s/datasynth -synthmodel compartment 1 CYLINDERGPD %E 0 0 %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null', CAMINO_path, CONFIG.kernels.dPar*1e-9, R*1e-6, schemeHrFilename, filename );
		[status result] = system( CMD );
		if status>0
			disp(result)
			error( '[AMICO_GenerateKernels_ACTIVEAX] problems generating the signal' );
		end
		idx = idx + 1;
	end
	
	% hindered
	for ICVF = CONFIG.kernels.IC_VFs
		d_perp = CONFIG.kernels.dPar * ( 1.0 - ICVF );
		filename = fullfile( AMICO_data_path, CONFIG.protocol, 'tmp', sprintf('A_%03d.Bfloat',idx) );
		if exist( filename, 'file' ), delete( filename ); end
		CMD = sprintf( '%s/datasynth -synthmodel compartment 1 ZEPPELIN %E 0 0 %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null', CAMINO_path, CONFIG.kernels.dPar*1e-9, d_perp*1e-9, schemeHrFilename, filename );
		[status result] = system( CMD );
		if status>0
			disp(result)
			error( '[AMICO_GenerateKernels_ACTIVEAX] problems generating the signal' );
		end
		idx = idx + 1;
	end

	% isotropic
	filename = fullfile( AMICO_data_path, CONFIG.protocol, 'tmp', sprintf('A_%03d.Bfloat',idx) );
	if exist( filename, 'file' ), delete( filename ); end
	CMD = sprintf( '%s/datasynth -synthmodel compartment 1 BALL %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null', CAMINO_path, CONFIG.kernels.dIso*1e-9, schemeHrFilename, filename );
	[status result] = system( CMD );
	if status>0
		disp(result)
		error( '\t\t- ERROR: problems generating the signal\n' );
	end
	idx = idx + 1;

	fprintf( '   [ %.1f seconds ]\n', toc(TIME) );
end
