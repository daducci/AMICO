%
% Simulate data with the CAMINO toolkit
%
function AMICO_GenerateKernels_ACTIVEAX( AUX, idx_IN, idx_OUT )

	global CONFIG AMICO_data_path CAMINO_path

	TIME = tic();

	fprintf( '\t* Simulating "ActiveAx" kernels:\n' );
	ATOMS_path = fullfile(AMICO_data_path,CONFIG.protocol,'common');
	filenameHr = [tempname '.Bfloat'];	

	% check if high-resolution scheme has been created
	schemeHrFilename = fullfile(ATOMS_path,'protocol_HR.scheme');
	if ~exist( schemeHrFilename, 'file' )
		error( '[AMICO_GenerateKernels_ACTIVEAX] File "protocol_HR.scheme" not found in folder "%s"', ATOMS_path )
	end


	% Restricted
	% ==========
	idx = 1;
	for R = CONFIG.kernels.IC_Rs
		TIME2 = tic();
		fprintf( '\t\t- A_%03d... ', idx );

		% generate
		if exist( filenameHr, 'file' ), delete( filenameHr ); end
		CMD = sprintf( '%s/datasynth -synthmodel compartment 1 CYLINDERGPD %E 0 0 %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null', CAMINO_path, CONFIG.kernels.dPar*1e-9, R*1e-6, schemeHrFilename, filenameHr );
		[status result] = system( CMD );
		if status>0
			disp(result)
			error( '[AMICO_GenerateKernels_ACTIVEAX] Problems generating the signal with datasynth' );
		end

		% rotate and save
		fid = fopen( filenameHr, 'r', 'b' );
		signal = fread(fid,'float');
		fclose(fid);
		delete( filenameHr );
		lm = AMICO_RotateKernel( signal, AUX, idx_IN, idx_OUT, false );
		save( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), '-v6', 'lm' )

		idx = idx + 1;
		fprintf( '[%.1f seconds]\n', toc(TIME2) );
	end


	% Hindered
	% ========
	for ICVF = CONFIG.kernels.IC_VFs
		TIME2 = tic();
		fprintf( '\t\t- A_%03d... ', idx );

		% generate
		d_perp = CONFIG.kernels.dPar * ( 1.0 - ICVF );
		if exist( filenameHr, 'file' ), delete( filenameHr ); end
		CMD = sprintf( '%s/datasynth -synthmodel compartment 1 ZEPPELIN %E 0 0 %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null', CAMINO_path, CONFIG.kernels.dPar*1e-9, d_perp*1e-9, schemeHrFilename, filenameHr );
		[status result] = system( CMD );
		if status>0
			disp(result)
			error( '[AMICO_GenerateKernels_ACTIVEAX] problems generating the signal' );
		end
		
		% rotate and save
		fid = fopen( filenameHr, 'r', 'b' );
		signal = fread(fid,'float');
		fclose(fid);
		delete( filenameHr );
		lm = AMICO_RotateKernel( signal, AUX, idx_IN, idx_OUT, false );
		save( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), '-v6', 'lm' )

		idx = idx + 1;
		fprintf( '[%.1f seconds]\n', toc(TIME2) );
	end


	% Isotropic
	% =========
	TIME2 = tic();
	fprintf( '\t\t- A_%03d... ', idx );

	% generate
	if exist( filenameHr, 'file' ), delete( filenameHr ); end
	CMD = sprintf( '%s/datasynth -synthmodel compartment 1 BALL %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null', CAMINO_path, CONFIG.kernels.dIso*1e-9, schemeHrFilename, filenameHr );
	[status result] = system( CMD );
	if status>0
		disp(result)
		error( '[AMICO_GenerateKernels_ACTIVEAX] problems generating the signal' );
	end
	
	% resample and save
	lm = AMICO_RotateKernel( signal, AUX, idx_IN, idx_OUT, true );
	save( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), '-v6', 'lm' )
	
	idx = idx + 1;
	fprintf( '[%.1f seconds]\n', toc(TIME2) );


	fprintf( '\t  [ %.1f seconds ]\n', toc(TIME) );
end
