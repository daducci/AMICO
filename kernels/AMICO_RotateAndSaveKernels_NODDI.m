function [ KERNELS ] = AMICO_RotateAndSaveKernels_NODDI( AUX, idx_IN, idx_OUT, Ylm_OUT )

	global CONFIG AMICO_data_path

	OUTPUT_path = fullfile(AMICO_data_path,CONFIG.protocol,'tmp');

	KERNELS = {};
	KERNELS.model   = 'NODDI';
	KERNELS.nS      = CONFIG.scheme.nS;
	KERNELS.nA      = numel(CONFIG.kernels.IC_VFs) * numel(CONFIG.kernels.IC_ODs) + 1; % number of atoms
	
	KERNELS.dPar    = CONFIG.kernels.dPar;

	KERNELS.A       = zeros( [KERNELS.nS KERNELS.nA-1 181 181], 'single' );
	KERNELS.A_kappa = zeros( 1, KERNELS.nA-1, 'single' );
	KERNELS.A_icvf  = zeros( 1, KERNELS.nA-1, 'single' );;
	KERNELS.A_ecvf  = zeros( 1, KERNELS.nA-1, 'single' );;
	
	KERNELS.Aiso    = zeros( [KERNELS.nS 1], 'single' );
	KERNELS.Aiso_d  = NaN;

	% coupled atoms
	for i = 1:KERNELS.nA-1
		TIME2 = tic();
		fprintf( '\t- A_%03d.mat... ', i );
		load( fullfile( OUTPUT_path, sprintf('A_%03d.mat',i) ), 'ATOM' );

		KERNELS.A(:,i,:,:) = AMICO_RotateKernel( ATOM.signal, CONFIG.scheme, AUX, idx_IN, idx_OUT, Ylm_OUT );
		KERNELS.A_kappa(i) = ATOM.kappa;
		KERNELS.A_icvf(i)  = ATOM.ICVF;
		KERNELS.A_ecvf(i)  = ATOM.ECVF;

		fprintf( '[%.1f seconds]\n', toc(TIME2) );
	end

	% isotropic
	TIME2 = tic();
	fprintf( '\t- A_%03d.mat... ', KERNELS.nA );
	load( fullfile( OUTPUT_path, sprintf('A_%03d.mat',KERNELS.nA) ), 'ATOM' );

	KERNELS.Aiso   = AMICO_ResampleIsoKernel( ATOM.signal, CONFIG.scheme, AUX, idx_IN, idx_OUT, Ylm_OUT );
	KERNELS.Aiso_d = ATOM.d_iso;

	fprintf( '[%.1f seconds]\n', toc(TIME2) );

