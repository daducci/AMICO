function [ KERNELS ] = AMICO_ResampleKernels_NODDI( idx_OUT, Ylm_OUT )

	global CONFIG AMICO_data_path

	ATOMS_path = fullfile(AMICO_data_path,CONFIG.protocol,'common');

	% Setup structure
	% ===============
	KERNELS = {};
	KERNELS.model   = 'NODDI';
	KERNELS.nS      = CONFIG.scheme.nS;
	KERNELS.nA      = numel(CONFIG.kernels.IC_VFs) * numel(CONFIG.kernels.IC_ODs) + 1; % number of atoms
	
	KERNELS.dPar    = CONFIG.kernels.dPar;

	KERNELS.A       = zeros( [KERNELS.nS KERNELS.nA-1 181 181], 'single' );
	KERNELS.A_kappa = zeros( 1, KERNELS.nA-1, 'single' );
	KERNELS.A_icvf  = zeros( 1, KERNELS.nA-1, 'single' );;
	
	KERNELS.Aiso    = zeros( [KERNELS.nS 1], 'single' );
	KERNELS.Aiso_d  = NaN;
	
	
	% Coupled atoms
	% =============
	idx = 1;
	for ii = 1:numel(CONFIG.kernels.IC_ODs)
	for jj = 1:numel(CONFIG.kernels.IC_VFs)
		TIME2 = tic();
		fprintf( '\t- A_%03d...  ', idx );
		
		load( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), 'lm' );

		KERNELS.A(:,idx,:,:) = AMICO_ResampleKernel( lm, idx_OUT, Ylm_OUT, false );
		KERNELS.A_kappa(idx) = 1 ./ tan(CONFIG.kernels.IC_ODs(ii)*pi/2);
		KERNELS.A_icvf(idx)  = CONFIG.kernels.IC_VFs(jj);
		idx = idx + 1;

		fprintf( '[%.1f seconds]\n', toc(TIME2) );
	end
	end

	
	% Isotropic
	% =========
	TIME2 = tic();
	fprintf( '\t- A_%03d...  ', idx );
	
	load( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), 'lm' );

	KERNELS.Aiso   = AMICO_ResampleKernel( lm, idx_OUT, Ylm_OUT, true );
	KERNELS.Aiso_d = CONFIG.kernels.dIso;
	idx = idx + 1;

	fprintf( '[%.1f seconds]\n', toc(TIME2) );

