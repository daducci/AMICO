function [ KERNELS ] = AMICO_ResampleKernels_ACTIVEAX( idx_OUT, Ylm_OUT )

	global CONFIG AMICO_data_path

	ATOMS_path = fullfile(AMICO_data_path,CONFIG.protocol,'common');

	% Setup structure
	% ===============
	nIC = numel(CONFIG.kernels.IC_Rs);
	nEC = numel(CONFIG.kernels.IC_VFs);
	
	KERNELS = {};
	KERNELS.model    = 'ACTIVEAX';
	KERNELS.nS       = CONFIG.scheme.nS;
	KERNELS.nA       = nIC + nEC + 1; % number of atoms
	
	KERNELS.dPar     = CONFIG.kernels.dPar;

	KERNELS.Aic      = zeros( [KERNELS.nS nIC 181 181], 'single' );
	KERNELS.Aic_R    = zeros( 1, nIC, 'single' );
	
	KERNELS.Aec      = zeros( [KERNELS.nS nIC 181 181], 'single' );
	KERNELS.Aec_icvf = zeros( 1, nEC, 'single' );
	
	KERNELS.Aiso     = zeros( [KERNELS.nS 1], 'single' );
	KERNELS.Aiso_d   = NaN;


	% Restricted
	% ==========
	idx = 1;
	for i = 1:nIC
		TIME2 = tic();
		fprintf( '\t- A_%03d...  ', idx );
		
		load( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), 'lm' );

		KERNELS.Aic(:,i,:,:) = AMICO_ResampleKernel( lm, idx_OUT, Ylm_OUT, false );
		KERNELS.Aic_R(i)     = CONFIG.kernels.IC_Rs(i);
		idx = idx + 1;

		fprintf( '[%.1f seconds]\n', toc(TIME2) );
	end


	% Hindered
	% ========
	for i = 1:nEC
		TIME2 = tic();
		fprintf( '\t- A_%03d...  ', idx );
		
		load( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), 'lm' );

		KERNELS.Aec(:,i,:,:) = AMICO_ResampleKernel( lm, idx_OUT, Ylm_OUT, false );
		KERNELS.Aec_icvf(i)  = CONFIG.kernels.IC_VFs(i);
		idx = idx + 1;

		fprintf( '[%.1f seconds]\n', toc(TIME2) );
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

