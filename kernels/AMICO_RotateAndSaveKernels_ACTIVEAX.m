function [ KERNELS ] = AMICO_RotateAndSaveKernels_ACTIVEAX( AUX, idx_IN, idx_OUT, Ylm_OUT )

	global CONFIG AMICO_data_path

	OUTPUT_path = fullfile(AMICO_data_path,CONFIG.protocol,'tmp');

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

	% restricted
	idx = 1;
	for i = 1:nIC
		TIME2 = tic();
		fprintf( '\t- A_%03d.Bfloat... ', idx );
		
		fid = fopen( fullfile( OUTPUT_path, sprintf('A_%03d.Bfloat',idx) ), 'r', 'b' );
		signal = fread(fid,'float');
		fclose(fid);

		KERNELS.Aic(:,i,:,:) = AMICO_RotateKernel( signal, CONFIG.scheme, AUX, idx_IN, idx_OUT, Ylm_OUT );
		KERNELS.Aic_R(i)     = CONFIG.kernels.IC_Rs(i);
		idx = idx + 1;

		fprintf( '[%.1f seconds]\n', toc(TIME2) );
	end

	% hindered
	for i = 1:nEC
		TIME2 = tic();
		fprintf( '\t- A_%03d.Bfloat... ', idx );
		
		fid = fopen( fullfile( OUTPUT_path, sprintf('A_%03d.Bfloat',idx) ), 'r', 'b' );
		signal = fread(fid,'float');
		fclose(fid);

		KERNELS.Aec(:,i,:,:) = AMICO_RotateKernel( signal, CONFIG.scheme, AUX, idx_IN, idx_OUT, Ylm_OUT );
		KERNELS.Aec_icvf(i)  = CONFIG.kernels.IC_VFs(i);
		idx = idx + 1;

		fprintf( '[%.1f seconds]\n', toc(TIME2) );
	end

	% isotropic
	TIME2 = tic();
	fprintf( '\t- A_%03d.Bfloat... ', idx );
	
	fid = fopen( fullfile( OUTPUT_path, sprintf('A_%03d.Bfloat',idx) ), 'r', 'b' );
	signal = fread(fid,'float');
	fclose(fid);

	KERNELS.Aiso   = AMICO_ResampleIsoKernel( signal, CONFIG.scheme, AUX, idx_IN, idx_OUT, Ylm_OUT );
	KERNELS.Aiso_d = CONFIG.kernels.dIso;
	idx = idx + 1;

	fprintf( '[%.1f seconds]\n', toc(TIME2) );

