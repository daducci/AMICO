%
% Simulate data with the NODDI Matlab Toolbox
%
function AMICO_GenerateKernels_NODDI( AUX, idx_IN, idx_OUT )

	global CONFIG AMICO_data_path

	TIME = tic();

	fprintf( '\t* Simulating "NODDI" kernels:\n' );
	ATOMS_path = fullfile(AMICO_data_path,CONFIG.protocol,'common');

	% check if high-resolution scheme has been created
	schemeHrFilename = fullfile(ATOMS_path,'protocol_HR.scheme');
	if ~exist( schemeHrFilename, 'file' )
		error( '[AMICO_GenerateKernels_ACTIVEAX] File "protocol_HR.scheme" not found in folder "%s"', ATOMS_path )
	end


	% Configure NODDI toolbox
	% =======================
	noddi = MakeModel( 'WatsonSHStickTortIsoV_B0' );

    dPar = CONFIG.kernels.dPar * 1E-9;
    dIso = CONFIG.kernels.dIso * 1E-9;
	noddi.GS.fixedvals(2) = dPar; % set the parallel diffusivity from AMICO's configuration
	noddi.GD.fixedvals(2) = dPar;
	noddi.GS.fixedvals(5) = dIso;	% set the isotropic diffusivity from AMICO's configuration
	noddi.GD.fixedvals(5) = dIso;

	schemeHR   = AMICO_LoadScheme( schemeHrFilename );
	protocolHR = AMICO_Scheme2noddi( schemeHR );

	
	% Coupled compartments
	% ====================
	IC_KAPPAs = 1 ./ tan(CONFIG.kernels.IC_ODs*pi/2);
	idx = 1;
	for ii = 1:numel(IC_KAPPAs)
		kappa = IC_KAPPAs(ii);
		signal_ic = SynthMeasWatsonSHCylNeuman_PGSE( [dPar 0 kappa], protocolHR.grad_dirs, protocolHR.G', protocolHR.delta', protocolHR.smalldel', [0;0;1], 0 );

		for jj = 1:numel(CONFIG.kernels.IC_VFs)
			TIME2 = tic();
			fprintf( '\t\t- A_%03d... ', idx );
			
			% generate
			v_ic = CONFIG.kernels.IC_VFs(jj);
			dPerp = dPar * (1 - v_ic);
			signal_ec = SynthMeasWatsonHinderedDiffusion_PGSE( [dPar dPerp kappa], protocolHR.grad_dirs, protocolHR.G', protocolHR.delta', protocolHR.smalldel', [0;0;1] );
			signal = v_ic*signal_ic + (1-v_ic)*signal_ec;

			% rotate and save
			lm = AMICO_RotateKernel( signal, AUX, idx_IN, idx_OUT, false );
			save( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), '-v6', 'lm' )

			idx = idx+1;
			fprintf( '[%.1f seconds]\n', toc(TIME2) );
		end
	end


	% Isotropic
	% =========
	TIME2 = tic();
	fprintf( '\t\t- A_%03d... ', idx );

	% generate
	signal = SynthMeasIsoGPD( dIso, protocolHR );

	% resample and save
	lm = AMICO_RotateKernel( signal, AUX, idx_IN, idx_OUT, true );
	save( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), '-v6', 'lm' )
	
	idx = idx + 1;
	fprintf( '[%.1f seconds]\n', toc(TIME2) );


	fprintf( '\t  [ %.1f seconds ]\n', toc(TIME) );
end
