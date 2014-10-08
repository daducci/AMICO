%
% Simulate data with the NODDI toolbox (its path must be set in MATLAB path)
%
function AMICO_GenerateKernels_NODDI()

	global CONFIG AMICO_data_path

	TIME = tic();

	fprintf( '\n-> Simulating high-resolution "NODDI" kernels:\n' );

	% check if original scheme exists
	if ~exist( CONFIG.schemeFilename, 'file' )
		error( '\t[AMICO_GenerateKernels_NODDI] File "%s" not found', CONFIG.schemeFilename )
	end

	% check if high-resolution scheme has been created
	schemeHrFilename = fullfile(AMICO_data_path,CONFIG.protocol,'protocol_HR.scheme');
	if ~exist( schemeHrFilename, 'file' )
		error( '[AMICO_GenerateKernels_NODDI] File "protocol_HR.scheme" not found in folder "%s"', tmp )
	end


	% Generate the single compartments along z-axis
	noddi = MakeModel( 'WatsonSHStickTortIsoV_B0' );

    dPar = CONFIG.kernels.dPar * 1E-9;
    dIso = CONFIG.kernels.dIso * 1E-9;
	noddi.GS.fixedvals(2) = dPar; % set the parallel diffusivity from AMICO's configuration
	noddi.GD.fixedvals(2) = dPar;
	noddi.GS.fixedvals(5) = dIso;	% set the isotropic diffusivity from AMICO's configuration
	noddi.GD.fixedvals(5) = dIso;

	schemeHR   = AMICO_LoadScheme( schemeHrFilename );
	protocolHR = AMICO_Scheme2noddi( schemeHR );

	IC_KAPPAs = 1 ./ tan(CONFIG.kernels.IC_ODs*pi/2);
	idx = 1;
	for ii = 1:numel(IC_KAPPAs)
		ATOM = [];
		ATOM.kappa = IC_KAPPAs(ii);
		signal_ic = SynthMeasWatsonSHCylNeuman_PGSE( [dPar 0 ATOM.kappa], protocolHR.grad_dirs, protocolHR.G', protocolHR.delta', protocolHR.smalldel', [0;0;1], 0 );

		for jj = 1:numel(CONFIG.kernels.IC_VFs)
			v_ic = CONFIG.kernels.IC_VFs(jj);
			v_ex = 1 - v_ic;
			dPerp = dPar * (1 - v_ic);

			signal_ec = SynthMeasWatsonHinderedDiffusion_PGSE( [dPar dPerp ATOM.kappa], protocolHR.grad_dirs, protocolHR.G', protocolHR.delta', protocolHR.smalldel', [0;0;1] );

			ATOM.ICVF  = v_ic;
			ATOM.ECVF  = v_ex;
			ATOM.signal = v_ic*signal_ic + v_ex*signal_ec;

			filename = fullfile( AMICO_data_path, CONFIG.protocol, 'tmp', sprintf('A_%03d.mat',idx) );
			save( filename, 'ATOM' )
			idx = idx+1;
		end
	end

	% isotropic atom
	ATOM = [];
	ATOM.d_iso  = dIso;
	ATOM.signal = SynthMeasIsoGPD( ATOM.d_iso, protocolHR );

	filename = fullfile( AMICO_data_path, CONFIG.protocol, 'tmp', sprintf('A_%03d.mat',idx) );
	save( filename, 'ATOM' )

	fprintf( '   [ %.1f seconds ]\n', toc(TIME) );
end
