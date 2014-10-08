%
% Set the default parameters for each model
%
function AMICO_SetModel( model )
	
	global CONFIG

	CONFIG.kernels = [];
	CONFIG.kernels.model  = upper(model);

	CONFIG.OPTIMIZATION = [];
	CONFIG.OPTIMIZATION.SPAMS_param         = [];
	CONFIG.OPTIMIZATION.LS_param            = optimset('TolX',1e-4);
	CONFIG.OPTIMIZATION.SPAMS_param.mode    = 2;
	CONFIG.OPTIMIZATION.SPAMS_param.pos     = true;

	switch ( CONFIG.kernels.model )

		case 'NODDI'
			CONFIG.kernels.dPar   = 1.7;										% units of 1E-9
			CONFIG.kernels.dIso   = 3.0;										% units of 1E-9
			CONFIG.kernels.IC_VFs = linspace(0.1, 0.99,12);
			CONFIG.kernels.IC_ODs = [0.03, 0.06, linspace(0.09,0.99,10)];

			CONFIG.OPTIMIZATION.SPAMS_param.lambda  = 5e-1;
			CONFIG.OPTIMIZATION.SPAMS_param.lambda2 = 1e-3;

		case 'ACTIVEAX'
			CONFIG.kernels.dPar   = 0.6;										% units of 1E-9
			CONFIG.kernels.dIso   = 2.0;										% units of 1E-9
			CONFIG.kernels.IC_Rs  = [ 0.01 linspace(0.5,10,20)];						% units of 1E-6 (micrometers)
			CONFIG.kernels.IC_VFs = [0.3:0.1:0.9];

			CONFIG.OPTIMIZATION.SPAMS_param.lambda  = 0.25;
			CONFIG.OPTIMIZATION.SPAMS_param.lambda2 = 4;

		otherwise
			error( '\t[KERNELS_Generate] Model "%s" not recognized', CONFIG.kernels.model )
	end
