function AMICO_Fit_ACTIVEAX()

	global CONFIG
	global niiSIGNAL niiMASK
	global KERNELS bMATRIX

	% dataset for ESTIMATED PARAMETERS
	niiMAP = niiMASK;
	niiMAP.hdr.dime.dim(1) = 4;
	niiMAP.hdr.dime.dim(5) = 3;
	niiMAP.hdr.dime.datatype = 16;
	niiMAP.hdr.dime.bitpix = 32;
	niiMAP.hdr.dime.glmin = 0;
	niiMAP.hdr.dime.glmax = 1;
	niiMAP.hdr.dime.calmin = 0;
	niiMAP.hdr.dime.calmax = 1;
	niiMAP.img = zeros( niiMAP.hdr.dime.dim(2:5), 'single' );

	% dataset for ESTIMATED DIRECTIONS
	niiDIR = niiMASK;
	niiDIR.hdr.dime.dim(1) = 4;
	niiDIR.hdr.dime.dim(5) = 3;
	niiDIR.hdr.dime.datatype = 16;
	niiDIR.hdr.dime.bitpix = 32;
	niiDIR.hdr.dime.glmin = -1;
	niiDIR.hdr.dime.glmax =  1;
	niiDIR.hdr.dime.calmin = -1;
	niiDIR.hdr.dime.calmax =  1;
	niiDIR.img = zeros( niiMAP.hdr.dime.dim(2:5), 'single' );


	fprintf( '\n-> Fitting %s model to data:\n', CONFIG.kernels.model );
	nIC = numel(CONFIG.kernels.IC_Rs);
	nEC = numel(CONFIG.kernels.IC_VFs);

	TIME = tic;
	for iz = 1:niiSIGNAL.hdr.dime.dim(4)
	for iy = 1:niiSIGNAL.hdr.dime.dim(3)
	for ix = 1:niiSIGNAL.hdr.dime.dim(2)
		if niiMASK.img(ix,iy,iz)==0, continue, end
		
		% read the signal
		b0 = mean( squeeze( niiSIGNAL.img(ix,iy,iz,CONFIG.scheme.b0_idx) ) );
		if ( b0 < 1e-3 ), continue, end
		y = double( squeeze( niiSIGNAL.img(ix,iy,iz,:) ) ./ ( b0 + eps ) );

		% find the MAIN DIFFUSION DIRECTION using DTI
		[ ~, ~, V ] = AMICO_FitTensor( y, bMATRIX );
		Vt = V(:,1);
		if ( Vt(2)<0 ), Vt = -Vt; end

		% build the DICTIONARY
		[ i1, i2 ] = AMICO_Dir2idx( Vt );
		A = double( [ KERNELS.Aic(CONFIG.scheme.dwi_idx,:,i1,i2) KERNELS.Aec(CONFIG.scheme.dwi_idx,:,i1,i2) KERNELS.Aiso(CONFIG.scheme.dwi_idx) ] );
	
		% fit AMICO
		y = y(CONFIG.scheme.dwi_idx);
		yy = [ 1 ; y ];
		AA = [ ones(1,size(A,2)) ; A ];

		% estimate IC and EC compartments and promote sparsity
		x = full( mexLasso( yy, AA, CONFIG.OPTIMIZATION.SPAMS_param ) );

		% STORE results	
		niiDIR.img(ix,iy,iz,:) = Vt;

		f1 = sum( x( 1:nIC ) );
		f2 = sum( x( (nIC+1):(nIC+nEC) ) );
		v = f1 / ( f1 + f2 + eps );
		a = 2 * KERNELS.Aic_R * x(1:nIC) / ( f1 + eps );
		
		niiMAP.img(ix,iy,iz,1) = v;
		niiMAP.img(ix,iy,iz,2) = a;
		niiMAP.img(ix,iy,iz,3) = (4*v) / ( pi*a^2 + eps );
	end
	end
	end
	TIME = toc(TIME);
	fprintf( '   [ %.0fh %.0fm %.0fs ]\n', floor(TIME/3600), floor(mod(TIME/60,60)), mod(TIME,60) )

	
	% save output maps
	fprintf( '\n-> Saving output maps:\n' );
	
	save_untouch_nii( niiMAP, fullfile(CONFIG.OUTPUT_path,'FIT_parameters.nii') );
	save_untouch_nii( niiDIR, fullfile(CONFIG.OUTPUT_path,'FIT_dir.nii') );
	
	fprintf( '   [ AMICO/FIT_*.nii ]\n' )
