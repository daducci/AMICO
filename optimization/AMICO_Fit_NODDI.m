function AMICO_fit_NODDI()

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

	% precompute norms of coupled atoms (for the l1 minimization)
	A = double( KERNELS.A(CONFIG.scheme.dwi_idx,:,1,1) );
	A_norm = repmat( 1./sqrt( sum(A.^2) ), [size(A,1),1] );


	fprintf( '\n-> Fitting %s model to data:\n', CONFIG.kernels.model );
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
		A = double( [ KERNELS.A(CONFIG.scheme.dwi_idx,:,i1,i2) KERNELS.Aiso(CONFIG.scheme.dwi_idx) ] );
	
		% fit AMICO
		y = y(CONFIG.scheme.dwi_idx);
		yy = [ 1 ; y ];
		AA = [ ones(1,size(A,2)) ; A ];

		% estimate CSF partial volume and remove it
		x = lsqnonneg( AA, yy, CONFIG.OPTIMIZATION.LS_param );
		y = y - x(end)*A(:,end);

		% estimate IC and EC compartments and promote sparsity
		An = A(:,1:end-1) .* A_norm;
		x = full( mexLasso( y, An, CONFIG.OPTIMIZATION.SPAMS_param ) );

		% debias
		idx = x>0;
		idx(end+1) = true;
		x(idx) = lsqnonneg( AA(:,idx), yy, CONFIG.OPTIMIZATION.LS_param );

		% STORE results	
		niiDIR.img(ix,iy,iz,:) = Vt;

		xx =  x(1:end-1);
		xx = xx ./ ( sum(xx) + eps );
		f1 = KERNELS.A_icvf * xx;
		f2 = (1-KERNELS.A_icvf) * xx;
		niiMAP.img(ix,iy,iz,1) = f1 / (f1+f2+eps);

		kappa = KERNELS.A_kappa * xx;
		niiMAP.img(ix,iy,iz,2) = 2/pi * atan2(1,kappa);

		niiMAP.img(ix,iy,iz,3) = x(end);
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
