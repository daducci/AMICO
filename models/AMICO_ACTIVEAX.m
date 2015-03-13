classdef AMICO_ACTIVEAX

properties
	id, name                % id and name of the model
    dPar                    % parallel diffusivity of the tensors [units of mm^2/s]
    dIso                    % isotropic diffusivity [units of mm^2/s]
    IC_Rs                   % radii of the axons [units of 1E-6 (micrometers)]
    IC_VFs                  % volume fractions of the axons
    OUTPUT_names            % suffix of the output maps
    OUTPUT_descriptions     % description of the output maps
end


methods

    % =================================
    % Setup the parameters of the model
    % =================================
	function obj = AMICO_ACTIVEAX()
        global CONFIG

        % set the parameters of the model
        obj.id        = 'ACTIVEAX';
        obj.name      = 'ActiveAx';
        obj.dPar      = 0.6 * 1E-3;
        obj.dIso      = 2.0 * 1E-3;
		obj.IC_Rs     = [0.01 linspace(0.5,10,20)];
		obj.IC_VFs    = [0.3:0.1:0.9];

        obj.OUTPUT_names        = { 'v', 'a', 'd' };
        obj.OUTPUT_descriptions = {'Intra-cellular volume fraction', 'Mean axonal diameter', 'Axonal density'};

        % set the parameters to fit it
        CONFIG.OPTIMIZATION.SPAMS_param.mode    = 2;
        CONFIG.OPTIMIZATION.SPAMS_param.pos     = true;
        CONFIG.OPTIMIZATION.SPAMS_param.lambda  = 0.25; % l1 regularization
        CONFIG.OPTIMIZATION.SPAMS_param.lambda2 = 4;    % l2 regularization
    end


    % ==================================================================
    % Generate high-resolution kernels and rotate them in harmonic space
    % ==================================================================
    function GenerateKernels( obj, ATOMS_path, schemeHR, AUX, idx_IN, idx_OUT )
        global CONFIG AMICO_data_path CAMINO_path

        % check if high-resolution scheme has been created
        schemeHrFilename = fullfile(ATOMS_path,'protocol_HR.scheme');
        if ~exist( schemeHrFilename, 'file' )
            error( '[AMICO_GenerateKernels_ACTIVEAX] File "protocol_HR.scheme" not found in folder "%s"', ATOMS_path )
        end

        filenameHr = [tempname '.Bfloat'];

        % Restricted
        % ==========
        idx = 1;
        for R = obj.IC_Rs
            TIME = tic();
            fprintf( '\t\t- A_%03d... ', idx );

            % generate
            if exist( filenameHr, 'file' ), delete( filenameHr ); end
            CMD = sprintf( '%s/datasynth -synthmodel compartment 1 CYLINDERGPD %E 0 0 %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null', CAMINO_path, obj.dPar*1e-6, R*1e-6, schemeHrFilename, filenameHr );
            [status result] = system( CMD );
            if status>0
                disp(result)
                error( '[AMICO_ACTIVEAX.GenerateKernels] Problems generating the signal with datasynth' );
            end

            % rotate and save
            fid = fopen( filenameHr, 'r', 'b' );
            signal = fread(fid,'float');
            fclose(fid);
            delete( filenameHr );
            lm = AMICO_RotateKernel( signal, AUX, idx_IN, idx_OUT, false );
            save( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), '-v6', 'lm' )

            idx = idx + 1;
            fprintf( '[%.1f seconds]\n', toc(TIME) );
        end


        % Hindered
        % ========
        for ICVF = obj.IC_VFs
            TIME = tic();
            fprintf( '\t\t- A_%03d... ', idx );

            % generate
            d_perp = obj.dPar * ( 1.0 - ICVF );
            if exist( filenameHr, 'file' ), delete( filenameHr ); end
            CMD = sprintf( '%s/datasynth -synthmodel compartment 1 ZEPPELIN %E 0 0 %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null', CAMINO_path, obj.dPar*1e-6, d_perp*1e-6, schemeHrFilename, filenameHr );
            [status result] = system( CMD );
            if status>0
                disp(result)
                error( '[AMICO_ACTIVEAX.GenerateKernels] problems generating the signal' );
            end

            % rotate and save
            fid = fopen( filenameHr, 'r', 'b' );
            signal = fread(fid,'float');
            fclose(fid);
            delete( filenameHr );
            lm = AMICO_RotateKernel( signal, AUX, idx_IN, idx_OUT, false );
            save( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), '-v6', 'lm' )

            idx = idx + 1;
            fprintf( '[%.1f seconds]\n', toc(TIME) );
        end


        % Isotropic
        % =========
        TIME = tic();
        fprintf( '\t\t- A_%03d... ', idx );

        % generate
        if exist( filenameHr, 'file' ), delete( filenameHr ); end
        CMD = sprintf( '%s/datasynth -synthmodel compartment 1 BALL %E -schemefile %s -voxels 1 -outputfile %s 2> /dev/null', CAMINO_path, obj.dIso*1e-6, schemeHrFilename, filenameHr );
        [status result] = system( CMD );
        if status>0
            disp(result)
            error( '[AMICO_ACTIVEAX.GenerateKernels] problems generating the signal' );
        end

        % resample and save
        fid = fopen( filenameHr, 'r', 'b' );
        signal = fread(fid,'float');
        fclose(fid);
        delete( filenameHr );
        lm = AMICO_RotateKernel( signal, AUX, idx_IN, idx_OUT, true );
        save( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), '-v6', 'lm' )

        idx = idx + 1;
        fprintf( '[%.1f seconds]\n', toc(TIME) );
    end


    % ==============================================
    % Project kernels from harmonic to subject space
    % ==============================================
    function ResampleKernels( obj, ATOMS_path, idx_OUT, Ylm_OUT )
        global CONFIG AMICO_data_path KERNELS

        % Setup the KERNELS structure
        % ===========================
        nIC = numel(obj.IC_Rs);
        nEC = numel(obj.IC_VFs);

        KERNELS = {};
        KERNELS.model    = 'ACTIVEAX';
        KERNELS.nS       = CONFIG.scheme.nS;
        KERNELS.nA       = nIC + nEC + 1; % number of atoms

        KERNELS.dPar     = obj.dPar;

        KERNELS.Aic      = zeros( [KERNELS.nS nIC 181 181], 'single' );
        KERNELS.Aic_R    = zeros( 1, nIC, 'single' );

        KERNELS.Aec      = zeros( [KERNELS.nS nEC 181 181], 'single' );
        KERNELS.Aec_icvf = zeros( 1, nEC, 'single' );

        KERNELS.Aiso     = zeros( [KERNELS.nS 1], 'single' );
        KERNELS.Aiso_d   = NaN;


        % Restricted
        % ==========
        idx = 1;
        for i = 1:nIC
            TIME = tic();
            fprintf( '\t- A_%03d...  ', idx );

            load( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), 'lm' );

            KERNELS.Aic(:,i,:,:) = AMICO_ResampleKernel( lm, idx_OUT, Ylm_OUT, false );
            KERNELS.Aic_R(i)     = obj.IC_Rs(i);
            idx = idx + 1;

            fprintf( '[%.1f seconds]\n', toc(TIME) );
        end


        % Hindered
        % ========
        for i = 1:nEC
            TIME = tic();
            fprintf( '\t- A_%03d...  ', idx );

            load( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), 'lm' );

            KERNELS.Aec(:,i,:,:) = AMICO_ResampleKernel( lm, idx_OUT, Ylm_OUT, false );
            KERNELS.Aec_icvf(i)  = obj.IC_VFs(i);
            idx = idx + 1;

            fprintf( '[%.1f seconds]\n', toc(TIME) );
        end


        % Isotropic
        % =========
        TIME = tic();
        fprintf( '\t- A_%03d...  ', idx );

        load( fullfile( ATOMS_path, sprintf('A_%03d.mat',idx) ), 'lm' );

        KERNELS.Aiso   = AMICO_ResampleKernel( lm, idx_OUT, Ylm_OUT, true );
        KERNELS.Aiso_d = obj.dIso;
        idx = idx + 1;

        fprintf( '[%.1f seconds]\n', toc(TIME) );

    end


    % ===========================
    % Fit the model to each voxel
    % ===========================
    function [DIRs, MAPs] = Fit( obj )
        global CONFIG
        global niiSIGNAL niiMASK
        global KERNELS bMATRIX

        % setup the output files
        MAPs         = zeros( [CONFIG.dim(1:3) numel(obj.OUTPUT_names)], 'single' );
        DIRs         = zeros( [CONFIG.dim(1:3) 3], 'single' );

    	nIC = numel(obj.IC_Rs);
        nEC = numel(obj.IC_VFs);

        progress = ProgressBar( nnz(niiMASK.img) );
        for iz = 1:niiSIGNAL.hdr.dime.dim(4)
        for iy = 1:niiSIGNAL.hdr.dime.dim(3)
        for ix = 1:niiSIGNAL.hdr.dime.dim(2)
            if niiMASK.img(ix,iy,iz)==0, continue, end
            progress.update();

            % read the signal
            b0 = mean( squeeze( niiSIGNAL.img(ix,iy,iz,CONFIG.scheme.b0_idx) ) );
            if ( b0 < 1e-3 ), continue, end
            y = double( squeeze( niiSIGNAL.img(ix,iy,iz,:) ) ./ ( b0 + eps ) );
            y( y < 0 ) = 0; % [NOTE] this should not happen!

            % find the MAIN DIFFUSION DIRECTION using DTI
            [ ~, ~, V ] = AMICO_FitTensor( y, bMATRIX );
            Vt = V(:,1);
            if ( Vt(2)<0 ), Vt = -Vt; end

            % build the DICTIONARY
            [ i1, i2 ] = AMICO_Dir2idx( Vt );
            if numel(KERNELS.Aiso_d) > 0
                A = double( [ KERNELS.Aic(CONFIG.scheme.dwi_idx,:,i1,i2) KERNELS.Aec(CONFIG.scheme.dwi_idx,:,i1,i2) KERNELS.Aiso(CONFIG.scheme.dwi_idx) ] );
            else
                A = double( [ KERNELS.Aic(CONFIG.scheme.dwi_idx,:,i1,i2) KERNELS.Aec(CONFIG.scheme.dwi_idx,:,i1,i2) ] );
            end

            % fit AMICO
            y = y(CONFIG.scheme.dwi_idx);
            yy = [ 1 ; y ];
            AA = [ ones(1,size(A,2)) ; A ];

            % estimate IC and EC compartments and promote sparsity
            x = full( mexLasso( yy, AA, CONFIG.OPTIMIZATION.SPAMS_param ) );

            % STORE results
            DIRs(ix,iy,iz,:) = Vt;

            f1 = sum( x( 1:nIC ) );
            f2 = sum( x( (nIC+1):(nIC+nEC) ) );
            v = f1 / ( f1 + f2 + eps );
            a = 2 * KERNELS.Aic_R * x(1:nIC) / ( f1 + eps );

            MAPs(ix,iy,iz,1) = v;
            MAPs(ix,iy,iz,2) = a;
            MAPs(ix,iy,iz,3) = (4*v) / ( pi*a^2 + eps );
        end
        end
        end
        progress.close();
    end


    % ================================================================
    % Simulate signal according to tensor model (1 fiber along z-axis)
    % ================================================================
    function [ signal ] = TensorSignal( obj, D, XYZB )
        nDIR   = size( XYZB, 1 );
        signal = zeros( nDIR, 1 );
        for d = 1:nDIR
            signal(d) = exp(-XYZB(d,4) * XYZB(d,1:3) * D * XYZB(d,1:3)');
        end
    end

end

end
