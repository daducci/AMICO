%
% Call the Fit() method of the model to fit
%
function AMICO_Fit()

	global CONFIG niiMASK


	% Dispatch to the right handler for each model
    % ============================================
	if ~isempty(CONFIG.model)
        % delete previous output
        [~,~,~] = mkdir( CONFIG.OUTPUT_path );
		delete( fullfile(CONFIG.OUTPUT_path,'*') );

        % fit the model to the data
		[DIRs, MAPs] = CONFIG.model.Fit();
	else
		error( '[AMICO_Fit] Model not set' )
    end


	% Save CONFIGURATION and OUTPUT to file
    % =====================================
    fprintf( '\n-> Saving output to "AMICO/*":\n' );
    
    fprintf( '\t- CONFIG.mat' );
    save( fullfile(CONFIG.OUTPUT_path,'CONFIG.mat'), '-v6', 'CONFIG' )
    fprintf( ' [OK]\n' );
    
    fprintf( '\t- FIT_dir.nii' );
    niiMAP = niiMASK;
    niiMAP.hdr.dime.dim(1) = 4;
    niiMAP.hdr.dime.datatype = 16;
    niiMAP.hdr.dime.bitpix = 32;
    niiMAP.hdr.dime.glmin = -1;
    niiMAP.hdr.dime.glmax = 1;
    niiMAP.hdr.dime.calmin = niiMAP.hdr.dime.glmin;
    niiMAP.hdr.dime.calmax = niiMAP.hdr.dime.glmax;

    niiMAP.img = DIRs;
    niiMAP.hdr.dime.dim(5) = size(DIRs,4);
    save_untouch_nii( niiMAP, fullfile(CONFIG.OUTPUT_path,'FIT_dir.nii') );
    fprintf( ' [OK]\n' );

    niiMAP.hdr.dime.dim(5) = 1;
    for i = 1:numel(CONFIG.model.OUTPUT_names)
        fprintf( '\t- AMICO/FIT_%s.nii', CONFIG.model.OUTPUT_names{i} );

        niiMAP.img = MAPs(:,:,:,i);
        niiMAP.hdr.hist.descrip = CONFIG.model.OUTPUT_descriptions{i};
        niiMAP.hdr.dime.glmin = min(niiMAP.img(:));
        niiMAP.hdr.dime.glmax = max(niiMAP.img(:));
        niiMAP.hdr.dime.calmin = niiMAP.hdr.dime.glmin;
        niiMAP.hdr.dime.calmax = niiMAP.hdr.dime.glmax;
        save_untouch_nii( niiMAP, fullfile(CONFIG.OUTPUT_path,['FIT_' CONFIG.model.OUTPUT_names{i} '.nii']) );

        fprintf( ' [OK]\n' );
    end

    fprintf( '   [ DONE ]\n' )
end
