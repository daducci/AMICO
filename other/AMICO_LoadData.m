%
% Load data and perform some preprocessing
%
fprintf( '\n-> Loading and setup:\n' );

fprintf( '\t* Loading DWI...\n' );

% DWI dataset
niiSIGNAL = load_untouch_nii( CONFIG.dwiFilename );
niiSIGNAL.img = single(niiSIGNAL.img);
CONFIG.dim    = niiSIGNAL.hdr.dime.dim(2:5);
CONFIG.pixdim = niiSIGNAL.hdr.dime.pixdim(2:4);
fprintf( '\t\t- dim    = %d x %d x %d x %d\n' , CONFIG.dim );
fprintf( '\t\t- pixdim = %.3f x %.3f x %.3f\n', CONFIG.pixdim );

% DWI scheme
fprintf( '\t* Loading SCHEME...\n' );
CONFIG.scheme = AMICO_LoadScheme( CONFIG.schemeFilename );
if CONFIG.scheme.nS ~= CONFIG.dim(4)
	error( '[AMICO_LoadData] Data and scheme do not match\n' );
end
fprintf( '\t\t- %d measurements divided in %d shells (%d b=0)\n', CONFIG.scheme.nS, numel(CONFIG.scheme.shells), CONFIG.scheme.b0_count );


% BINARY mask
fprintf( '\t* Loading MASK...\n' );
if ~exist(CONFIG.maskFilename,'file')
	error( '[AMICO_LoadData] no mask specified\n' );
end

niiMASK = load_untouch_nii( CONFIG.maskFilename );
if nnz( CONFIG.dim(1:3) - niiMASK.hdr.dime.dim(2:4) ) > 0
	error( '[AMICO_LoadData] Data and mask do not match\n' );
end
fprintf( '\t\t- dim    = %d x %d x %d\n' , niiMASK.hdr.dime.dim(2:4) );
fprintf( '\t\t- voxels = %d\n' , nnz(niiMASK.img) );


% precompute the b-matrix to be used in DTI fitting
XYZB = CONFIG.scheme.camino(:,1:3);
XYZB(:,4) = CONFIG.scheme.b;
bMATRIX = zeros([3 3 size(XYZB,1)]);
for i = 1:size(XYZB,1)
	bMATRIX(:,:,i) = XYZB(i,4) * XYZB(i,1:3)' * XYZB(i,1:3);
end
bMATRIX = squeeze([bMATRIX(1,1,:),2*bMATRIX(1,2,:),2*bMATRIX(1,3,:),bMATRIX(2,2,:),2*bMATRIX(2,3,:),bMATRIX(3,3,:)])';
clear XYZB i


fprintf( '   [ DONE ]\n' );
