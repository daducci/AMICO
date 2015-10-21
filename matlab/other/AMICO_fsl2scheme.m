%
% Create a scheme file from bavals+bvecs and write to file.
% If required, b-values can be rounded up to a specific threshold (bstep parameter).
%
function AMICO_fsl2scheme( bvalsFilename, bvecsFilename, schemeFilename, bStep )    
    if nargin < 3
        error( '[AMICO_fsl2scheme] USAGE: AMICO_fsl2scheme <bvalsFilename> <bvecsFilename> <schemeFilename> [bStep]' )
    end
    if nargin < 4
        bStep = 1;
    end
    
    % load files and check size
    bvecs = dlmread( bvecsFilename );
    bvals = dlmread( bvalsFilename );
    if size(bvecs,1) ~= 3 || size(bvals,1) ~= 1 || size(bvecs,2) ~= size(bvals,2)
        error( '[AMICO_fsl2scheme] incorrect/incompatible bval/bvecs files' )
    end

    % if requested, round the b-values
    if bStep > 1
        bvals = round(bvals/bStep) * bStep;
    end
    
    % write corresponding scheme file
    dlmwrite( schemeFilename, 'VERSION: BVECTOR', 'delimiter','' );
    dlmwrite( schemeFilename, [ bvecs ; bvals ]', '-append', 'delimiter',' ' );
end
