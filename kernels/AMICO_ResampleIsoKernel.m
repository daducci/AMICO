function [ Kr ] = AMICO_ResampleIsoKernel( K, scheme, AUX, idx_IN, idx_OUT, Ylm_OUT )
	Kr = zeros( size(scheme.camino,1), 1, 1, 'single' );
	Ylm_rot = AUX.Ylm_rot{ 1, 1 };
	for s = 1:numel(scheme.shells)
		Kr( idx_OUT{s}, 1, 1 ) = single( Ylm_OUT{s} * AUX.fit * K( idx_IN{s} ) );
	end
