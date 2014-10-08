%  Break a multiple-scan NIFTI file into multiple single-scan NIFTI files
%
%  Usage: expand_nii_scan(multi_scan_NIFTI_filename, [path_for_scan_files], [img_idx)
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function expand_nii_scan(filename, newpath, img_idx)

   if ~exist('newpath','var'), newpath = pwd; end
   if ~exist('img_idx','var'), img_idx = []; end

   nii = load_nii(filename, img_idx);
   cd(newpath);
   
   if isempty(img_idx)
      num_scan = nii.hdr.dime.dim(5);
      img_idx = 1:num_scan;
   else
      num_scan = length(img_idx);
   end

   nii.hdr.dime.dim(5) = 1;

   for i = 1:num_scan
      fn = [nii.fileprefix '_' sprintf('%04d',img_idx(i))];
      nii_i = nii;

      if nii.hdr.dime.datatype == 128 & nii.hdr.dime.bitpix == 24
         nii_i.img = nii_i.img(:,:,:,:,i);
      else
         nii_i.img = nii_i.img(:,:,:,i);
      end
 
      save_nii(nii_i, fn);
   end

   return;					% expand_nii_scan

