function out = get_rot_sample(im, pos, base_target_sz, rotFactors, rot_window, scale_model_sz)

% out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)
% 
% Extracts a sample for the scale filter at the current
% location and scale.

nRots = length(rotFactors);

for s = 1:nRots
    patch_sz = floor(base_target_sz);
    
%     xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
%     ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
%     
%     % check for out-of-bounds coordinates, and set them to the values at
%     % the borders
%     xs(xs < 1) = 1;
%     ys(ys < 1) = 1;
%     xs(xs > size(im,2)) = size(im,2);
%     ys(ys > size(im,1)) = size(im,1);
    
    % Rotate the image with the selected theta
    %imgrotated= imrotate(im, 360-rotFactors(s));
    center= [pos(2) pos(1)];
    patch_sz= [patch_sz(2) patch_sz(1)];
    if mod(patch_sz(1), 2)==0
        patch_sz(1)= patch_sz(1)+1;
    end
    if mod(patch_sz(2), 2)==0
        patch_sz(2)= patch_sz(2)+1;
    end
    
    % extract image
    im_patch = extractRotatedPatch(im, center, patch_sz(1), patch_sz(2), rotFactors(s));
    im_patch= uint8(im_patch);
%     if(rotFactors(s)==-5)
%         im_patch1= im_patch;
%     elseif (rotFactors(s)==5)
%         im_patch2= im_patch;
%     end
%     imshow(im_patch)
%     pause
%     
        
    % resize image to model size
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
    % extract scale features
    temp_hog = fhog(single(im_patch_resized), 4);
    temp = temp_hog(:,:,1:31);
%     figure(4); montage2(temp_hog);
    
    if s == 1
        out = zeros(numel(temp), nRots, 'single');
    end
    
    % window
    out(:,s) = temp(:) * rot_window(s);
end

% H2=fhog(single(im_patch2),8,9,.2,1);
% H1=fhog(single(im_patch1),8,9,.2,1);
% figure(1); imshow(im_patch1); figure(2); imshow(im_patch2);
% figure(3), montage2(H1); figure(4); montage2(H2);
% pause