function out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz, currentRotFactor)

% out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)
% 
% Extracts a sample for the scale filter at the current
% location and scale.

nScales = length(scaleFactors);

for s = 1:nScales
    patch_sz = floor(base_target_sz * scaleFactors(s));
    
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
    
    % check for out-of-bounds coordinates, and set them to the values at
    % the borders
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    % extract image
     im_patch = im(ys, xs, :);
%     center= [pos(2) pos(1)];
%     %make sure it's always odd valued
%     if mod(patch_sz(1), 2)==0
%         patch_sz(1)= patch_sz(1)+1;
%     end
%     if mod(patch_sz(2), 2)==0
%         patch_sz(2)= patch_sz(2)+1;
%     end
%     patch_sz= [patch_sz(2) patch_sz(1)];
%     im_patch = extractRotatedPatch(im, center, patch_sz(1), patch_sz(2), currentRotFactor);

    % resize image to model size
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
    % extract scale features
    temp_hog = fhog(single(im_patch_resized), 4);
    temp = temp_hog(:,:,1:31);
    
    if s == 1
        out = zeros(numel(temp), nScales, 'single');
    end
    
    % window
    out(:,s) = temp(:) * scale_window(s);
end