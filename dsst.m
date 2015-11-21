function [positions, fps] = dsst(params)

% [positions, fps] = dsst(params)

% parameters
padding = params.padding;                         	%extra area surrounding the target
output_sigma_factor = params.output_sigma_factor;	%spatial bandwidth (proportional to target)
lambda = params.lambda;
learning_rate = params.learning_rate;
nScales = params.number_of_scales;
nRots= 11;
scale_step = params.scale_step;
scale_sigma_factor = params.scale_sigma_factor;
scale_model_max_area = params.scale_model_max_area;

video_path = params.video_path;
img_files = params.img_files;
pos = floor(params.init_pos);

target_sz = floor(params.wsize);

visualization = params.visualization;

num_frames = numel(img_files);

init_target_sz = target_sz;

% target size att scale = 1
base_target_sz = target_sz;
% if mod(base_target_sz(1), 2)==0
%     base_target_sz(1)= base_target_sz(1)+1;
% end
% if mod(base_target_sz(2), 2)==0
%     base_target_sz(2)= base_target_sz(2)+1;
% end

% window size, taking padding into account
sz = floor(base_target_sz * (1 + padding));

% desired translation filter output (gaussian shaped), bandwidth
% proportional to target size
output_sigma = sqrt(prod(base_target_sz)) * output_sigma_factor;
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = single(fft2(y));


% desired scale filter output (gaussian shaped), bandwidth proportional to
% number of scales
scale_sigma = nScales/sqrt(33) * scale_sigma_factor;
ss = (1:nScales) - ceil(nScales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));

rr = (1:nRots) - ceil(nRots/2);
yr = exp(-0.5 * (rr.^2) / scale_sigma^2);
yrf = single(fft(yr));


% store pre-computed translation filter cosine window
cos_window = single(hann(sz(1)) * hann(sz(2))');

% store pre-computed scale filter cosine window
if mod(nScales,2) == 0
    scale_window = single(hann(nScales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(nScales));
end;

% store pre-computed rotation filter cosine window
if mod(nRots,2) == 0
    rot_window = single(hann(nRots+1));
    rot_window = rot_window(2:end);
else
    rot_window = single(hann(nRots));
end;

% scale factors
ss = 1:nScales;
scaleFactors = scale_step.^(ceil(nScales/2) - ss);

rot_max= nRots-1;
rot_step= 1;
rotFactors = (-rot_max/2:rot_step:rot_max/2);

% compute the resize dimensions used for feature extraction in the scale
% estimation
scale_model_factor = 1;
if prod(init_target_sz) > scale_model_max_area
    scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
end
scale_model_sz = floor(init_target_sz * scale_model_factor);

currentScaleFactor = 1;
currentRotFactor = 0;

% to calculate precision
positions = zeros(numel(img_files), 4);

% to calculate FPS
time = 0;

% find maximum and minimum scales
im = imread([video_path img_files{1}]);
min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));


for frame = 1:num_frames,
    %load image
    im = imread([video_path img_files{frame}]);

    tic;
    
    if frame > 1
        
        % extract the test sample feature map for the translation filter
        xt = get_translation_sample(im, pos, sz, currentScaleFactor, currentRotFactor, cos_window);
        
        % calculate the correlation response of the translation filter
        xtf = fft2(xt);
        response = real(ifft2(sum(hf_num .* xtf, 3) ./ (hf_den + lambda)));
        
        % find the maximum translation response
        [row, col] = find(response == max(response(:)), 1);
        
        % update the position
        pos = pos + round((-sz/2 + [row, col]) * currentScaleFactor);
        
        % extract the test sample feature map for the scale filter
        xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz, currentRotFactor);
        
        % calculate the correlation response of the scale filter
        xsf = fft(xs,[],2);
        scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda)));
        
        % find the maximum scale response
        recovered_scale = find(scale_response == max(scale_response(:)), 1);
        
        % update the scale
        currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
        
        % extract the test sample feature map for the scale filter
        xr = get_rot_sample(im, pos, base_target_sz.*currentScaleFactor, currentRotFactor+rotFactors, rot_window, scale_model_sz);
        
        % calculate the correlation response of the scale filter
        xrf = fft(xr,[],2);
        rot_response = real(ifft(sum(rf_num .* xrf, 1) ./ (rf_den + lambda)));
        
        % find the maximum rotation response
        recovered_rot = find(rot_response == max(rot_response(:)), 1);
        
        % update the rotation
        currentRotFactor = currentRotFactor + rotFactors(recovered_rot);
        [frame currentRotFactor]
    end
    
    % extract the training sample feature map for the translation filter
    xl = get_translation_sample(im, pos, sz, currentScaleFactor, currentRotFactor, cos_window);
    
    % calculate the translation filter update
    xlf = fft2(xl);
    new_hf_num = bsxfun(@times, yf, conj(xlf));
    new_hf_den = sum(xlf .* conj(xlf), 3);
    
    % extract the training sample feature map for the scale filter
    xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz, currentRotFactor);
    
    % calculate the scale filter update
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    
    % extract the test sample feature map for the scale filter
    xr = get_rot_sample(im, pos, base_target_sz.*currentScaleFactor, currentRotFactor+ rotFactors, rot_window, scale_model_sz);
    
    % calculate the scale filter update
    xrf = fft(xr,[],2);
    new_rf_num = bsxfun(@times, yrf, conj(xrf));
    new_rf_den = sum(xrf .* conj(xrf), 1);

    if frame == 1
        % first frame, train with a single image
        hf_den = new_hf_den;
        hf_num = new_hf_num;
        
        sf_den = new_sf_den;
        sf_num = new_sf_num;
        
        rf_den = new_rf_den;
        rf_num = new_rf_num;
    else
        % subsequent frames, update the model
        hf_den = (1 - learning_rate) * hf_den + learning_rate * new_hf_den;
        hf_num = (1 - learning_rate) * hf_num + learning_rate * new_hf_num;
        sf_den = (1 - learning_rate) * sf_den + learning_rate * new_sf_den;
        sf_num = (1 - learning_rate) * sf_num + learning_rate * new_sf_num;
        rf_den = (1 - learning_rate) * rf_den + learning_rate * new_rf_den;
        rf_num = (1 - learning_rate) * rf_num + learning_rate * new_rf_num;
    end
    
    % calculate the new target size
    target_sz = floor(base_target_sz * currentScaleFactor);
%     if mod(target_sz(1), 2)==0
%         target_sz(1)= target_sz(1)+1;
%     end
%     if mod(target_sz(2), 2)==0
%         target_sz(2)= target_sz(2)+1;
%     end

    %save position
    positions(frame,:) = [pos target_sz];
    
    time = time + toc;
    
    
    %visualization
%     visualization=0;
    if visualization == 1
        theta= currentRotFactor*pi/180;
        rotatemat = [cos(theta) sin(theta);-sin(theta) cos(theta)]; % generate rotation matrix

        xv=[pos(2)-target_sz(2)/2 pos(2)-target_sz(2)/2+target_sz(2) pos(2)-target_sz(2)/2+target_sz(2) pos(2)-target_sz(2)/2 pos(2)-target_sz(2)/2];
        yv=[pos(1)-target_sz(1)/2 pos(1)-target_sz(1)/2 pos(1)-target_sz(1)/2+target_sz(1) pos(1)-target_sz(1)/2+target_sz(1) pos(1)-target_sz(1)/2];
        xv= xv-pos(2);
        yv= yv-pos(1);
        R(1,:)=xv;R(2,:)=yv;
        XY= rotatemat*R;
        xvR= XY(1, :);
        yvR= XY(2, :);
        xvR= xvR+pos(2);
        yvR= yvR+pos(1);

        imshow(uint8(im))
        hold on
        plot(xvR,yvR, 'r');axis equal;
        hold off;
        pause(0.1);
        
%         rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
%         if frame == 1,  %first frame, create GUI
%             figure('NumberTitle','off', 'Name',['Tracker - ' video_path]);
%             im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
%             rect_handle = rectangle('Position',rect_position, 'EdgeColor','g');
%             text_handle = text(10, 10, int2str(frame));
%             set(text_handle, 'color', [0 1 1]);
%         else
%             try  %subsequent frames, update GUI
%                 set(im_handle, 'CData', im)
%                 set(rect_handle, 'Position', rect_position)
%                 set(text_handle, 'string', int2str(frame));
%             catch
%                 return
%             end
%         end
        
%         drawnow
%         pause
    end
end
fps = num_frames/time;
end