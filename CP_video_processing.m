%% ========================================================================
% MARKER TRACKING FOR PHRESCO CRUMPLED PAPER EXAMPLE
% ========================================================================
% Morphological Computation Group
% University of Bristol
% 2025
%
% Description:
%   Tracks servo (green) and marker (blue) positions from video for 
%   physical reservoir computing with crumpled paper system.
%   Outputs tracking data for NARMA benchmark evaluation.
%
% Date: 2025
% ========================================================================

clear; clc; close all;

%% ========================================================================
% USER PARAMETERS
% ========================================================================

% Video input
videoFile = 'motion_tracking_50_3freq_r2.mov';  % Input video filename
outputMatFile = 'motion_tracking_50_3freq_2.mat';  % Output data filename

% System parameters
NUM_MARKERS = 11;  % Number of blue markers to track

% Get frame rate from user
fprintf('=== Video Processing for Physical Reservoir Computing ===\n');
fprintf('Video file: %s\n', videoFile);
defaultFPS = 60;
userInput = input(sprintf('Enter video frame rate (Hz) [default=%d]: ', defaultFPS), 's');
if isempty(userInput)
    FRAME_RATE = defaultFPS;
else
    FRAME_RATE = str2double(userInput);
end
fprintf('Using frame rate: %d Hz\n', FRAME_RATE);

%% ========================================================================
% VIDEO INITIALIZATION
% ========================================================================

fprintf('\n--- Initializing Video ---\n');
v = VideoReader(videoFile);
actualFPS = v.FrameRate;
numFrames = floor(v.Duration * v.FrameRate) - 1;

% Verify frame rate
if abs(actualFPS - FRAME_RATE) > 1
    warning('Actual video frame rate (%.2f) differs from specified (%.2f)', ...
            actualFPS, FRAME_RATE);
    useActual = input('Use actual frame rate? (y/n): ', 's');
    if strcmpi(useActual, 'y')
        FRAME_RATE = actualFPS;
    end
end

fprintf('Processing %d frames at %.2f Hz\n', numFrames, FRAME_RATE);

%% ========================================================================
% MEMORY ALLOCATION
% ========================================================================

fprintf('\n--- Allocating Memory ---\n');

% Servo tracking
servoPos = NaN(numFrames, 2);           % [x, y] position of servo marker
servoDisplacement = NaN(numFrames, 1);   % Frame-to-frame displacement

% Marker tracking  
markerPos = NaN(numFrames, NUM_MARKERS, 2);  % [frame, marker, xy]
markerDisplacement = NaN(numFrames, NUM_MARKERS);  % Distance from origin

% For tracking continuity
prevMarkerPos = NaN(NUM_MARKERS, 2);

% Suppress figure warnings
warning('off', 'images:initSize:adjustingMag');

%% ========================================================================
% COLOR DETECTION PARAMETERS
% ========================================================================

% HSV ranges for color detection
% Green (servo marker)
GREEN_HUE = [0.25, 0.55];
GREEN_SAT = [0.30, 1.00];
GREEN_VAL = [0.20, 1.00];

% Blue (position markers)
BLUE_HUE = [0.55, 0.85];
BLUE_SAT = [0.40, 1.00];
BLUE_VAL = [0.10, 1.00];

%% ========================================================================
% VIDEO PROCESSING LOOP
% ========================================================================

fprintf('\n--- Processing Video ---\n');
fprintf('Tracking %d markers...\n', NUM_MARKERS);

% Setup visualization
fig = figure('Name', 'Marker Tracking', 'Position', [100, 100, 800, 600]);

% Progress tracking
progressStep = floor(numFrames / 10);

for frameIdx = 1:numFrames
    
    % Progress indicator
    if mod(frameIdx, progressStep) == 0
        fprintf('  Progress: %d%%\n', round(100 * frameIdx / numFrames));
    end
    
    % Read and convert frame
    frame = readFrame(v);
    hsvFrame = rgb2hsv(frame);
    
    % Extract HSV channels
    hue = hsvFrame(:,:,1);
    sat = hsvFrame(:,:,2);
    val = hsvFrame(:,:,3);
    
    % =====================================================================
    % SERVO (GREEN) DETECTION
    % =====================================================================
    
    greenMask = (hue > GREEN_HUE(1) & hue < GREEN_HUE(2)) & ...
                (sat > GREEN_SAT(1) & sat < GREEN_SAT(2)) & ...
                (val > GREEN_VAL(1) & val < GREEN_VAL(2));
    
    % Clean up mask - keep largest blob
    greenMask = bwareafilt(greenMask, 1);
    
    % Extract servo position
    statsGreen = regionprops(greenMask, 'Centroid');
    if ~isempty(statsGreen)
        servoPos(frameIdx, :) = statsGreen(1).Centroid;
        
        % Calculate frame-to-frame displacement
        if frameIdx > 1 && all(~isnan(servoPos(frameIdx-1, :)))
            dx = servoPos(frameIdx, 1) - servoPos(frameIdx-1, 1);
            dy = servoPos(frameIdx, 2) - servoPos(frameIdx-1, 2);
            servoDisplacement(frameIdx) = sqrt(dx^2 + dy^2);
        else
            servoDisplacement(frameIdx) = 0;
        end
    end
    
    % =====================================================================
    % BLUE MARKER DETECTION
    % =====================================================================
    
    blueMask = (hue > BLUE_HUE(1) & hue < BLUE_HUE(2)) & ...
               (sat > BLUE_SAT(1) & sat < BLUE_SAT(2)) & ...
               (val > BLUE_VAL(1) & val < BLUE_VAL(2));
    
    % Morphological operations to clean up detection
    blueMask = imopen(blueMask, strel('disk', 2));   % Remove noise
    blueMask = imclose(blueMask, strel('disk', 2));  % Fill gaps
    blueMask = imfill(blueMask, 'holes');            % Fill holes
    
    % Find all blue regions
    statsBlue = regionprops(blueMask, 'Centroid', 'Area');
    
    % Process if enough markers detected
    if length(statsBlue) >= NUM_MARKERS
        
        % Extract centroids
        centroids = reshape([statsBlue.Centroid], 2, [])';
        
        if frameIdx == 1
            % ============================================================
            % FIRST FRAME: Initialize tracking
            % ============================================================
            % Randomly select NUM_MARKERS from detected blobs
            randIdx = randperm(size(centroids, 1));
            selectedCentroids = centroids(randIdx(1:NUM_MARKERS), :);
            markerPos(frameIdx, :, :) = selectedCentroids;
            prevMarkerPos = selectedCentroids;
            
        else
            % ============================================================
            % SUBSEQUENT FRAMES: Nearest neighbor tracking
            % ============================================================
            currentMarkers = NaN(NUM_MARKERS, 2);
            assigned = zeros(size(centroids, 1), 1);
            
            % Match each previous marker to nearest current detection
            for markerIdx = 1:NUM_MARKERS
                prevPt = prevMarkerPos(markerIdx, :);
                
                % Calculate distances to all unassigned detections
                distances = sqrt(sum((centroids - prevPt).^2, 2));
                distances(assigned == 1) = inf;  % Exclude already assigned
                
                % Find closest unassigned detection
                [~, closestIdx] = min(distances);
                currentMarkers(markerIdx, :) = centroids(closestIdx, :);
                assigned(closestIdx) = 1;
            end
            
            % Store positions
            markerPos(frameIdx, :, :) = currentMarkers;
            prevMarkerPos = currentMarkers;
            
            % Calculate displacement (distance from origin - to be revised)
            for markerIdx = 1:NUM_MARKERS
                x = markerPos(frameIdx, markerIdx, 1);
                y = markerPos(frameIdx, markerIdx, 2);
                markerDisplacement(frameIdx, markerIdx) = sqrt(x^2 + y^2);
            end
        end
    else
        % Not enough markers detected - mark as NaN
        markerPos(frameIdx, :, :) = NaN;
        fprintf('  Warning: Frame %d - only %d markers detected\n', ...
                frameIdx, length(statsBlue));
    end
    
    % =====================================================================
    % VISUALIZATION
    % =====================================================================
    
    imshow(frame); hold on;
    
    % Draw servo marker
    if ~any(isnan(servoPos(frameIdx, :)))
        plot(servoPos(frameIdx, 1), servoPos(frameIdx, 2), ...
             'go', 'MarkerSize', 10, 'LineWidth', 2);
        text(servoPos(frameIdx, 1) + 5, servoPos(frameIdx, 2), ...
             'Servo', 'Color', 'g', 'FontSize', 10, 'FontWeight', 'bold');
    end
    
    % Draw tracked markers
    for markerIdx = 1:NUM_MARKERS
        pt = squeeze(markerPos(frameIdx, markerIdx, :));
        if ~any(isnan(pt))
            plot(pt(1), pt(2), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
            text(pt(1) + 5, pt(2), sprintf('M%d', markerIdx), ...
                 'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
        end
    end
    
    title(sprintf('Frame %d / %d', frameIdx, numFrames));
    hold off;
    drawnow;
end

fprintf('Video processing complete!\n');

%% ========================================================================
% SAVE TRACKING DATA
% ========================================================================

fprintf('\n--- Saving Data ---\n');

% Organize data structure
trackingData = struct();

% Metadata
trackingData.metadata.videoFile = videoFile;
trackingData.metadata.numFrames = numFrames;
trackingData.metadata.numMarkers = NUM_MARKERS;
trackingData.metadata.processDate = datestr(now);

% Frame information
trackingData.frameNumbers = (1:numFrames)';
trackingData.samplerate = FRAME_RATE;

% Servo data
trackingData.servoPos = servoPos;                 % [x, y] positions
trackingData.servoDisplacement = servoDisplacement; % Frame-to-frame displacement

% Marker data  
trackingData.blueXPos = markerPos;                % [frame, marker, xy]
trackingData.markerDisplacement = markerDisplacement; % Distance from origin

% Legacy field names for compatibility
trackingData.input = servoDisplacement;
trackingData.output = markerDisplacement;

% Save to file
save(outputMatFile, 'trackingData');

% Report summary
fprintf('\n=== TRACKING SUMMARY ===\n');
fprintf('Video: %s\n', videoFile);
fprintf('Frames processed: %d\n', numFrames);
fprintf('Frame rate: %.2f Hz\n', FRAME_RATE);
fprintf('Duration: %.2f seconds\n', numFrames / FRAME_RATE);
fprintf('Markers tracked: %d\n', NUM_MARKERS);
fprintf('Output saved to: %s\n', outputMatFile);

validServoFrames = sum(~isnan(servoPos(:, 1)));
validMarkerFrames = sum(~isnan(markerPos(:, 1, 1)));
fprintf('\nTracking success rate:\n');
fprintf('  Servo: %.1f%% (%d/%d frames)\n', ...
        100 * validServoFrames / numFrames, validServoFrames, numFrames);
fprintf('  Markers: %.1f%% (%d/%d frames)\n', ...
        100 * validMarkerFrames / numFrames, validMarkerFrames, numFrames);

fprintf('\n✅ Processing complete!\n');

% Close figure
close(fig);