%% ========================================================================
% MARKER TRACKING FOR PHRESCO CRUMPLED PAPER EXAMPLE
% ========================================================================
% Morphological Computation Group
% University of Bristol
% 2025
%
% Description:
%    NARMA benchmark for physical reservoir computing:
%      1. Load marker tracking data from a MAT file
%      2. Generate NARMA series as ground truth
%      3. Train a linear readout to predict NARMA from marker positions
%      4. Evaluate and visualize the results with comprehensive error analysis
%
% Date: 2025
% ========================================================================

clear

% =========================================================================
% MAIN EXECUTION
% =========================================================================
results = narma_benchmark( ...
    'motion_tracking_50_3freq.mat', ... % MAT file with tracking data
    'TrainFraction', 0.8,            ... % 80% train, 20% test
    'WashoutFraction', 0.2,          ... % 20% initial washout
    'NarmaOrder', 2,                ... % NARMA-2 (can be 2, 3, 4, etc.)
    'InterpFactor', 1,               ... % Upsample (to assist trajectory visualisation when needed)
    'CenterSignals', true,           ... % Centre signals for better visualization
    'NormalizeSignals', true);       ... % Normalize to [-1,1] for better visualization

fprintf('\n=== FINAL RESULTS ===\n');
fprintf('NMSE: %.4f\n', results.metrics.NMSE);

% =========================================================================
% MAIN BENCHMARK FUNCTION
% =========================================================================
function results = narma_benchmark(matFile, varargin)

% -------------------------------------------------------------------------
% INPUT PARSING
% -------------------------------------------------------------------------
p = inputParser;
addRequired(p, 'matFile');
addParameter(p, 'TrainFraction', 0.7, @(x)isnumeric(x)&&x>0&&x<1);
addParameter(p, 'WashoutFraction', 0.2, @(x)isnumeric(x)&&x>=0&&x<1);
addParameter(p, 'InterpFactor', 1, @(x)isnumeric(x)&&x>=1);
addParameter(p, 'InterpMethod', 'pchip', @(x)ischar(x)||isstring(x));
addParameter(p, 'NarmaOrder', 2, @(x)isnumeric(x)&&x>=2&&mod(x,1)==0);
addParameter(p, 'CenterSignals', false, @islogical);
addParameter(p, 'NormalizeSignals', false, @islogical);

parse(p, matFile, varargin{:});
opts = p.Results;

% =========================================================================
% STEP 1: LOAD TRACKING DATA
% =========================================================================
fprintf('\n=== Loading tracking data from %s ===\n', matFile);

data = load(matFile);

% Handle different possible MAT file structures
if isfield(data, 'trackingData')
    tracking = data.trackingData;
elseif length(fieldnames(data)) == 1
    fn = fieldnames(data);
    tracking = data.(fn{1});
else
    tracking = data;
end

% Extract essential data
frame_numbers = tracking.frameNumbers;
servo_input = tracking.servoPos(:,1);      % Servo displacement (input)
markers_3D = tracking.blueXPos;            % Marker positions [samples x markers x 2(X,Y)]

% Get sample rate for physical time
if isfield(tracking, 'samplerate')
    sample_rate = tracking.samplerate;
    time_vec = frame_numbers / sample_rate;  % Convert to seconds
else
    sample_rate = 1;
    time_vec = frame_numbers;
end

% Get dimensions
[nSamples, nMarkers, nDims] = size(markers_3D);
fprintf('Loaded: %d samples, %d markers, %d dimensions\n', nSamples, nMarkers, nDims);
fprintf('Duration: %.2f seconds at %d Hz\n', time_vec(end)-time_vec(1), sample_rate);

% Reshape marker data: [samples x markers x 2] -> [samples x (markers*2)]
% This treats X,Y coordinates as separate channels
marker_states = reshape(markers_3D, nSamples, nMarkers * nDims);
fprintf('Reservoir states: %d channels (X,Y for each marker)\n', size(marker_states, 2));

% Ensure column vectors
servo_input = servo_input(:);
time_vec = time_vec(:);

% =========================================================================
% STEP 2: GENERATE NARMA GROUND TRUTH
% =========================================================================
fprintf('\n=== Generating NARMA-%d target series ===\n', opts.NarmaOrder);

% Normalize servo input to [0, 0.5] (standard NARMA input range)
u_norm = servo_input - min(servo_input);
u_norm = 0.5 * u_norm / max(u_norm);

% Initialize NARMA output
y_narma = zeros(nSamples, 1);

% Generate NARMA series based on order
switch opts.NarmaOrder
    case 2
        % NARMA-2 equations
        for k = 2:nSamples-1
            y_narma(k+1) = 0.4*y_narma(k) + ...
                           0.4*y_narma(k)*y_narma(k-1) + ...
                           0.6*u_norm(k)^3 + 0.1;
        end
        
    case 3
        % NARMA-3 equations
        a = 0.3; b = 0.05; c = 1.5; d = 0.1;
        for k = 3:nSamples-1
            y_sum = sum(y_narma(k-2:k));
            y_narma(k+1) = a*y_narma(k) + ...
                           b*y_narma(k)*y_sum + ...
                           c*u_norm(k-2)*u_norm(k) + d;
        end
        
    otherwise
        % NARMA-n for n >= 4
        n = opts.NarmaOrder;
        a = 0.3; b = 0.05; c = 1.5; d = 0.1;
        for k = n:nSamples-1
            y_sum = sum(y_narma(k-(n-1):k));
            y_narma(k+1) = a*y_narma(k) + ...
                           b*y_narma(k)*y_sum + ...
                           c*u_norm(k-n+1)*u_norm(k) + d;
        end
end

% Check for numerical issues and normalize if needed
if max(abs(y_narma)) > 10
    fprintf('WARNING: NARMA output exceeded safe range. Normalizing...\n');
    y_narma = y_narma / max(abs(y_narma));
end

fprintf('NARMA output range: [%.3f, %.3f]\n', min(y_narma), max(y_narma));

% =========================================================================
% STEP 3: INTERPOLATION (OPTIONAL)
% =========================================================================
if opts.InterpFactor > 1
    fac = round(opts.InterpFactor);
    fprintf('\n=== Upsampling signals by %dx ===\n', fac);
    
    % Create new time grid
    t_old = 1:nSamples;
    t_new = linspace(1, nSamples, nSamples * fac);
    
    % Interpolate all signals
    marker_states = interp1(t_old, marker_states, t_new', opts.InterpMethod);
    servo_input = interp1(t_old, servo_input, t_new', opts.InterpMethod);
    y_narma = interp1(t_old, y_narma, t_new', opts.InterpMethod);
    time_vec = interp1(t_old, time_vec, t_new', 'linear');
    
    % Update sample count
    nSamples = length(t_new);
    fprintf('New sample count: %d\n', nSamples);
end

% =========================================================================
% STEP 4: PREPROCESSING (OPTIONAL)
% =========================================================================
% Store normalization statistics for potential inverse transform
norm_stats = struct();

if opts.NormalizeSignals
    fprintf('\n=== Normalizing signals to [-1, 1] ===\n');
    
    % Store original ranges
    norm_stats.marker_min = min(marker_states, [], 1);
    norm_stats.marker_max = max(marker_states, [], 1);
    norm_stats.servo_min = min(servo_input);
    norm_stats.servo_max = max(servo_input);
    norm_stats.narma_min = min(y_narma);
    norm_stats.narma_max = max(y_narma);
    
    % Normalize to [-1, 1]
    marker_states = 2 * (marker_states - norm_stats.marker_min) ./ (norm_stats.marker_max - norm_stats.marker_min) - 1;
    servo_input = 2 * (servo_input - norm_stats.servo_min) / (norm_stats.servo_max - norm_stats.servo_min) - 1;
    y_narma = 2 * (y_narma - norm_stats.narma_min) / (norm_stats.narma_max - norm_stats.narma_min) - 1;
end

if opts.CenterSignals
    fprintf('=== Centering signals (removing mean) ===\n');
    
    marker_states = marker_states - mean(marker_states, 1);
    servo_input = servo_input - mean(servo_input);
    y_narma = y_narma - mean(y_narma);
end

% =========================================================================
% STEP 5: PREPARE FEATURE MATRIX
% =========================================================================
% Feature matrix: [marker_states, bias_term]
X = [marker_states, ones(nSamples, 1)];
y = y_narma;

fprintf('\nFeature matrix: %d samples × %d features (including bias)\n', ...
    size(X, 1), size(X, 2));

% =========================================================================
% STEP 6: TRAIN/TEST SPLIT
% =========================================================================
fprintf('\n=== Splitting data ===\n');

% Washout: discard initial transient samples
washout_samples = round(opts.WashoutFraction * nSamples);
valid_idx = (washout_samples + 1):nSamples;

% Split remaining data into train/test
n_valid = length(valid_idx);
n_train = round(opts.TrainFraction * n_valid);

train_idx = valid_idx(1:n_train);
test_idx = valid_idx(n_train + 1:end);

fprintf('Washout: %d samples (%.1f%%)\n', washout_samples, opts.WashoutFraction * 100);
fprintf('Training: %d samples (%.1f%%)\n', length(train_idx), opts.TrainFraction * 100);
fprintf('Testing: %d samples (%.1f%%)\n', length(test_idx), (1-opts.TrainFraction) * 100);

% Extract train/test sets
X_train = X(train_idx, :);
y_train = y(train_idx);
X_test = X(test_idx, :);
y_test = y(test_idx);

% =========================================================================
% STEP 7: TRAIN LINEAR READOUT
% =========================================================================
fprintf('\n=== Training linear readout (least squares) ===\n');

% Solve: W = argmin ||X_train * W - y_train||²
W = X_train \ y_train;

fprintf('Readout weights computed: %d weights\n', length(W));

% =========================================================================
% STEP 8: EVALUATE PERFORMANCE
% =========================================================================
fprintf('\n=== Evaluating on test set ===\n');

% Make predictions
y_pred = X_test * W;

% Compute metrics
mse = mean((y_pred - y_test).^2);
nmse = mse / var(y_test);
nrmse = sqrt(mse) / std(y_test);
mae = mean(abs(y_pred - y_test));

% R-squared
ss_res = sum((y_test - y_pred).^2);
ss_tot = sum((y_test - mean(y_test)).^2);
r2 = 1 - ss_res / ss_tot;

% Store metrics
metrics = struct('MSE', mse, 'NMSE', nmse, 'NRMSE', nrmse, 'MAE', mae, 'R2', r2);

fprintf('\nTest Set Performance:\n');
fprintf('  MSE:   %.6f\n', mse);
fprintf('  NMSE:  %.6f\n', nmse);
fprintf('  NRMSE: %.6f\n', nrmse);
fprintf('  MAE:   %.6f\n', mae);
fprintf('  R²:    %.6f\n', r2);

% =========================================================================
% STEP 9: VISUALIZATION
% =========================================================================

% --- Figure 1: Main Results - Prediction vs Ground Truth ---
figure('Name', 'NARMA Results', 'Position', [100, 100, 1200, 500]);

% Subplot 1: Full time series
subplot(1, 2, 1);
time_test = time_vec(test_idx);
plot(time_test, y_test, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground Truth');
hold on;
plot(time_test, y_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Prediction');
xlabel('Time (seconds)');
ylabel('NARMA Output');
title(sprintf('NARMA-%d Prediction (NMSE = %.4f)', opts.NarmaOrder, nmse));
legend('Location', 'best');
grid on;

% Subplot 2: Zoomed view (first 20% of test data)
subplot(1, 2, 2);
zoom_idx = 1:round(0.2 * length(test_idx));
plot(time_test(zoom_idx), y_test(zoom_idx), 'b-', 'LineWidth', 1.5);
hold on;
plot(time_test(zoom_idx), y_pred(zoom_idx), 'r--', 'LineWidth', 1.5);
xlabel('Time (seconds)');
ylabel('NARMA Output');
title('Zoomed View (20% of test data)');
legend('Ground Truth', 'Prediction', 'Location', 'best');
grid on;

% --- Figure 2: System Overview with All Markers ---
figure('Name', 'System Overview', 'Position', [100, 650, 1200, 700]);

% Input signal
subplot(3, 1, 1);
plot(time_vec, servo_input, 'k-', 'LineWidth', 1);
ylabel('Servo Input');
title('Physical Reservoir Computing: Input → Markers → NARMA Output');
grid on;
xlim([time_vec(1), time_vec(end)]);

% All marker positions (X and Y separately)
subplot(3, 1, 2);
colors = lines(nMarkers);
hold on;
% Plot each marker's X and Y coordinates
for i = 1:nMarkers
    x_idx = (i-1)*2 + 1;
    y_idx = (i-1)*2 + 2;
    % Use transparency to avoid clutter
    plot(time_vec, marker_states(:, x_idx), '-', 'LineWidth', 0.8, ...
         'Color', [colors(i,:) 0.5], 'DisplayName', sprintf('M%d-X', i));
    plot(time_vec, marker_states(:, y_idx), '--', 'LineWidth', 0.8, ...
         'Color', [colors(i,:) 0.5], 'DisplayName', sprintf('M%d-Y', i));
end
ylabel('Position');
title(sprintf('All %d Markers (X: solid, Y: dashed)', nMarkers));
grid on;
xlim([time_vec(1), time_vec(end)]);
% Show legend only if markers <= 6
if nMarkers <= 6
    legend('Location', 'eastoutside', 'NumColumns', 2);
end

% NARMA output
subplot(3, 1, 3);
plot(time_vec, y_narma, 'g-', 'LineWidth', 1);
hold on;
% Mark train/test regions
xline(time_vec(train_idx(1)), 'k--', 'LineWidth', 1);
xline(time_vec(test_idx(1)), 'r--', 'LineWidth', 1);
text(mean(time_vec(train_idx)), min(y_narma)*1.1, 'TRAIN', ...
     'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(mean(time_vec(test_idx)), min(y_narma)*1.1, 'TEST', ...
     'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'red');
ylabel('NARMA Target');
xlabel('Time (seconds)');
grid on;
xlim([time_vec(1), time_vec(end)]);

% --- Figure 3: Error Analysis (NMSE-based) ---
figure('Name', 'Error Analysis', 'Position', [1350, 100, 800, 600]);

% Error over time
subplot(2, 1, 1);
error = y_pred - y_test;
plot(time_test, error, 'k-', 'LineWidth', 1);
hold on;
yline(0, 'r--', 'LineWidth', 1);
xlabel('Time (seconds)');
ylabel('Prediction Error');
title('Prediction Error Over Time');
grid on;

% Error histogram
subplot(2, 1, 2);
histogram(error, 30, 'FaceColor', [0.3 0.3 0.3]);
xlabel('Error');
ylabel('Frequency');
title(sprintf('Error Distribution (μ=%.4f, σ=%.4f)', mean(error), std(error)));
grid on;

% --- Figure 4: 2D Marker Trajectories ---
if nMarkers <= 16  % Only show if reasonable number of markers
    figure('Name', 'Marker Trajectories', 'Position', [1350, 750, 1200, 900]);
    
    % Determine subplot layout
    nCols = ceil(sqrt(nMarkers));
    nRows = ceil(nMarkers / nCols);
    
    for i = 1:nMarkers
        subplot(nRows, nCols, i);
        x_idx = (i-1)*2 + 1;
        y_idx = (i-1)*2 + 2;
        
        % Plot trajectory colored by time
        scatter(marker_states(:, x_idx), marker_states(:, y_idx), ...
                2, time_vec, 'filled');
        colormap(jet);
        
        axis equal;
        grid on;
        title(sprintf('Marker %d', i));
        xlabel('X'); 
        ylabel('Y');
        
        % Add colorbar to first subplot
        if i == 1
            cb = colorbar;
            ylabel(cb, 'Time (s)', 'FontSize', 8);
        end
    end
    sgtitle('2D Marker Trajectories (colored by time)');
end

% =========================================================================
% STEP 10: RETURN RESULTS
% =========================================================================
results = struct();
results.metrics = metrics;
results.weights = W;
results.predictions = y_pred;
results.ground_truth = y_test;
results.train_idx = train_idx;
results.test_idx = test_idx;
results.time = time_vec;
results.sample_rate = sample_rate;
results.opts = opts;
results.norm_stats = norm_stats;
results.nMarkers = nMarkers;

end