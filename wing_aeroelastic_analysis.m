% Clear all
clear; clc; close all;

%% Defining initial parameters
c_1 = 0.3;
c_2 = 0.4;
c_r = c_1 + c_2;
c_t = 1.75*c_1;
c_mean = (c_r + c_t)/2;
s = 2;
rho = 1.225;

% Lift distribution function
cl_alpha = @(eta) 2*pi .* sqrt(1 - eta.^2);
% Chord function
c = @(eta) c_r - (c_r - c_t).*eta;
% ec function
ec = @(eta) c(eta) - c_1 - 0.25.*c(eta);

% GJ constants
GJ_1 = 8500;
GJ_2 = 7500;
GJ = @(eta) (GJ_1 * (1 - 0.25.*eta)) .* (eta < 0.5) + ...
            (GJ_2 * (1 - 0.2.*eta)) .* (eta >= 0.5);

%% Part 1: Divergence dynamic pressure convergence study
n_max = 10;

% Cell arrays for mode functions and derivatives
f = cell(n_max, 1);
f_prime = cell(n_max, 1);

% Matrices to store results
q_div = zeros(1, n_max);
error_pct = zeros(1, n_max);

% Initialize convergence flag
converged = false;
n_converged = n_max;  % Initialize to n_max

% Create mode functions and derivatives
for i = 1:n_max
    f{i} = @(eta) i.*eta.^i;              % f_i = i*η^i
    f_prime{i} = @(eta) i.^2 .* eta.^(i-1); % f'_i = i^2*η^(i-1)
end

fprintf('========================================\n');
fprintf('CONVERGENCE STUDY\n');
fprintf('========================================\n');
fprintf('Mode\t\tq_d [N/m²]\t\tError [%%]\n');
fprintf('----\t\t----------\t\t----------\n');

% Try each mode function until convergence
for n = 1:n_max
    % Initialize stiffness matrices (n×n)
    E = zeros(n, n);
    K = zeros(n, n);
    
    % Build E and K matrices
    for i = 1:n
        for j = 1:n

            % E matrix: 
            integrand_E1 = @(eta) (GJ_1 * (1 - 0.25*eta)) .* ...
                                  f_prime{i}(eta) .* f_prime{j}(eta);
            E_part1 = integral(integrand_E1, 0, 0.5);
            
            integrand_E2 = @(eta) (GJ_2 * (1 - 0.2*eta)) .* ...
                                  f_prime{i}(eta) .* f_prime{j}(eta);
            E_part2 = integral(integrand_E2, 0.5, 1.0);
            
            E(i,j) = (s) * (E_part1 + E_part2);
            
            % K matrix: 
            integrand_K = @(eta) c(eta) .* ec(eta) .* cl_alpha(eta) .* ...
                                 f{i}(eta) .* f{j}(eta);
            K(i,j) = (-s) * integral(integrand_K, 0, 1);
        end
    end
    
    % Solve [E] + q[K] = 0
    eigenvalues = eig(E, -K);
    eigenvalues = sort(eigenvalues);
    q_div(n) = eigenvalues(1);
    
    % Display results
    if n == 1
        fprintf('%d\t\t%.2f\t\t--\n', n, q_div(n));
    else
        error_pct(n) = abs((q_div(n) - q_div(n-1)) / q_div(n)) * 100;
        fprintf('%d\t\t%.2f\t\t%.4f\n', n, q_div(n), error_pct(n));
        
        % Check convergence
        if error_pct(n) < 0.1
            fprintf('\n✓ CONVERGED at n=%d modes (error < 0.1%%)\n\n', n);
            converged = true;
            n_converged = n;
            break;
        end
    end
end

% Final results
fprintf('========================================\n');
fprintf('FINAL RESULTS\n');
fprintf('========================================\n');

if converged
    q_d_final = q_div(n_converged);
    fprintf('✓ Converged at: n = %d modes\n', n_converged);
    fprintf('Divergence Dynamic Pressure: q_d = %.2f N/m²\n', q_d_final);
    fprintf('Divergence Speed: U_d = %.2f m/s\n', sqrt(2*abs(q_d_final)/rho));
else
    fprintf('✗ Did NOT converge within %d modes\n', n_max);
    fprintf('Using last value: q_d = %.2f N/m² (n=%d)\n', q_div(n_max), n_max);
    q_d_final = q_div(n_max);
    n_converged = n_max;
end

% Plot convergence
figure('Position', [100, 100, 800, 500]);
plot(1:n_converged, q_div(1:n_converged), 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
xlabel('Number of Modes', 'FontSize', 12);
ylabel('Divergence Dynamic Pressure q_d [N/m²]', 'FontSize', 12);
title('Convergence of q_d vs Number of Modes', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

if converged
    hold on;
    plot(n_converged, q_d_final, 'ro', 'MarkerSize', 15, 'LineWidth', 3);
    legend('q_d', sprintf('Converged (n=%d)', n_converged), 'Location', 'best');
else
    title('q_d vs Number of Modes (NOT CONVERGED)', 'FontSize', 14, 'FontWeight', 'bold');
end


