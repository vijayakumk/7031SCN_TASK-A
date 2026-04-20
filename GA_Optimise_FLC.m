clear; clc; close all;

% =========================
% LOAD FIS
% =========================
fis = readfis('AsthmaSeverityFLC.fis');

% =========================
% TEST DATA
% 10 patients with 4 inputs each
% [PeakFlowRate, SymptomFrequency, InhalerUse, ExerciseTolerance]
% =========================
X = [
    620,  1,  1,  9;    % Patient 1  — clearly Intermittent
    580,  1,  2,  8;    % Patient 2  — Intermittent
    450,  3,  3,  6;    % Patient 3  — Mild Persistent
    400,  3,  4,  5;    % Patient 4  — Mild Persistent
    350,  4,  5,  5;    % Patient 5  — Mild-Moderate border
    300,  5,  7,  4;    % Patient 6  — Moderate Persistent
    280,  5,  8,  3;    % Patient 7  — Moderate Persistent
    200,  6, 10,  2;    % Patient 8  — Moderate-Severe border
    150,  7, 12,  2;    % Patient 9  — Severe Persistent
    100,  7, 13,  1;    % Patient 10 — clearly Severe Persistent
];

% Target severity scores based on GINA classification
% Intermittent=15, Mild Persistent=37, Moderate Persistent=62, Severe=85
Y = [
    15;    % Patient 1
    18;    % Patient 2
    35;    % Patient 3
    38;    % Patient 4
    50;    % Patient 5
    60;    % Patient 6
    63;    % Patient 7
    72;    % Patient 8
    82;    % Patient 9
    88;    % Patient 10
];

% =========================
% GA SETUP
% Optimising 6 key membership function centres
% across inputs and output
%
% x(1) — PeakFlowRate: centre of Moderate MF
% x(2) — PeakFlowRate: start of High MF plateau
% x(3) — SymptomFrequency: centre of Occasional MF
% x(4) — InhalerUse: centre of Moderate MF
% x(5) — AsthmaSeverity: centre of Mild_Persistent MF
% x(6) — AsthmaSeverity: centre of Moderate_Persistent MF
% =========================
nvars = 6;

lb = [300  450  2   5   28  50];   % lower bounds for each gene
ub = [500  650  4  10   48  75];   % upper bounds for each gene

opts = optimoptions('ga', ...
    'PopulationSize', 50, ...
    'MaxGenerations', 100, ...
    'EliteCount',     5, ...
    'CrossoverFraction', 0.8, ...
    'Display',        'iter', ...
    'PlotFcn',        @gaplotbestf);

% =========================
% RUN GA
% =========================
fprintf('Starting GA optimisation...\n');
fprintf('Population: 50 | Generations: 100\n\n');

obj = @(x) localFitness(x, fis, X, Y);
[xbest, fbest] = ga(obj, nvars, [], [], [], [], lb, ub, [], opts);

disp('===== BEST PARAMETERS FOUND BY GA =====')
fprintf('PeakFlowRate Moderate centre : %.2f\n', xbest(1));
fprintf('PeakFlowRate High start      : %.2f\n', xbest(2));
fprintf('SymptomFreq  Occasional centre: %.2f\n', xbest(3));
fprintf('InhalerUse   Moderate centre  : %.2f\n', xbest(4));
fprintf('Severity     Mild centre       : %.2f\n', xbest(5));
fprintf('Severity     Moderate centre   : %.2f\n', xbest(6));

disp('===== BEST FITNESS (MSE) =====')
disp(fbest)

% =========================
% BUILD OPTIMISED FIS
% =========================
optFis = fis;

% Input 1: PeakFlowRate — update Moderate (trimf) and High (trapmf)
% Moderate trimf: [a centre c] — keep spread, shift centre
optFis.Inputs(1).MembershipFunctions(2).Parameters = ...
    [xbest(1)-150, xbest(1), xbest(1)+150];

% High trapmf: [a b c d] — shift where plateau starts
optFis.Inputs(1).MembershipFunctions(3).Parameters = ...
    [xbest(2)-100, xbest(2), 700, 700];

% Input 2: SymptomFrequency — update Occasional (trimf)
optFis.Inputs(2).MembershipFunctions(2).Parameters = ...
    [xbest(3)-2, xbest(3), xbest(3)+2];

% Input 3: InhalerUse — update Moderate (trimf)
optFis.Inputs(3).MembershipFunctions(2).Parameters = ...
    [xbest(4)-4, xbest(4), xbest(4)+4];

% Output: AsthmaSeverity — update Mild_Persistent and Moderate_Persistent
optFis.Outputs(1).MembershipFunctions(2).Parameters = ...
    [xbest(5)-15, xbest(5), xbest(5)+15];

optFis.Outputs(1).MembershipFunctions(3).Parameters = ...
    [xbest(6)-15, xbest(6), xbest(6)+15];

% =========================
% BEFORE VS AFTER EVALUATION
% =========================
n = size(X, 1);
before = zeros(n, 1);
after  = zeros(n, 1);

for i = 1:n
    before(i) = evalfis(fis,    X(i,:));
    after(i)  = evalfis(optFis, X(i,:));
end

% =========================
% RESULT TABLE
% =========================
result_table = table(...
    X(:,1), X(:,2), X(:,3), X(:,4), ...
    before, after, Y, ...
    abs(before - Y), abs(after - Y), ...
    'VariableNames', { ...
        'PeakFlow', 'SymptomFreq', 'InhalerUse', 'ExerciseTol', ...
        'Before_GA', 'After_GA', 'Target', ...
        'Error_Before', 'Error_After'});

disp('===== FULL RESULT TABLE =====')
disp(result_table)

% =========================
% ERROR METRICS
% =========================
mse_before = mean((before - Y).^2);
mse_after  = mean((after  - Y).^2);
mae_before = mean(abs(before - Y));
mae_after  = mean(abs(after  - Y));
improvement = ((mae_before - mae_after) / mae_before) * 100;

fprintf('\n===== ERROR METRICS =====\n');
fprintf('MSE Before GA : %.4f\n', mse_before);
fprintf('MSE After  GA : %.4f\n', mse_after);
fprintf('MAE Before GA : %.4f\n', mae_before);
fprintf('MAE After  GA : %.4f\n', mae_after);
fprintf('Improvement   : %.2f%%\n', improvement);

% =========================
% PLOT 1: Target vs Before vs After
% =========================
figure;
plot(1:n, Y,      '-o', 'LineWidth', 2, 'DisplayName', 'Target');
hold on;
plot(1:n, before, '-s', 'LineWidth', 2, 'DisplayName', 'Before GA');
plot(1:n, after,  '-d', 'LineWidth', 2, 'DisplayName', 'After GA');
hold off;
xlabel('Patient Number');
ylabel('Asthma Severity Score (0-100)');
legend('Location', 'best');
title('Asthma FLC Output: Before vs After GA Optimisation');
xticks(1:n);
xticklabels({'P1','P2','P3','P4','P5','P6','P7','P8','P9','P10'});
grid on;
saveas(gcf, 'Plot1_Target_vs_Predicted.png');

% =========================
% PLOT 2: Error bar chart before vs after
% =========================
figure;
bar([abs(before-Y), abs(after-Y)]);
legend('Error Before GA', 'Error After GA', 'Location', 'northeast');
xlabel('Patient Number');
ylabel('Absolute Error');
title('Per-Patient Error: Before vs After GA Optimisation');
xticks(1:n);
xticklabels({'P1','P2','P3','P4','P5','P6','P7','P8','P9','P10'});
grid on;
saveas(gcf, 'Plot2_Error_Comparison.png');

% =========================
% PLOT 3: All 6 surface plots — BEFORE GA
% =========================
inputNames = {'Peak Flow Rate (L/min)', ...
              'Symptom Frequency (days/week)', ...
              'Inhaler Use (puffs/week)', ...
              'Exercise Tolerance (0-10)'};

combinations = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];
comboTitles  = {
    'PeakFlowRate vs SymptomFrequency';
    'PeakFlowRate vs InhalerUse';
    'PeakFlowRate vs ExerciseTolerance';
    'SymptomFrequency vs InhalerUse';
    'SymptomFrequency vs ExerciseTolerance';
    'InhalerUse vs ExerciseTolerance'
};

for i = 1:6
    figure;
    gensurf(fis, combinations(i,:));
    title(['Before GA — ' comboTitles{i}], 'FontSize', 11);
    xlabel(inputNames{combinations(i,1)}, 'FontSize', 9);
    ylabel(inputNames{combinations(i,2)}, 'FontSize', 9);
    zlabel('Asthma Severity (0-100)', 'FontSize', 9);
    colormap jet; colorbar; grid on; view(45, 30);
    saveas(gcf, ['Plot3_Before_Surface_' num2str(i) '.png']);
end

% =========================
% PLOT 4: All 6 surface plots — AFTER GA
% =========================
for i = 1:6
    figure;
    gensurf(optFis, combinations(i,:));
    title(['After GA — ' comboTitles{i}], 'FontSize', 11);
    xlabel(inputNames{combinations(i,1)}, 'FontSize', 9);
    ylabel(inputNames{combinations(i,2)}, 'FontSize', 9);
    zlabel('Asthma Severity (0-100)', 'FontSize', 9);
    colormap jet; colorbar; grid on; view(45, 30);
    saveas(gcf, ['Plot4_After_Surface_' num2str(i) '.png']);
end

% =========================
% PLOT 5: Membership functions Before vs After
% — for each input and the output
% =========================
varLabels = {'input', 'input', 'input', 'input', 'output'};
varIndex  = {1, 2, 3, 4, 1};
varNames  = {'PeakFlowRate', 'SymptomFrequency', ...
             'InhalerUse', 'ExerciseTolerance', 'AsthmaSeverity'};

for i = 1:5
    % Before
    figure;
    plotmf(fis, varLabels{i}, varIndex{i});
    title(['Before GA: ' varNames{i} ' Membership Functions']);
    saveas(gcf, ['Plot5_Before_MF_' varNames{i} '.png']);

    % After
    figure;
    plotmf(optFis, varLabels{i}, varIndex{i});
    title(['After GA: ' varNames{i} ' Membership Functions']);
    saveas(gcf, ['Plot5_After_MF_' varNames{i} '.png']);
end

% =========================
% PLOT 6: Side-by-side surface comparison
% for the most important combination (PeakFlow vs Symptoms)
% =========================
figure;
subplot(1,2,1);
gensurf(fis, [1 2]);
title('Before GA: PeakFlowRate vs SymptomFrequency');
xlabel('Peak Flow Rate'); ylabel('Symptom Frequency');
zlabel('Severity'); colormap jet; view(45,30);

subplot(1,2,2);
gensurf(optFis, [1 2]);
title('After GA: PeakFlowRate vs SymptomFrequency');
xlabel('Peak Flow Rate'); ylabel('Symptom Frequency');
zlabel('Severity'); colormap jet; view(45,30);

saveas(gcf, 'Plot6_Surface_SideBySide.png');

% =========================
% RULE VIEWER — optimised FLC
% =========================
ruleview(optFis);

% =========================
% SAVE OPTIMISED FIS
% =========================
writeFIS(optFis, 'AsthmaSeverityFLC_Optimised.fis');

fprintf('\n===== ALL DONE =====\n');
fprintf('Saved files:\n');
fprintf('  AsthmaSeverityFLC_Optimised.fis\n');
fprintf('  Plot1_Target_vs_Predicted.png\n');
fprintf('  Plot2_Error_Comparison.png\n');
fprintf('  Plot3_Before_Surface_1 to 6.png\n');
fprintf('  Plot4_After_Surface_1 to 6.png\n');
fprintf('  Plot5_Before/After_MF per variable.png\n');
fprintf('  Plot6_Surface_SideBySide.png\n');

% ==========================================================
% LOCAL FITNESS FUNCTION
% ==========================================================
function err = localFitness(x, fis, X, Y)

    f = fis;

    % Update PeakFlowRate Moderate (trimf)
    f.Inputs(1).MembershipFunctions(2).Parameters = ...
        [x(1)-150, x(1), x(1)+150];

    % Update PeakFlowRate High (trapmf)
    f.Inputs(1).MembershipFunctions(3).Parameters = ...
        [x(2)-100, x(2), 700, 700];

    % Update SymptomFrequency Occasional (trimf)
    f.Inputs(2).MembershipFunctions(2).Parameters = ...
        [x(3)-2, x(3), x(3)+2];

    % Update InhalerUse Moderate (trimf)
    f.Inputs(3).MembershipFunctions(2).Parameters = ...
        [x(4)-4, x(4), x(4)+4];

    % Update output Mild_Persistent (trimf)
    f.Outputs(1).MembershipFunctions(2).Parameters = ...
        [x(5)-15, x(5), x(5)+15];

    % Update output Moderate_Persistent (trimf)
    f.Outputs(1).MembershipFunctions(3).Parameters = ...
        [x(6)-15, x(6), x(6)+15];

    % Evaluate all patients
    n    = size(X, 1);
    pred = zeros(n, 1);

    for k = 1:n
        try
            pred(k) = evalfis(f, X(k,:));
        catch
            pred(k) = 999;
        end
    end

    % Mean Squared Error as fitness score
    err = mean((pred - Y).^2);
end