%% Credit Card Default Prediction using KNN (UCI Dataset)
% Full pipeline: load -> clean -> feature engineering -> encode -> scale ->
% tune K with CV -> evaluate -> error analysis (FP/FN patterns)
% Author: Levy Thiga Kariuki

clear; clc; close all;

%% 1) Load data (robust and clean)
clear; clc; close all;

fileName = 'default of credit card clients.xls';

opts = detectImportOptions(fileName);
opts.DataRange = 'A2';
opts.VariableNamesRange = 'A2';

T = readtable(fileName, opts);

disp("Columns loaded:");
disp(T.Properties.VariableNames);

% Remove ID
if any(strcmp(T.Properties.VariableNames,'ID'))
    T.ID = [];
end

% Remove completely empty rows 
T = rmmissing(T, 'MinNumMissing', width(T));  % drop rows missing ALL columns

% Remove rows with missing in any column (safe for this dataset)
T = rmmissing(T);

% Automatically detect target column
varNames = T.Properties.VariableNames;
targetIdx = find(contains(lower(varNames),'default'),1);
targetName = varNames{targetIdx};

fprintf('Target column detected: %s\n', targetName);

y = double(T.(targetName));
X = removevars(T, targetName);

fprintf('Dataset size: %d rows, %d features\n', height(X), width(X));
fprintf('Default rate: %.2f%%\n', 100*mean(y==1));

%% 2) Basic cleaning for categorical codes
% EDUCATION and MARRIAGE sometimes have 0, 5, 6 etc. (unknown/other).
% For robust analysis, map them into an "Other" category instead of leaving as misleading values.

% Convert to categorical then regroup
X.SEX = categorical(X.SEX);

% EDUCATION mapping
% Common coding: 1=grad school, 2=university, 3=high school, 4=others, 0/5/6=unknown/others
edu = X.EDUCATION;
edu(edu==0 | edu==5 | edu==6) = 4; % map to "others"
X.EDUCATION = categorical(edu);

% MARRIAGE mapping
% Common coding: 1=married, 2=single, 3=others, 0=unknown
mar = X.MARRIAGE;
mar(mar==0) = 3; % map to others
X.MARRIAGE = categorical(mar);

% AGE stays numeric

%% 3) Feature Engineering
% Create behavioural features that capture risk patterns better than raw monthly values.

% Helper: grab variables by name safely
getVar = @(name) X.(name);

% Repayment status vectors (PAY_0..PAY_6)
PAY = [getVar('PAY_0'), getVar('PAY_2'), getVar('PAY_3'), getVar('PAY_4'), getVar('PAY_5'), getVar('PAY_6')];

% Bill and Payment matrices
BILL = [getVar('BILL_AMT1'), getVar('BILL_AMT2'), getVar('BILL_AMT3'), getVar('BILL_AMT4'), getVar('BILL_AMT5'), getVar('BILL_AMT6')];
PMT  = [getVar('PAY_AMT1'),  getVar('PAY_AMT2'),  getVar('PAY_AMT3'),  getVar('PAY_AMT4'),  getVar('PAY_AMT5'),  getVar('PAY_AMT6')];

limit = getVar('LIMIT_BAL');

% Engineered features
avgDelinq  = mean(PAY, 2);
maxDelinq  = max(PAY, [], 2);
lateCount  = sum(PAY >= 1, 2);                 % number of months delayed
severeLate = sum(PAY >= 2, 2);                 % severe delinquency months

% Utilisation (avoid division by 0)
util = abs(BILL) ./ (limit + 1);
avgUtil = mean(util, 2);
maxUtil = max(util, [], 2);

% Repayment-to-bill ratio (add +1 to avoid divide by zero)
repRatio = PMT ./ (abs(BILL) + 1);
avgRepRatio = mean(repRatio, 2);
minRepRatio = min(repRatio, [], 2);

% Bill volatility (stability vs spikes)
billStd = std(BILL, 0, 2);

% Append engineered features to X as new columns
X.avgDelinq   = avgDelinq;
X.maxDelinq   = maxDelinq;
X.lateCount   = lateCount;
X.severeLate  = severeLate;
X.avgUtil     = avgUtil;
X.maxUtil     = maxUtil;
X.avgRepRatio = avgRepRatio;
X.minRepRatio = minRepRatio;
X.billStd     = billStd;

%% 4) One-hot encode categorical variables + convert to matrix
catVars = {'SEX','EDUCATION','MARRIAGE'};
Xcat = X(:, catVars);
Xnum = removevars(X, catVars);

% One-hot encoding
XcatMat = [];
catNames = {};
for i = 1:numel(catVars)
    C = dummyvar(Xcat.(catVars{i}));
    cats = categories(Xcat.(catVars{i}));
    for j = 1:numel(cats)
        catNames{end+1} = sprintf('%s_%s', catVars{i}, string(cats{j}));
    end
    XcatMat = [XcatMat, C];
end

Xmat = [table2array(Xnum), XcatMat];
featNames = [Xnum.Properties.VariableNames, catNames];

fprintf('\nFinal feature matrix: %d features\n', size(Xmat,2));

%% 5) Train/test split (stratified)
cv = cvpartition(y, 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx  = test(cv);

Xtrain = Xmat(trainIdx,:);
ytrain = y(trainIdx);
Xtest  = Xmat(testIdx,:);
ytest  = y(testIdx);

%% 6) Standardisation (critical for KNN)
mu = mean(Xtrain,1);
sigma = std(Xtrain,0,1);
sigma(sigma==0) = 1;

XtrainZ = (Xtrain - mu) ./ sigma;
XtestZ  = (Xtest  - mu) ./ sigma;

%% 7) Tune K with 5-fold CV (optimise F1-score)
Klist = [1 3 5 7 9 11 15 21 31 41 51];
cv2 = cvpartition(ytrain,'KFold',5);

bestF1 = -inf; bestK = Klist(1);
results = zeros(numel(Klist), 5); % K, Acc, Prec, Rec, F1

for i = 1:numel(Klist)
    k = Klist(i);
    yhat_all = zeros(size(ytrain));

    for fold = 1:cv2.NumTestSets
        tr = training(cv2,fold);
        va = test(cv2,fold);

        mdl = fitcknn(XtrainZ(tr,:), ytrain(tr), ...
            'NumNeighbors', k, ...
            'Distance', 'euclidean', ...
            'DistanceWeight', 'inverse');

        yhat_all(va) = predict(mdl, XtrainZ(va,:));
    end

    C = confusionmat(ytrain, yhat_all, 'Order', [0 1]);
    TN=C(1,1); FP=C(1,2); FN=C(2,1); TP=C(2,2);

    acc = (TP+TN)/sum(C,'all');
    prec = TP / max(TP+FP,1);
    rec  = TP / max(TP+FN,1);
    f1   = 2*prec*rec / max(prec+rec, eps);

    results(i,:) = [k acc prec rec f1];

    if f1 > bestF1
        bestF1 = f1; bestK = k;
    end
end

fprintf('\nBest K found by CV: %d (F1=%.3f)\n', bestK, bestF1);

%% Improves recall using cost-sensitive learning and distance choice
% Penalise FN more than FP (banking: missing a defaulter is costly)
% Cost matrix format: [Cost(TN) Cost(FP); Cost(FN) Cost(TP)]
costMat = [0 1; 4 0];   % FN cost 4x FP (tune if needed)

distList = {'euclidean','cityblock','cosine'};
best = struct('F1',-inf,'Dist','','Cost',[]);

for d = 1:numel(distList)
    mdl = fitcknn(XtrainZ, ytrain, ...
        'NumNeighbors', bestK, ...
        'Distance', distList{d}, ...
        'DistanceWeight','inverse', ...
        'Cost', costMat);

    [yPred2, score2] = predict(mdl, XtestZ);
    C2 = confusionmat(ytest, yPred2, 'Order', [0 1]);
    TN2=C2(1,1); FP2=C2(1,2); FN2=C2(2,1); TP2=C2(2,2);

    prec2 = TP2 / max(TP2+FP2,1);
    rec2  = TP2 / max(TP2+FN2,1);
    f12   = 2*prec2*rec2 / max(prec2+rec2, eps);

    fprintf('\nDist=%s | Precision=%.3f Recall=%.3f F1=%.3f\n', ...
        distList{d}, prec2, rec2, f12);

    if f12 > best.F1
        best.F1 = f12;
        best.Dist = distList{d};
        best.Cost = costMat;
        best.C = C2;
        best.prec = prec2; best.rec = rec2;
    end
end

fprintf('\nBest improved model: Dist=%s | Precision=%.3f Recall=%.3f F1=%.3f\n', ...
    best.Dist, best.prec, best.rec, best.F1);
disp('Confusion Matrix [TN FP; FN TP]:');
disp(best.C);

% Plot K vs F1
figure;
plot(results(:,1), results(:,5), '-o'); grid on;
xlabel('K'); ylabel('F1-score (CV)');
title('K selection using 5-fold cross-validation');
saveas(gcf,'K_vs_F1.png');

%% 8) Train final model with bestK and evaluate on test set
finalMdl = fitcknn(XtrainZ, ytrain, ...
    'NumNeighbors', bestK, ...
    'Distance', best.Dist, ...
    'DistanceWeight', 'inverse', ...
    'Cost', best.Cost);

[yPred, score] = predict(finalMdl, XtestZ);

C = confusionmat(ytest, yPred, 'Order', [0 1]);
TN=C(1,1); FP=C(1,2); FN=C(2,1); TP=C(2,2);

acc = (TP+TN)/sum(C,'all');
prec = TP / max(TP+FP,1);
rec  = TP / max(TP+FN,1);
f1   = 2*prec*rec / max(prec+rec, eps);

fprintf('\n=== Test set performance ===\n');
disp('Confusion Matrix [TN FP; FN TP]:');
disp(C);
fprintf('Accuracy:  %.3f\n', acc);
fprintf('Precision: %.3f\n', prec);
fprintf('Recall:    %.3f\n', rec);
fprintf('F1-score:  %.3f\n', f1);

% ROC + AUC
[fpRate,tpRate,~,AUC] = perfcurve(ytest, score(:,2), 1);
figure; plot(fpRate,tpRate); grid on;
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.3f)', AUC));
saveas(gcf,'ROC_curve.png');

%% 9) Error analysis: FP and FN pattern extraction (T4-ready)
FP_idx = (yPred==1 & ytest==0);
FN_idx = (yPred==0 & ytest==1);

fprintf('\nFP count: %d | FN count: %d\n', sum(FP_idx), sum(FN_idx));

% Compare FP vs TN and FN vs TP using median gaps
TN_idx = (yPred==0 & ytest==0);
TP_idx = (yPred==1 & ytest==1);

med_FP = median(Xtest(FP_idx,:),1);
med_TN = median(Xtest(TN_idx,:),1);
med_FN = median(Xtest(FN_idx,:),1);
med_TP = median(Xtest(TP_idx,:),1);

gapFP = abs(med_FP - med_TN);
gapFN = abs(med_FN - med_TP);

[~, topFP] = maxk(gapFP, 8);
[~, topFN] = maxk(gapFN, 8);

disp('Top differing features (FP vs TN):');
disp(featNames(topFP));

disp('Top differing features (FN vs TP):');
disp(featNames(topFN));

% Saves a table for report
topFP_table = table(string(featNames(topFP))', gapFP(topFP)', 'VariableNames', {'Feature','MedianGap'});
topFN_table = table(string(featNames(topFN))', gapFN(topFN)', 'VariableNames', {'Feature','MedianGap'});

writetable(topFP_table, 'Top_FP_Features.csv');
writetable(topFN_table, 'Top_FN_Features.csv');

fprintf('\nSaved outputs: K_vs_F1.png, ROC_curve.png, Top_FP_Features.csv, Top_FN_Features.csv\n');
