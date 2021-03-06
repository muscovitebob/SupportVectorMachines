rng('default');
rng(333);

% sinc func regression

X = transpose(-3:0.01:3);
Y = sinc(X) + 0.1 * randn(length(X), 1);
Xtrain = X(1:2: end); Ytrain = Y(1:2: end);
Xtest = X(2:2: end); Ytest = Y(2:2: end);

gams = [10, 10e3, 10e6]; sigmas = [0.01, 1, 100];
[sigmasMesh, gamsMesh] = meshgrid(sigmas, gams);
parameterSpace = num2cell([gamsMesh(:), sigmasMesh(:)], 2);

perfList = cellfun(@(cell) crossvalidate({ Xtrain , Ytrain , 'f', ...
    cell(:, 1), cell(:, 2), 'RBF_kernel'}, 10, 'mse'), parameterSpace);
plot(perfList)

parameterSpace{5} 
parameterSpace{6}

niceParams = cellfun(@(index) parameterSpace{index}, {1 2 5 6}, 'UniformOutput', false);
niceModelSpecs = cellfun(@(cell) { Xtrain , Ytrain , 'f', ...
    cell(:, 1), cell(:, 2), 'RBF_kernel'}, niceParams, 'UniformOutput', false);
niceModels = cellfun(@(cell) trainlssvm(cell), niceModelSpecs, 'UniformOutput', false);
niceSpecAndModel = cellfun(@(index) {niceModelSpecs{index}, ...
    niceModels{index}}, {1 2 3 4}, 'UniformOutput', false);
[Yest, Zt] = cellfun(@(cell) simlssvm(cell{1}, ...
    {cell{2}.alpha, cell{2}.b}, Xtest), niceSpecAndModel, 'UniformOutput', false);
test_mses = cellfun(@(cell) immse(cell, Ytest), Yest);
plot(test_mses)

modelSpec= {Xtrain, Ytrain, 'f', parameterSpace{5}(1), parameterSpace{5}(2), 'RBF_kernel'};
model = trainlssvm(modelSpec);
Yest = simlssvm(modelSpec, {model.alpha, model.b}, Xtest)
figure;
plotlssvm(modelSpec, {model.alpha, model.b});
hold on;
scatter(Xtest, Ytest)
hold off;

[gamSimp, sigSimp, costSimp] = tunelssvm({Xtrain, Ytrain, 'f', ...
    [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {10, 'mse'});

[gamGrid, sigGrid, costGrid] = tunelssvm({Xtrain, Ytrain, 'f', ...
    [], [], 'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm', {10, 'mse'});

% bayesian 
sig = 0.4; gam = 10;
modelSpecBay = {Xtrain, Ytrain, 'f', gam, sig}
crit_Ls = arrayfun(@(level) bay_lssvm(modelSpecBay, level), [1 2 3])
bayOptims = arrayfun(@(level) bay_optimize(modelSpecBay, level), [1 2 3], 'UniformOutput', false);

sigErrs = bay_errorbar({Xtrain, Ytrain, 'f', bayOptims{2}.gam, bayOptims{3}.kernel_pars}, 'figure')

% robust ls-svm
dataGeneratingFun = @(X) sinc(X) + 0.1 * rand(size(X));
X = transpose(-6:0.2:6); Y = dataGeneratingFun(X);
outSet1 = [15 17 19]; outSet2 = [41 44 46];
Y(outSet1) = 0.7 * 0.3 + rand(size(outSet1)); Y(outSet2) = 1.5 * 0.2 + rand(size(outSet2)); 
scatter(X, Y)

Model = initlssvm(X, Y, 'f', [], [],'RBF_kernel');
tunedNaiveModel = tunelssvm(Model, 'simplex', ...
    'crossvalidatelssvm', {10, 'mse'});
plotlssvm(tunedNaiveModel)

% huber
model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
loss = 'whuber';
modelLoss1 = tunelssvm(model, 'simplex', 'rcrossvalidatelssvm', {10, 'mae'}, loss);
modelLoss1Tr = robustlssvm(modelLoss1);
plotlssvm(modelLoss1Tr)

% hampel

model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
loss = 'whampel';
modelLoss1 = tunelssvm(model, 'simplex', 'rcrossvalidatelssvm', {10, 'mae'}, loss);
modelLoss1Tr = robustlssvm(modelLoss1);
plotlssvm(modelLoss1Tr)

% logistic

model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
loss = 'wlogistic';
modelLoss1 = tunelssvm(model, 'simplex', 'rcrossvalidatelssvm', {10, 'mae'}, loss);
modelLoss1Tr = robustlssvm(modelLoss1);
plotlssvm(modelLoss1Tr)

% myriad

model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
loss = 'wmyriad';
modelLoss1 = tunelssvm(model, 'simplex', 'rcrossvalidatelssvm', {10, 'mae'}, loss);
modelLoss1Tr = robustlssvm(modelLoss1);
plotlssvm(modelLoss1Tr)

% time series prediction
% logmap

load logmap.mat

order = 10;
Xinit = windowize(Z, 1:(order+1));
Y = Xinit(:, end);
X = Xinit(:, 1:order);

gam = 10; sig = 10;
timeSeriesModel1 = trainlssvm({X, Y, 'f', gam, sig});

Xs = Z(end - order + 1:end, 1);

nb = length(Ztest);
prediction = predict({X, Y, 'f', gam, sig}, Xs , nb);

figure;
hold on;
plot(Ztest , 'k');
plot(prediction, 'r');
hold off;

modelSpecBayLog = {X, Y, 'f', gam, sig}
crit_Ls = arrayfun(@(level) bay_lssvm(modelSpecBayLog, level), [1 2 3])
bayOptims = arrayfun(@(level) bay_optimize(modelSpecBayLog, level), [1 2 3], 'UniformOutput', false);
prediction2 = predict(bayOptims{3}, Xs, nb)

gam = 2.4; sig = 24.2154;

figure;
hold on;
plot(Ztest , 'k');
plot(prediction2, 'r');
hold off;

[gamSimp, sigSimp, costSimp] = tunelssvm({X, Y, 'f', ...
    [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {10, 'mse'});

prediction3 = predict({X, Y, 'f', gamSimp, sigSimp}, Xs , nb);
figure;
hold on;
plot(Ztest , 'k');
plot(prediction3, 'r');
hold off;

% grid search over the orders

orderCell = {5 10 20 30 40 50 60 70 80 90 100};
windowizedData = cellfun(@(order) {order, windowize(Z, 1:(order+1))}, orderCell, 'UniformOutput', false);
Yes = cellfun(@(tuple) tuple{2}(:, end), windowizedData, 'UniformOutput', false);
Xes = cellfun(@(tuple) tuple{2}(:, 1:tuple{1}), windowizedData, 'UniformOutput', false);
Xses = cellfun(@(order) Z(end - order + 1:end, 1), orderCell, 'UniformOutput', false);

YandXandXses = arrayfun(@(index) {Yes{index} Xes{index} Xses{index}}, 1:length(orderCell), 'UniformOutput', false);

predictions = cellfun(@(cell) predict({cell{2} , cell{1} , 'f', ...
    gam, sig}, cell{3} ,nb), YandXandXses, 'UniformOutput', false);

hold on;
plot(Ztest , 'k');
plot(predictions{8}, 'r');
hold off;

% santa fe

load santafe.mat;

order = 10;
Xinit = windowize(Z, 1:(order+1));
Y = Xinit(:, end);
X = Xinit(:, 1:order);

gam = 10; sig = 10;
timeSeriesModel1 = trainlssvm({X, Y, 'f', gam, sig});

Xs = Z(end - order + 1:end, 1);

nb = length(Ztest);
prediction = predict({X, Y, 'f', gam, sig}, Xs , nb);

figure;
hold on;
plot(Ztest , 'k');
plot(prediction, 'r');
hold off;

modelSpecBayLog = {X, Y, 'f', gam, sig}
crit_Ls = arrayfun(@(level) bay_lssvm(modelSpecBayLog, level), [1 2 3])
bayOptims = arrayfun(@(level) bay_optimize(modelSpecBayLog, level), [1 2 3], 'UniformOutput', false);
prediction2 = predict(bayOptims{3}, Xs, nb)

gam = 62.47; sig = 9.25;

orderCell = {5 10 20 30 40 50 60 70 80 90 100};
windowizedData = cellfun(@(order) {order, windowize(Z, 1:(order+1))}, orderCell, 'UniformOutput', false);
Yes = cellfun(@(tuple) tuple{2}(:, end), windowizedData, 'UniformOutput', false);
Xes = cellfun(@(tuple) tuple{2}(:, 1:tuple{1}), windowizedData, 'UniformOutput', false);
Xses = cellfun(@(order) Z(end - order + 1:end, 1), orderCell, 'UniformOutput', false);

YandXandXses = arrayfun(@(index) {Yes{index} Xes{index} Xses{index}}, 1:length(orderCell), 'UniformOutput', false);

predictions = cellfun(@(cell) predict({cell{2} , cell{1} , 'f', ...
    gam, sig}, cell{3} ,nb), YandXandXses, 'UniformOutput', false);

hold on;
plot(Ztest , 'k');
plot(predictions{9}, 'r');
hold off;