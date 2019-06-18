rng('default');
rng(333);

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

sig = 0.4; gam = 10;
modelSpecBay = {Xtrain, Ytrain, 'f', gam, sig}
crit_Ls = arrayfun(@(level) bay_lssvm(modelSpecBay, level), [1 2 3])
bayOptims = arrayfun(@(level) bay_optimize(modelSpecBay, level), [1 2 3], 'UniformOutput', false);

sigErrs = bay_errorbar({Xtrain, Ytrain, 'f', bayOptims{2}.gam, bayOptims{3}.kernel_pars}, 'figure')

dataGeneratingFun = @(X) sinc(X) + 0.1 * rand(size(X));
X = transpose(-6:0.2:6); Y = dataGeneratingFun(X);
outSet1 = [15 17 19]; outSet2 = [41 44 46];
Y(outSet1) = 0.7 * 0.3 + rand(size(outSet1)); Y(outSet2) = 1.5 * 0.2 + rand(size(outSet2)); 
scatter(X, Y)

Model = initlssvm(X, Y, 'f', [], [],'RBF_kernel');
tunedNaiveModel = tunelssvm(Model, 'simplex', ...
    'crossvalidatelssvm', {10, 'mse'});
plotlssvm(tunedNaiveModel)

