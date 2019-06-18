rng('default')
rng(666)

[X1, X2, Y1, Y2] = deal(randn(50, 2) + 1,  ...
    randn(51, 2) - 1, ones(50, 1), -ones(51, 1));
figure; hold on; plot(X1(:,1),X1(:,2), 'ro');
plot(X2(:,1),X2(:,2), 'bo')
hold off;

line([-1.3, 1.4], [1.9, -1])

load iris.mat;

gam = 1; degrees = {[1 1], [1 2], [1 3], [1 4], [1 5], [1 6]};
params = cellfun(@(degree) ...
    {Xtrain, Ytrain, 'c', gam, degree, 'poly_kernel'}, ...
    degrees, 'UniformOutput', false);
polyModels = cellfun(@(x) trainlssvm(x), params, ...
    'UniformOutput',false);

polyModelSim = cellfun(@(model) simlssvm(...
    {model.xtrain, model.ytrain, model.type, model.gam, model.kernel_pars, model.kernel_type}, ...
    {model.alpha, model.b}, Xtest), polyModels,'UniformOutput', false);

polyPerf = cellfun(@(sim) ...
    [sum(sim ~= Ytest), sum(sim ~= Ytest)/length(Ytest)*100], polyModelSim, ...
    'UniformOutput', false);

plotterPoly = @(model) plotlssvm({model.xtrain, model.ytrain, model.type, ...
    model.gam, model.kernel_pars, model.kernel_type}, {
model.alpha, model.b});
plotterPoly(polyModels{5})

gam = 1; sigmas = [0.001, 0.005, 0.01, 0.1, 1, 2, 10, 30, 50, 100];
RBFParams = arrayfun(@(sigma) {Xtrain, Ytrain, 'c', gam, sigma,'RBF_kernel'}, ...
    sigmas, 'UniformOutput', false);
RBFModels = cellfun(@(param) trainlssvm(param), ...
    RBFParams, 'UniformOutput',false);
RBFModelSim = cellfun(@(model) simlssvm({model.xtrain, model.ytrain, model.type, ...
    model.gam, model.kernel_pars, model.kernel_type}, ...
    {model.alpha, model.b}, Xtest), RBFModels,'UniformOutput', false);
RBFPerf = cellfun(@(sim) [sum(sim ~= Ytest), sum(sim ~= Ytest)/length(Ytest)*100], ...
    RBFModelSim, 'UniformOutput', false);

sigma = 0.01; gams = [0.1, 1, 2, 5, 10, 30, 50, 100];
RBFParamsG = arrayfun(@(gamPar) {Xtrain, Ytrain, 'c', gamPar, sigma,'RBF_kernel'}, ...
    gams, 'UniformOutput', false);
RBFModelsG = cellfun(@(param) trainlssvm(param), ...
    RBFParamsG, 'UniformOutput', false);
RBFModelSimG = cellfun(@(model) simlssvm({model.xtrain, model.ytrain, model.type, ...
    model.gam, model.kernel_pars, model.kernel_type}, ...
    {model.alpha, model.b}, Xtest), RBFModelsG, 'UniformOutput', false);
RBFPerfG = cellfun(@(sim) [sum(sim ~= Ytest), ...
    sum(sim ~= Ytest)/length(Ytest)*100], RBFModelSimG, 'UniformOutput', false);

plotterRBF = @(model) plotlssvm({model.xtrain, model.ytrain, model.type, ...
    model.gam, model.kernel_pars, model.kernel_type}, {
model.alpha, model.b});
plotterRBF(RBFModelsG{4})

gams = 10e-3:10e2:10e3; sigmas = 10e-3:10e2:10e3;
[sigmasMesh, gamsMesh] = meshgrid(sigmas, gams);
parameterSpace = num2cell([gamsMesh(:), sigmasMesh(:)], 2);

randomPerfList = cellfun(@(cell) rsplitvalidate({ Xtrain , Ytrain , 'c', ...
    cell(:, 1), cell(:, 2), 'RBF_kernel'}, 0.80,'misclass'), parameterSpace);
plot(randomPerfList)

tenFoldPerfList = cellfun(@(cell) crossvalidate({ Xtrain , Ytrain , 'c', ...
    cell(:, 1), cell(:, 2), 'RBF_kernel'}, 10,'misclass'), parameterSpace);
plot(tenFoldPerfList)

leaveOnePerfList = cellfun(@(cell) leaveoneout({ Xtrain , Ytrain , 'c', ...
    cell(:, 1), cell(:, 2), 'RBF_kernel'}, 'misclass'), parameterSpace);
plot(leaveOnePerfList)

parameterSpace{20}

[gamSimp, sigSimp, costSimp] = tunelssvm({Xtrain, Ytrain, 'c', ...
    [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});

[gamBrute, sigBrute, costBrute] = tunelssvm({Xtrain, Ytrain, 'c', ...
    [], [], 'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm', {10, 'misclass'});

rng('default')
rng(666)

finModel = {Xtrain, Ytrain, 'c', gamSimp, sigSimp, 'RBF_kernel'};
finParams = trainlssvm(finModel);
[Yest, Ylatent] = simlssvm(finModel, {finParams.alpha, finParams.b}, Xtest);
roc(Ylatent, Ytest)

bay_modoutClass(finModel, 'figure'); colorbar;

bay_modoutClass(RBFModelsG{4}, 'figure'); colorbar;

brutModel = {Xtrain, Ytrain, 'c', gamBrute, sigBrute, 'RBF_kernel'};
figure;
bay_modoutClass(brutModel, 'figure'); colorbar;

ripleyDat = load('ripley.mat');

figure;
gscatter(ripleyDat.Xtrain(:, 1), ripleyDat.Xtrain(:, 2), ripleyDat.Ytrain)

[ripleyModel, ripleyYest, ripleyYlatent] = RBFModelBuilder(ripleyDat.Xtrain, ripleyDat.Ytrain, ripleyDat.Xtest);
plotlssvm({ripleyModel.xtrain, ripleyModel.ytrain, ripleyModel.type, ...
    ripleyModel.gam, ripleyModel.kernel_pars, ripleyModel.kernel_type}, {ripleyModel.alpha, ripleyModel.b});
roc(ripleyYlatent, ripleyDat.Ytest)

breastDat = load('breast.mat');
gscatter(breastDat.trainset(:, 1), breastDat.trainset(:, 2), breastDat.labels_train)

[breastModel, breastYest, breastYlatent] = RBFModelBuilder(breastDat.trainset, breastDat.labels_train, breastDat.testset);
roc(breastYlatent, breastDat.labels_test)

diabetesDat = load('diabetes.mat');
gscatter(diabetesDat.trainset(:, 1), diabetesDat.trainset(:, 2), diabetesDat.labels_train)

[diabetesModel, diabetesYest, diabetesYlatent] = RBFModelBuilder(diabetesDat.trainset, diabetesDat.labels_train, diabetesDat.testset);
roc(diabetesYlatent, diabetesDat.labels_test)

function [trainedModel, Yest, Ylatent] = RBFModelBuilder(Xtrain, Ytrain, Xtest)
[gamSimp, sigSimp, ~] = tunelssvm({Xtrain, Ytrain, 'c', ...
    [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
modelSpec= {Xtrain, Ytrain, 'c', gamSimp, sigSimp, 'RBF_kernel'};
trainedModel = trainlssvm(modelSpec);
[Yest, Ylatent] = simlssvm(modelSpec, {trainedModel.alpha, trainedModel.b}, Xtest);
end