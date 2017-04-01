function test(testc)

load PA8SampleCases
load PA8Data 

%% 1) FitGaussianParameters [ FitGaussianParameters.m ]
switch testc
  case 1
 X = exampleINPUT.t1a1; 
 [mu sigma] = FitGaussianParameters(X); 
 assert(mu,exampleOUTPUT.t1o1,1e-6);
 assert(sigma,exampleOUTPUT.t1o2,1e-6); 

%% 2) FitLinearGaussianParameters [ FitLinearGaussianParameters.m ]
case 2
 X = exampleINPUT.t2a1;
 U = exampleINPUT.t2a2;
 [Beta sigma] = FitLinearGaussianParameters(X, U);
 assert(Beta,exampleOUTPUT.t2o1,1e-6);
 assert(sigma,exampleOUTPUT.t2o2,1e-6);

%% 3) ComputeLogLikelihood [ ComputeLogLikelihood.m ]
case 3
% VisualizeDataset(trainData.data)  
P = exampleINPUT.t3a1;
G = exampleINPUT.t3a2;
dataset = exampleINPUT.t3a3;
loglikelihood = ComputeLogLikelihood(P, G, dataset);
assert(loglikelihood,exampleOUTPUT.t3,1e-6);
%% 4) LearnCPDsGivenGraph [ LearnCPDsGivenGraph.m ]

case 4
dataset = exampleINPUT.t4a1;
G = exampleINPUT.t4a2;
labels = exampleINPUT.t4a3;
PTrue = exampleOUTPUT.t4o1;
loglikelihoodTrue = exampleOUTPUT.t4o2;
[P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels);
 assert(P,exampleOUTPUT.t4o1,1e-6);
  assert(loglikelihood,exampleOUTPUT.t4o2,1e-6); 

%% 5) ClassifyDataset [ ClassifyDataset.m ]
case 5
 % test ClassifyDataset.m 
testData.data = exampleINPUT.t5a1;
testData.labels = exampleINPUT.t5a2;
P1 = exampleINPUT.t5a3;
G1 = exampleINPUT.t5a4;
accuracy1 = ClassifyDataset(testData.data, testData.labels, P1, G1);
assert(accuracy1,exampleOUTPUT.t5,1e-6);
%VisualizeModels(P1, G1);

% Compare structure G1 (no edges) and G2 (tree) load PA8Data % for trainData.data

%G1 = zeros(10, 2); 
%[P1 likelihood1] = LearnCPDsGivenGraph(trainData.data, G1, trainData.labels);
%accuracy1 = ClassifyDataset(testData.data, testData.labels, P1, G1);
%VisualizeModels(P1, G1);

%G2 = exampleINPUT.t5a4;
%[P2 likelihood2] = LearnCPDsGivenGraph(trainData.data, G2, trainData.labels);
%accuracy2 = ClassifyDataset(testData.data, testData.labels, P2, G2);
%VisualizeModels(P2, G2); 

%% 6) LearnGraphStructure [ LearnGraphStructure.m ]
case 6
 dataset = exampleINPUT.t6a1; 
 [A W] = LearnGraphStructure(dataset);
assert(A,exampleOUTPUT.t6o1,1e-6);
assert(W,exampleOUTPUT.t6o2,1e-6);
%% 7) LearnGraphAndCPDs [ LearnGraphAndCPDs.m ]
case 7
dataset = exampleINPUT.t7a1; labels = exampleINPUT.t7a2;
[P G loglikelihood] = LearnGraphAndCPDs(dataset, labels);
assert(P,exampleOUTPUT.t7o1,1e-6);
assert(G,exampleOUTPUT.t7o2,1e-6);
assert(loglikelihood,exampleOUTPUT.t7o3,1e-6);

end
