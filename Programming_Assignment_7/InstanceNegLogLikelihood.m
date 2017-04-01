% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over X_2 and X_3, which takes on a value of 1
    % if X_2 = 5 and X_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    
    featureSet = GenerateAllFeatures(X, modelParams);

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.
    
    nll = 0;
    grad = zeros(size(theta));
    %%%
    % Your code here:
    [P,F] = GetUncalibratedCliqueTree(featureSet,theta);
    
   %load('Part2Sample.mat');
   %c1=all(abs(sampleUncalibratedTree.cliqueList(1).val - P.cliqueList(1).val) < 1e-10)
   %c2=all(abs(sampleUncalibratedTree.cliqueList(2).val - P.cliqueList(2).val) < 1e-10)
   [PCalibrated, logZ] = CliqueTreeCalibrate(P, false);
    
   %sampleLogZ
   %c1=all(abs(sampleCalibratedTree.cliqueList(1).val - P.cliqueList(1).val) < 1e-10)
   %c2=all(abs(sampleCalibratedTree.cliqueList(2).val - P.cliqueList(2).val) < 1e-10)
   
   featureCounts = GetFeatureCounts(theta, y, featureSet); 

   weightedFeatureCounts = theta.* featureCounts;

   regularizationCost=0.5*modelParams.lambda*sum(theta.^2);

   %fc=all(featureCounts == sampleFeatureCounts)
   %wfc = all(weightedFeatureCounts == sampleWeightedFeatureCounts)
   %rc = (abs(regularizationCost-sampleRegularizationCost) < 1.e-10)

   nll= logZ - sum(weightedFeatureCounts) + regularizationCost;

   modelFeatureCounts = GetModelExpectedFeatureCounts(theta, PCalibrated, featureSet);
   %mfc=all((abs(sampleModelFeatureCounts - modelFeatureCounts) < 1.0e-07))

   regularizationGradient = modelParams.lambda*theta;
   %rgc=all(abs(sampleRegularizationGradient-regularizationGradient) < 1.0e-07)

   grad=modelFeatureCounts - featureCounts + regularizationGradient;
   %gc=all((abs(sampleGrad- grad) < 1.0e-07))
  
end


function [P, F] = GetUncalibratedCliqueTree(featureSet,theta)
         nFeatures=length(featureSet.features);
    factors=[];
    for i=1:nFeatures
      f = EmptyFactorStruct;
      f.var=featureSet.features(i).var;
      f.card = repmat([26], 1, length(f.var));
      f.val=ones(1,prod(f.card));
      indV=AssignmentToIndex(featureSet.features(i).assignment,f.card);
      f.val(indV)=exp(theta(featureSet.features(i).paramIdx));
      
      factors=[factors f];
    end
      F=factors;
      P = CreateCliqueTree(factors);
 
end

function featureCounts = GetFeatureCounts(theta, Y, featureSet) 

  featureCounts=zeros(1,length(theta));
  for i=1:length(theta)
     fIdx=find([featureSet.features.paramIdx] == i);
     featuresForTheta=featureSet.features(fIdx);
     for j=1:length(featuresForTheta)
          match=all(featuresForTheta(j).assignment== Y(featuresForTheta(j).var));
          if (match)
            featureCounts(i) = featureCounts(i) + 1;
          end
     end
       %featureCounts(i)
  end

end

%
%Expected Model Feature Counts = Sum_Y'(P(Y'|x)f_i(Y',x)
%
function modelFeatureCounts = GetModelExpectedFeatureCounts(theta, PCalibrated, featureSet) 

%load('Part2Sample.mat');
  modelFeatureCounts=zeros(1,length(theta));

%Precalculate Probailities/Marginals (P(Y|x) from Calibrated CliqueTree
  vars = unique([featureSet.features.var]);
  Marg = repmat(struct('var', 0, 'card', 0, 'val', []), length(vars), length(vars));

  for i=1:length(vars)
     Marg(i,i) = getProbThetaYX(PCalibrated, vars(i));
     for j=i+1:length(vars)
        MM = getProbThetaYX(PCalibrated, [vars(i), vars(j)]);
        if (~isempty(MM))
            Marg(i,j)=MM;
            Marg(j,i)=MM;
        end     
     end
  end

 %Marg

  for i=1:length(theta)
     fIdx=find([featureSet.features.paramIdx] == i);
     featuresForTheta=featureSet.features(fIdx);
     sumFJTheta=0.0;
     for j=1:length(featuresForTheta)
        V=[featuresForTheta(j).var];
        if (length(V) == 1)
          PXY=Marg(V(1),V(1));
        else
          PXY=Marg(V(1),V(2));
        end
         %PXY
        % assig=featuresForTheta(j).assignment
         indPE=AssignmentToIndex(featuresForTheta(j).assignment,PXY.card);
         %PXY.val(indPE)
         sumFJTheta=sumFJTheta+PXY.val(indPE);
     end
       modelFeatureCounts(i)=sumFJTheta;
      %  i
      % sumFJTheta 
     %  sim=(abs(sampleModelFeatureCounts(i) - modelFeatureCounts(i)) < 1.0e-07)
  end
     % disp("MODEL FEATURE COUNTS");
     % sim=all((abs(sampleModelFeatureCounts - modelFeatureCounts) < 1.0e-07))
end



function  M = getProbThetaYX(PCalibrated, V)
%V
    clique = struct('var', 0, 'card', 0, 'val', []);
    M = struct('var', 0, 'card', 0, 'val', []);
    for k = 1:length(PCalibrated.cliqueList)
        % Find a clique with the variable of interest
        if (sum(ismember(PCalibrated.cliqueList(k).var, V)) == length(V))
            % A clique with the variable has been indentified
            clique = PCalibrated.cliqueList(k);   
            M = FactorMarginalization(clique, setdiff(clique.var, V));
            if any(M.val ~= 0)
              % Normalize
               M.val = M.val/sum(M.val);
            end     
            break
        end
    end
 end

