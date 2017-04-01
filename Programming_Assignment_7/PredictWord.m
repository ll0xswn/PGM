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

function [pred] = PredictWord(X, y, theta, modelParams)

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

    
   [P,F] = GetUncalibratedCliqueTree(featureSet,theta);
  
   [PCalibrated, logZ] = CliqueTreeCalibrate(P, false);
  
   F1=FactorMarginalization(PCalibrated.cliqueList(1),[2]);
   F1.val=F1.val/sum(F1.val);
   F2=FactorMarginalization(PCalibrated.cliqueList(1),[1]);
   F2.val=F2.val/sum(F2.val);
   F3=FactorMarginalization(PCalibrated.cliqueList(2),[2]);
   F3.val=F3.val/sum(F3.val);

   [m, i1]=max(F1.val);
   [m, i2]=max(F2.val);
   [m, i3]=max(F3.val);

   ypred=[i1 i2 i3]
   y
   pred=(ypred==y)
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



