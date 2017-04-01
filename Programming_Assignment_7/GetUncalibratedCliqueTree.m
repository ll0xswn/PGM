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
