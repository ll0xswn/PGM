function modelFeatureCounts = GetModelExpectedFeatureCounts(theta, PCalibrated, featureSet) 

%load('Part2Sample.mat');
  modelFeatureCounts=zeros(1,length(theta));

%PRecalculate Probailities/Marginals from Calibrated CliqueTree
  vars = unique([featureSet.features.var])
  Marg = repmat(struct('var', 0, 'card', 0, 'val', []), length(vars), length(vars))

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
      % sim=(abs(sampleModelFeatureCounts(i) - modelFeatureCounts(i)) < 1.0e-07)
  end
      %sim=all((abs(sampleModelFeatureCounts - modelFeatureCounts) < 1.0e-07))
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
