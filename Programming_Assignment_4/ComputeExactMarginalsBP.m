%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)

% initialization
% you should set it to the correct value in your code
M = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  

     rawVars=[F.var];
     vars=unique(rawVars);
     nVars=length(vars);
     M=repmat(struct('var', [], 'card', [], 'val', []), nVars, 1);
     
     P = CreateCliqueTree(F, E);
     P = CliqueTreeCalibrate(P, isMax);

     cliques=P.cliqueList;

   if(isMax == 0)
     for i=1:nVars   
     %i      
         for j=1:length(cliques)
            if(ismember(i,cliques(j).var))
                 %vars=cliques(j).var
                 V=setdiff(cliques(j).var,i);           
                 M(i) = FactorMarginalization(cliques(j),V );               
                 M(i).val = M(i).val ./ sum(M(i).val);
                break;
            end
         end;
      end 
  end

  if(isMax ==1)
     for i=1:nVars   
     %i      
         for j=1:length(cliques)
            if(ismember(i,cliques(j).var))
                 %vars=cliques(j).var
                 V=setdiff(cliques(j).var,i);           
                 M(i) = FactorMaxMarginalization(cliques(j),V );               
                break;
            end
         end;
      end 
  end



end
