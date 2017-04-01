% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  
  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  D = I.DecisionFactors(1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  % 
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D 
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

    EUF=CalculateExpectedUtilityFactor( I )  
    OptimalDecisionRule= EUF;
    OptimalDecisionRule.val(:)=0;
    if ( length( I.DecisionFactors.var) > 1)
    % D has parents
        I2A=IndexToAssignment((1:length(EUF.val)),EUF.card);
        %decision variable column
        dvar_idx = find(EUF.var ==  I.DecisionFactors.var(1))
        %find full assugnment for parents of D
        x_assig_idx= find(I2A(:,dvar_idx) == 1)
        x_assig=I2A(x_assig_idx,:);

        d = I.DecisionFactors.card(1)
        d_vec=(1:d)';

       for jj=1:size(x_assig,1)
          block=repmat(x_assig(jj,:),d,1);
          block(:,dvar_idx)=d_vec;
          val_ind=AssignmentToIndex(block, EUF.card)

          [maxU, id_max] = max(EUF.val(val_ind))
 
          %val_ind(id_max)

          OptimalDecisionRule.val(val_ind(id_max))=maxU;
          %OptimalDecisionRule     
       end

         MEU=sum(OptimalDecisionRule.val);
         marked=(OptimalDecisionRule.val !=0);
         OptimalDecisionRule.val=marked;
    else 
        % D has NO parents
       [MEU,midx] = max(EUF.val)
       %dVal=zeros(1:D.card);
       %dVal(midx)=1;
       %OptimalDecisionRule.val=dVal;
       OptimalDecisionRule.val(midx)=1.0
    end

end
