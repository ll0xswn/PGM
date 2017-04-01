% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
  % Inputs: An influence diagram I with a single decision node and one or more utility nodes.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  % You may assume that there is a unique optimal decision.
  %
  % This is similar to OptimizeMEU except that we will have to account for
  % multiple utility factors.  We will do this by calculating the expected
  % utility factors and combining them, then optimizing with respect to that
  % combined expected utility factor.  
  MEU = [];
  OptimalDecisionRule = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  % A decision rule for D assigns, for each joint assignment to D's parents, 
  % probability 1 to the best option from the EUF for that joint assignment 
  % to D's parents, and 0 otherwise.  Note that when D has no parents, it is
  % a degenerate case we can handle separately for convenience.
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   UF=[I.UtilityFactors];
   
   EUF_P=[];
   for i=1:length(UF)
     I1=I;
     I1.UtilityFactors=UF(i);
     e=CalculateExpectedUtilityFactor( I1 );
     EUF_P=[EUF_P e];
   end
   
   FS=EUF_P(1);
   for i=2:length(EUF_P)
     FS=FactorSum(FS,EUF_P(i))
   end
    
    EUF=FS
    
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
       [MEU,midx] = max(EUF.val);
       %dVal=zeros(1:I.DecisionFactors.card);
       %dVal(midx)=1;
       %OptimalDecisionRule.val=dVal;
       OptimalDecisionRule.val(midx)=1.0;
    end



end
