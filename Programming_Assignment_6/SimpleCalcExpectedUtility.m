% Copyright (C) Daphne Koller, Stanford University, 2012

function EU = SimpleCalcExpectedUtility(I)

  % Inputs: An influence diagram, I (as described in the writeup).
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return Value: the expected utility of I
  % Given a fully instantiated influence diagram with a single utility node and decision node,
  % calculate and return the expected utility.  Note - assumes that the decision rule for the 
  % decision node is fully assigned.

  % In this function, we assume there is only one utility node.
  F = [I.RandomFactors I.DecisionFactors];
  U = I.UtilityFactors(1);
  EU = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  U_Vars= unique([U.var])
  X_Vars=unique([I.RandomFactors.var])
  X_D_Vars=unique([F.var])
  D_Vars= I.DecisionFactors.var

  V_Elim=setdiff(X_D_Vars,U_Vars)

  FP_V = VariableElimination(F, V_Elim);
  
  FP=FP_V(1);
  for i=2:length(FP_V)
    FP=FactorProduct(FP,FP_V(i));
  end
  P_X_D = FP;

 % FP=F(1);
 % for i=2:length(F)
 %   FP=FactorProduct(FP,F(i));
 %  end
 %P_X_D = VariableElimination(FP, V_Elim);
 
 assert(isempty(setdiff(P_X_D.var,U.var)));
 
  UTF=FactorProduct(P_X_D,U);

  EU=sum(UTF.val);
  
end
