% Copyright (C) Daphne Koller, Stanford University, 2012

function EUF = CalculateExpectedUtilityFactor( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: A factor over the scope of the decision rule D from I that
  % gives the conditional utility given each assignment for D.var
  %
  % Note - We assume I has a single decision node and utility node.
  EUF = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  F=[I.RandomFactors I.UtilityFactors];

  X_U_Vars=unique([I.RandomFactors.var I.UtilityFactors.var])
  D_Vars = I.DecisionFactors.var
  Min_D_Vars=setdiff(X_U_Vars, D_Vars)
  
  TEU=VariableElimination(F, Min_D_Vars);
  
  FP=TEU(1);
  for i=2:length(TEU)
    FP=FactorProduct(FP,TEU(i));
  end
  
  EUF=FP;
end  
