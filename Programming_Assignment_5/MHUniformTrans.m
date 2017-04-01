% MHUNIFORMTRANS
%
%  MCMC Metropolis-Hastings transition function that
%  utilizes the uniform proposal distribution.
%  A - The current joint assignment.  This should be
%      updated to be the next assignment
%  G - The network
%  F - List of all factors
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function A = MHUniformTrans(A, G, F)

% Draw proposed new state from uniform distribution
A_prop = ceil(rand(1, length(A)) .* G.card);

p_acceptance = 0.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% Compute acceptance probability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% acceptance=min(1, Pi(x')/Pi(x))
% Pi = joint distribution for a state ~ product(factors containing variables that changed)
% from state to state

%variables that changed
 delta_vars=find(A != A_prop);
 factorIndx=unique([G.var2factors{delta_vars}]);
 factors=F(factorIndx);

 LogPiX=0.0;
 LogPiX1=0.0;
 for i=1:length(factors)

      x=A(factors(i).var);
      x1=A_prop(factors(i).var);

      value_x=GetValueOfAssignment(factors(i),x);
      value_x1=GetValueOfAssignment(factors(i),x1);

      LogPiX = LogPiX + log(value_x);
      LogPiX1 = LogPiX1 + log(value_x1);
   end

     delta=LogPiX1 - LogPiX;
     p_acceptance=min(1, exp(delta));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Accept or reject proposal
if rand() < p_acceptance
    % disp('Accepted');
    A = A_prop;
end
