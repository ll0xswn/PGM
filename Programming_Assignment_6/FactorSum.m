function C = FactorSum(A,B) 

   if (isempty(A.var)), C = B; return; end;
   if (isempty(B.var)), C = A; return; end;

   % Check that variables in both A and B have the same cardinality
   [dummy iA iB] = intersect(A.var, B.var);
   if ~isempty(dummy)
	% A and B have at least 1 variable in common
	assert(all(A.card(iA) == B.card(iB)), 'Dimensionality mismatch in factors');
   end

   C.var = union(A.var, B.var);

% Construct the mapping between variables in A and B and variables in C.
% In the code below, we have that
%
%   mapA(i) = j, if and only if, A.var(i) == C.var(j)
% 
% and similarly 
%
%   mapB(i) = j, if and only if, B.var(i) == C.var(j)
%
% For example, if A.var = [3 1 4], B.var = [4 5], and C.var = [1 3 4 5],
% then, mapA = [2 1 3] and mapB = [3 4]; mapA(1) = 2 because A.var(1) = 3
% and C.var(2) = 3, so A.var(1) == C.var(2).

  [dummy, mapA] = ismember(A.var, C.var);
  [dummy, mapB] = ismember(B.var, C.var);

  C.card = zeros(1, length(C.var));
  C.card(mapA) = A.card;
  C.card(mapB) = B.card;

  C.val = zeros(1,prod(C.card));

  assignments = IndexToAssignment(1:prod(C.card), C.card);
  indxA = AssignmentToIndex(assignments(:, mapA), A.card);
  indxB = AssignmentToIndex(assignments(:, mapB), B.card);

  C.val = A.val(indxA) + B.val(indxB);

return