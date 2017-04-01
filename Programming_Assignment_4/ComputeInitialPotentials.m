%COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
%passed in as a parameter.
%
%   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
%   struct with three fields:
%   - nodes: cell array representing the cliques in the tree.
%   - edges: represents the adjacency matrix of the tree.
%   - factorList: represents the list of factors that were used to build
%   the tree. 
%   
%   It returns the standard form of a clique tree P that we will use through 
%   the rest of the assigment. P is struct with two fields:
%   - cliqueList: represents an array of cliques with appropriate factors 
%   from factorList assigned to each clique. Where the .val of each clique
%   is initialized to the initial potential of that clique.
%   - edges: represents the adjacency matrix of the tree. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function P = ComputeInitialPotentials(C)

% number of cliques
N = length(C.nodes);

% initialize cluster potentials 
P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
P.edges = zeros(N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% First, compute an assignment of factors from factorList to cliques. 
% Then use that assignment to initialize the cliques in cliqueList to 
% their initial potentials. 

% C.nodes is a list of cliques.
% So in your code, you should start with: P.cliqueList(i).var = C.nodes{i};
% Print out C to get a better understanding of its structure.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
P.edges = C.edges;

 for i=1:N
     P.cliqueList(i).var = C.nodes{i};
 end ;

 nFactors = length(C.factorList);

 vars = zeros(1);
 for i=1:nFactors
   %List of  Var Label:Cardinality Mappings
   vars(C.factorList(i).var)= C.factorList(i).card;
    
   for j=1:N
    % disp("Iter:");
  
      sd = setdiff(C.factorList(i).var, P.cliqueList(j).var);
       if (length(sd) < 1) 
           %C.factorList(i).var
           %P.cliqueList(j).var
           
               printf("Assign factor(%d) to clique %d\n", i, j);
           P.cliqueList(j).val= [P.cliqueList(j).val i];
           P.cliqueList(j).val;
           break;
       end;               
   end
 end

  %vars

 for i=1:N
  
     P.cliqueList(i).card = vars(P.cliqueList(i).var);
     factorIndx = P.cliqueList(i).val;

     % default for empty clique
     P.cliqueList(i).val=ones(1,prod(P.cliqueList(i).card));

     %factorIndx
     if ( length(factorIndx) > 0)        
         factors = C.factorList(factorIndx);

         %clqVars = P.cliqueList(i).var

% This part bombed ....
        % factorVars = factors(1).var;
        % if (length(factors) > 1)
        %    factorVars = cat(factors.var);
       %  end;

        factorVars = [factors.var];

        %check if additional factors needed
        varDiff = setdiff(P.cliqueList(i).var, factorVars);


        if (length(varDiff) > 0) 
             disp("Filling up");
       
             fillFactor = struct('var', [], 'card', [], 'val', []);
             fillFactor.var = varDiff;
             fillFactor.card = vars(varDiff);
             fillFactor.val=ones(1,prod(fillFactor.card));
  % The two lines below need to be adjusted depending on the data.
          % this works for submission
            % factors=[factors fillFactor];

           % this works for the OCR Run
             factors=[factors' fillFactor];
        end;
         Joint = ComputeJointDistribution(factors);
         %assert(length(Joint.val) ==  length(P.cliqueList(i).val))
         [dummy, mapJ] = ismember(Joint.var,   P.cliqueList(i).var);
         assignments = IndexToAssignment(1:prod(P.cliqueList(i).card), P.cliqueList(i).card);
         indxJ = AssignmentToIndex(assignments(:, mapJ), Joint.card);
         P.cliqueList(i).val=Joint.val(indxJ);

     end;
       % P.cliqueList(i)
 end;


end

