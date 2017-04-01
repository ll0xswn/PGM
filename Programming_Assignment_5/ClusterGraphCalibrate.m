% CLUSTERGRAPHCALIBRATE Loopy belief propagation for cluster graph calibration.
%   P = CLUSTERGRAPHCALIBRATE(P, useSmart) calibrates a given cluster graph, G,
%   and set of of factors, F. The function returns the final potentials for
%   each cluster. 
%   The cluster graph data structure has the following fields:
%   - .clusterList: a list of the cluster beliefs in this graph. These entries
%                   have the following subfields:
%     - .var:  indices of variables in the specified cluster
%     - .card: cardinality of variables in the specified cluster
%     - .val:  the cluster's beliefs about these variables
%   - .edges: A cluster adjacency matrix where edges(i,j)=1 implies clusters i
%             and j share an edge.
%  
%   UseSmart is an indicator variable that tells us whether to use the Naive or Smart
%   implementation of GetNextClusters for our message ordering
%
%   See also FACTORPRODUCT, FACTORMARGINALIZATION
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function [P MESSAGES] = ClusterGraphCalibrate(P,useSmartMP)

if(~exist('useSmartMP','var'))
  useSmartMP = 0;
end

N = length(P.clusterList);

MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);
[edgeFromIndx, edgeToIndx] = find(P.edges ~= 0);

for m = 1:length(edgeFromIndx),
    i = edgeFromIndx(m);
    j = edgeToIndx(m);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %
    %
    %
    % Set the initial message values
    % MESSAGES(i,j) should be set to the initial value for the
    % message from cluster i to cluster j
    %
    % The matlab/octave functions 'intersect' and 'find' may
    % be useful here (for making your code faster)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SIJ=intersect(P.clusterList(i).var, P.clusterList(j).var);
    
    MESSAGES(i,j).var = SIJ;
    for k=1:length(SIJ)
       MESSAGES(i,j).card(k) = P.clusterList(i).card(find(P.clusterList(i).var == SIJ(k)));  
    end
    MESSAGES(i,j).val = ones(1,prod(MESSAGES(i,j).card));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end;


% perform loopy belief propagation
tic;
iteration = 0;

lastMESSAGES = MESSAGES;

            %Needed for assignment quiz
            %residual=true;
while (1),
    iteration = iteration + 1;
    [i, j] = GetNextClusters(P, MESSAGES,lastMESSAGES, iteration, useSmartMP);
    prevMessage = MESSAGES(i,j);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    % We have already selected a message to pass, \delta_ij.
    % Compute the message from clique i to clique j and put it
    % in MESSAGES(i,j)
    % Finally, normalize the message to prevent overflow
    %
    % The function 'setdiff' may be useful to help you
    % obtain some speedup in this function
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %i
   %j
       SIJ=intersect(P.clusterList(i).var, P.clusterList(j).var);

       factors=[P.clusterList(i)];
       neighbors = sort(find(P.edges(i,:)));
       neigborsMinJ=setdiff(neighbors, j);
       for l=1:length(neigborsMinJ)       
           factors=[factors  MESSAGES(neigborsMinJ(l),i)];
       end
  
       %compute message
       Joint = factors(1);
       for fi = 2:length(factors)
	  Joint = FactorProduct(Joint, factors(fi));
       end
       %Joint
       M = FactorMarginalization(Joint, setdiff(Joint.var, SIJ));
       
       % Normalize Message_i->j
       M.val = M.val ./ sum(M.val);
       MESSAGES(i,j)=M;

       %sMSS=length([MESSAGES.val])
       %sLMSS=length([lastMESSAGES.val])
       %assert(sMSS==sLMSS);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
     %This needed for assignment quiz
     %   if (  (i==15 && j==40) && residual) %|| (i==15 && j==40) || (i==17 && j==2) ) (i==19 && j==3)
     %      iteration
     %       del=MessageDelta(MESSAGES(i,j), lastMESSAGES(i,j))
     %        if (del < 1.0e-06)
     %             iter19to3=iteration
     %             residual=false;
     %       end          
     %  end
    

    if(useSmartMP==1)
      lastMESSAGES(i,j)=prevMessage;
    end
    
    % Check for convergence every m iterations
    if mod(iteration, length(edgeFromIndx)) == 0
        if (CheckConvergence(MESSAGES, lastMESSAGES))
            break;
        end
        disp(['LBP Messages Passed: ', int2str(iteration), '...']);
        if(useSmartMP~=1)
          lastMESSAGES=MESSAGES;
        end
    end
    
    if (iteration > 100000)
      break;
    end
end;
toc;
disp(['Total number of messages passed: ', num2str(iteration)]);

% Compute final potentials and place them in P
for m = 1:length(edgeFromIndx),
    j = edgeFromIndx(m);
    i = edgeToIndx(m);
    P.clusterList(i) = FactorProduct(P.clusterList(i), MESSAGES(j, i));
end


% Get the max difference between the marginal entries of 2 messages -------
function delta = MessageDelta(Mes1, Mes2)
delta = max(abs(Mes1.val - Mes2.val));
return;


