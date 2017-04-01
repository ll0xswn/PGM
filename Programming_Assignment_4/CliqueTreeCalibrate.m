%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sum-product calibration
 if (isMax==0)
     [i, j] = GetNextCliques(P, MESSAGES)
     k=1;
     while (i!=0 && j!=0)
        k
        SIJ=intersect(P.cliqueList(i).var, P.cliqueList(j).var);
        %MESSAGES(i,j).var=SIJ;

        factors=[P.cliqueList(i)];
   
        neighbors = sort(find(P.edges(i,:)))
        neigborsMinJ=setdiff(neighbors, j);
        for l=1:length(neigborsMinJ)
          if (length(MESSAGES(neigborsMinJ(l),i).var) > 0) 
            factors=[factors  MESSAGES(neigborsMinJ(l),i)];
          end
        end
  
        %len=length(factors);
        M = ComputeMarginal(SIJ, factors, []);
        % Normalize Message_i->j
        M.val = M.val ./ sum(M.val);
        MESSAGES(i,j)=M;

        k++;
       [i, j] = GetNextCliques(P, MESSAGES);
     endwhile

 %max-sum calibration
 elseif(isMax==1)
   %disp("Start Max-Sum Propagation"); 
     for ic=1:N
        P.cliqueList(ic).val=log(P.cliqueList(ic).val);
     end
     
     [i, j] = GetNextCliques(P, MESSAGES)
     k=1;
     while (i!=0 && j!=0)
        k
        SIJ=intersect(P.cliqueList(i).var, P.cliqueList(j).var);
      
        factors=[P.cliqueList(i)];
        
        neighbors = sort(find(P.edges(i,:)))
        neigborsMinJ=setdiff(neighbors, j);
        for l=1:length(neigborsMinJ)
          if (length(MESSAGES(neigborsMinJ(l),i).var) > 0) 
            factors=[factors  MESSAGES(neigborsMinJ(l),i)];
          end
        end
          
        fsum=factors(1);
        for l=2:length(factors)
          fsum = FactorSum(fsum, factors(l));
        end
        M=FactorMaxMarginalization(fsum, setdiff(fsum.var,SIJ));
        MESSAGES(i,j)=M;

        k++;
       [i, j] = GetNextCliques(P, MESSAGES);
     endwhile
 end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sum-product potentials
if (isMax==0)
   
   for i=1:N
      factors=[P.cliqueList(i)]; 
      neighbors = sort(find(P.edges(i,:)));
      for l=1:length(neighbors)
          if (length(MESSAGES(neighbors(l),i).var) > 0) 
            factors=[factors  MESSAGES(neighbors(l),i)];
          end
      end

     Joint = ComputeJointDistribution(factors);
     P.cliqueList(i).val=Joint.val;
   end
elseif(isMax==1)
   for i=1:N
      factors=[P.cliqueList(i)]; 
 
      neighbors = sort(find(P.edges(i,:)));
      for l=1:length(neighbors)
          if (length(MESSAGES(neighbors(l),i).var) > 0) 
            factors=[factors  MESSAGES(neighbors(l),i)];
          end
      end

     fsum=factors(1);
     for l=2:length(factors)
       fsum = FactorSum(fsum, factors(l));
     end

     P.cliqueList(i).val=fsum.val;

  end
end;

return




