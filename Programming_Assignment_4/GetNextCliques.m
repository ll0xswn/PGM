%GETNEXTCLIQUES Find a pair of cliques ready for message passing
%   [i, j] = GETNEXTCLIQUES(P, messages) finds ready cliques in a given
%   clique tree, P, and a matrix of current messages. Returns indices i and j
%   such that clique i is ready to transmit a message to clique j.
%
%   We are doing clique tree message passing, so
%   do not return (i,j) if clique i has already passed a message to clique j.
%
%	 messages is a n x n matrix of passed messages, where messages(i,j)
% 	 represents the message going from clique i to clique j. 
%   This matrix is initialized in CliqueTreeCalibrate as such:
%      MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);
%
%   If more than one message is ready to be transmitted, return 
%   the pair (i,j) that is numerically smallest. If you use an outer
%   for loop over i and an inner for loop over j, breaking when you find a 
%   ready pair of cliques, you will get the right answer.
%
%   If no such cliques exist, returns i = j = 0.
%
%   See also CLIQUETREECALIBRATE
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function [i, j] = GetNextCliques(P, messages)

% initialization
% you should set them to the correct values in your code
i = 0;
j = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=size(messages,1);
for i=1:N
    %i
    neighbors = sort(find(P.edges(i,:)));
    hasJSentToI=zeros(length(neighbors),1);
    hasINotSentToJ = zeros(length(neighbors),1);

    for j=1:length(neighbors)
        %Check for neighbor messages sent to I
        hasJSentToI(j)= (length(messages(neighbors(j),i).var) > 0) ;  
        %Check for messages I sent to neighbors
        hasINotSentToJ(j)= (length(messages(i,neighbors(j)).var) < 1);
     end
  
     %hasJSentToI
     % neighbors that has NOT sent messages
     idxNotJtoI = find(hasJSentToI == 0);

     %hasINotSentToJ
     % neighbors that has NOT been sent messge to by I
     idxItoJ=find(hasINotSentToJ ==1);

    %all neighbors sent messages to I; I has not sent a message to J
    if ( (length(idxNotJtoI) == 0) && (length(idxItoJ) > 0))         
         %pick the smallest index neighbor 
         j = neighbors(idxItoJ(1));
         printf("%d ready to send to %d\n", i, j);
         return;
     end

  %if one message not received, make sure it is from the neighbor J to sent message to
   if (length(idxNotJtoI) == 1)
      if (ismember(idxNotJtoI, idxItoJ ) )
         j = neighbors(idxNotJtoI(1));
         printf("%d ready to send to %d\n", i, j);
         return;
      end
   end
 i=0; j=0;
end

return;
