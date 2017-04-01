% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
%   P(S_i|S_i-1) - table CPD
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m
  
  %P.c = zeros(1,K);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

  P = LearnEmissionCPDs(poseData, G, ClassProb);

  %index of the first state (pose) in the classProb: prob that first state=si
  ind1 =[] ;
  for  i=1:L
    ind1= [ind1 actionData(i).marg_ind(1)];
  end

  for ii=1:K
     P.c(ii) = sum(ClassProb(ind1,ii))/L;
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);
  
  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
   % S1, S2... Sm - hidden state vars
   % s1, s2, sK - state values, state assignments....; K=3 (clap, high kick, low kick)
   %P(S'=s'|S=s) - sum over transition matrices for all transitions in PairProb  
     for i=1:K
       for j=1:K
          indSdestSorig=AssignmentToIndex([j i], [K K]) ;      
          Ms1s=PairProb(:,indSdestSorig);     
          P.transMatrix(j,i)=sum(Ms1s);     
       end
     end

  
   % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb,1) * .05;

   % Normalize
   for i=1:K
       P.transMatrix(i,:)=P.transMatrix(i,:)/sum(P.transMatrix(i,:));
   end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
  
  logEmissionProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  logEmissionProb = ComputeEmissionFactors(poseData, P, G);
   

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  %PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % for each action
      for i=1:L
         % create factors
           numFactors= length(actionData(i).marg_ind)+ length(actionData(i).pair_ind) + 1;
           factors = repmat(struct('var', [], 'card', [], 'val', []), 1, numFactors);
           
           % Initial State prior factor
           factors(1).var=[1];
           factors(1).card=[K];
           factors(1).val=log(P.c);
           
           % Transition factors
           transIndx=actionData(i).pair_ind;
           %transCPDs=PairProb(transIndx,:);
           for j=1:length(transIndx)
             % factors(j+1).var=[j+1 j];
              factors(j+1).var=[j j+1];
              factors(j+1).card=[K K];
              %factors(j+1).val= log(transCPDs(j));     
              factors(j+1).val= log(P.transMatrix(:))';              
           end
           
           %Emission Factors P(PJ|S)
           offset=length(transIndx) + 1;
           stateIndx=actionData(i).marg_ind;
           for j=1:length(stateIndx)
	       factors(j+offset).var=[j];
	       factors(j+offset).card=[K];
	       factors(j+offset).val= logEmissionProb(stateIndx(j),:);              
           end
         
           [M, PCalibrated] = ComputeExactMarginalsHMM(factors);
           %Marginals are normalized
           %for each pose
           %ClassProb(stateIndx,:)=exp(M(:).val);   
           for mi=1:length(stateIndx)
              ClassProb(stateIndx(mi),:)=exp(M(mi).val);   
           end

            for jj =1:length(PCalibrated.cliqueList)
                vars = [PCalibrated.cliqueList(jj).var];
                PairProb(transIndx(vars(1)),:) = exp(PCalibrated.cliqueList(jj).val - logsumexp(PCalibrated.cliqueList(jj).val));
            end

        
           % unnormalized Clique Tree
           loglikelihood(iter)=loglikelihood(iter) + logsumexp(PCalibrated.cliqueList(1).val);
      end



  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting by decreasing loglikelihood
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);

end



 function [logEmissionProb] = ComputeEmissionFactors(dataset, P, G)
   % returns the updated ClassProb at the current iteration 
   %
   % Inputs:
   % dataset: N x 10 x 3, N pose instances represented by 10 parts
   % P: struct array model parameters (explained in PA description)
   % G: graph structure and parameterization (explained in PA description) 
   %
   % Outputs:
   % logEmissionProb - NxK; emission factors P(Pj|S) = Prod_i(P(Oi|Opar_i,S)
   % emission model for pose Pj with body parts O_1_10
   %
   % Copyright (C) Daphne Koller, Stanford Univerity, 2012
   
   N = size(dataset, 1); % n poses
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % YOUR CODE HERE
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    K = length(P.c); % number of states assignments
    NB=length(P.clg); % number of body parts NB=10 
    
    logEmissionProb = zeros(N,K);
   
    for i=1:N
   
         LogPKI=zeros(1,K);
         for cl=1:K
                LogPK_NI=0;
                % P(O1=o1,,,)=Prod_j(P(O_j|P_Opj,k)
                % body parts
                for j=1:NB
                    yi=dataset(i,j,1);
                    xi=dataset(i,j,2);
                    alphai=dataset(i,j,3);
   
                    logP_Oj_POj_k=0;
   
                    dimG=length([size(G)]);
                     
                    if (dimG == 3)
                      G_Head=G(j,1,cl);
                      PJ=G(j,2,cl);
                    else
                      %disp("2D G");
                      G_Head=G(j,1);
                      PJ=G(j,2);
                    end
   
                    if (G_Head !=0)
   
                       % G dimensions
                       %PJ=G(j,2);
   
                       %Y                                   
                       mu=P.clg(j).theta(cl,1) + P.clg(j).theta(cl,2)*dataset(i,PJ,1) + P.clg(j).theta(cl,3)*dataset(i,PJ,2)+ P.clg(j).theta(cl,4)*dataset(i,PJ,3);
                       sigma = P.clg(j).sigma_y(cl);
                       P_Y_Oj_POj_k=lognormpdf(yi,mu,sigma);
   
                       %X                    
                       mu=P.clg(j).theta(cl,5) + P.clg(j).theta(cl,6)*dataset(i,PJ,1) + P.clg(j).theta(cl,7)*dataset(i,PJ,2)+ P.clg(j).theta(cl,8)*dataset(i,PJ,3);
                       sigma = P.clg(j).sigma_x(cl);
                       P_X_Oj_POj_k=lognormpdf(xi,mu,sigma);
   
                       %Alpha                    
                       mu=P.clg(j).theta(cl,9) + P.clg(j).theta(cl,10)*dataset(i,PJ,1) + P.clg(j).theta(cl,11)*dataset(i,PJ,2)+ P.clg(j).theta(cl,12)*dataset(i,PJ,3);
                       sigma = P.clg(j).sigma_angle(cl);
                       P_A_Oj_POj_k=lognormpdf(alphai,mu,sigma);
   
                       logP_Oj_POj_k=P_Y_Oj_POj_k+P_X_Oj_POj_k+P_A_Oj_POj_k;
                    else
                      %head of graph
                       %Y                
                       mu=P.clg(j).mu_y(cl);
                       sigma = P.clg(j).sigma_y(cl);
                       P_Y_Oj_POj_k=lognormpdf(yi,mu,sigma);
   
                       %X
                       mu=P.clg(j).mu_x(cl);
                       sigma = P.clg(j).sigma_x(cl);
                       P_X_Oj_POj_k=lognormpdf(xi,mu,sigma);
   
                       %Alpha
                       mu=P.clg(j).mu_angle(cl);
                       sigma = P.clg(j).sigma_angle(cl);
                       P_A_Oj_POj_k=lognormpdf(alphai,mu,sigma);
   
                       logP_Oj_POj_k=P_Y_Oj_POj_k+P_X_Oj_POj_k+P_A_Oj_POj_k;
   
                    end
   
                LogPK_NI= LogPK_NI + logP_Oj_POj_k;
                %end body parts            
                end
                LogPKI(cl)=LogPK_NI;
         %end classes
         end
         % ??? do we need to normalize ????? NO!!!
        % LogNormConst=logsumexp(LogPKI);
        % LogNormPKI=bsxfun(@minus, LogPKI, LogNormConst);
        % logEmissionProb(i,:) = LogNormPKI;
         logEmissionProb(i,:) = LogPKI;
    end
end



function [P] = LearnEmissionCPDs(dataset, G, classProb)
   %
   % Inputs:
   % dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
   % G: graph parameterization as explained in PA description
   % classProb:N x K,  allocation of the N poses to the K     
   % classes. classProb(i,j) is the probability that example i belongs
   %   to class j; P(C=k|O1,...O10)
   
   % Outputs:
   % P: struct array parameters (explained in PA description)
   
   %
   % Copyright (C) Daphne Koller, Stanford Univerity, 2012
   
   N = size(dataset, 1);
   K = size(classProb,2);
   
   %body parts
   NB=size(dataset,2);
   
   P.clg=repmat(struct('mu_y', [], 'sigma_y', [], 'mu_x', [],'sigma_x',[],'mu_angle',[],'sigma_angle',[],'theta',[]), 1, NB);
   % estimate parameters
   % fill in P.c, MLE for class probabilities
   % fill in P.clg for each body part and each class
   % choose the right parameterization based on G(i,1)
   % compute the likelihood - you may want to use ComputeLogLikelihood.m
   % you just implemented.
   %%%%%%%%%%%%%%%%%%%%%%%%%
   % YOUR CODE HERE
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    for cl=1:K
       for j=1:NB
   
           dimG=length([size(G)]);
                     
           if (dimG == 3)
              G_Head=G(j,1,cl);
              PJ=G(j,2,cl);
           else
              %disp("2D G");
              G_Head=G(j,1);
              PJ=G(j,2);
           end
   
           WJK=classProb(:,cl);
   	   yj=dataset(:,j,1);
           xj=dataset(:,j,2);
           alphaj=dataset(:,j,3);
   
           if (G_Head !=0)
              ypj=dataset(:,PJ,1);
              xpj=dataset(:,PJ,2);
              alphapj=dataset(:,PJ,3);
              U=[ypj xpj alphapj];
              [theta_y sigma_y] = FitLG(yj, U, WJK);
              [theta_x sigma_x] = FitLG(xj, U, WJK);
              [theta_a sigma_a] = FitLG(alphaj, U, WJK);
   
              P.clg(j).theta(cl,:)=[theta_y(end) theta_y(1:end-1)' theta_x(end) theta_x(1:end-1)' theta_a(end) theta_a(1:end-1)'];
             
              P.clg(j).sigma_y(cl)=sigma_y;
              P.clg(j).sigma_x(cl)=sigma_x;
              P.clg(j).sigma_angle(cl)=sigma_a;
           else
   	      %head of graph
              [mu_y sigma_y] = FitG(yj, WJK);
              [mu_x sigma_x] = FitG(xj, WJK);
              [mu_a sigma_a] = FitG(alphaj, WJK);
              P.clg(j).mu_y(cl)=mu_y;
              P.clg(j).mu_x(cl)=mu_x;
              P.clg(j).mu_angle(cl)=mu_a;
              P.clg(j).sigma_y(cl)=sigma_y;
              P.clg(j).sigma_x(cl)=sigma_x;
              P.clg(j).sigma_angle(cl)=sigma_a;
           end
   
       end
    end
   
  end
