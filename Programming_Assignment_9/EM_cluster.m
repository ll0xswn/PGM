% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  %
  % Hint: This part should be similar to your work from PA8
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  P = LearnCPDsGivenGraph(poseData, G, ClassProb);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(i,j) is the probability that example i belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (i.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  [ClassProb loglikelihoodIter] = UpdateClassProb(poseData, P, G);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Compute log likelihood of dataset for this iteration
  % Hint: You should use the logsumexp() function here
  %loglikelihood(iter) = 0;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  loglikelihood(iter) = loglikelihoodIter;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);

end




function [P] = LearnCPDsGivenGraph(dataset, G, classProb)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% classProb:N x K, initial allocation of the N poses to the K     
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

P.c = zeros(1,K);
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

 for i=1:K
    P.c(i)=sum(classProb(:,i))/N;
 end

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




function [ClassProb loglikelihoodIter] = UpdateClassProb(dataset, P, G)
% returns the updated ClassProb at the current iteration 
%
% Inputs:
% dataset: N x 10 x 3, N pose instances represented by 10 parts
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% ClassProb: Updated ClassProb: N x K,  allocation of the N poses to the K
%  classes. ClassProb(i,j) is the probability that example i belongs
%   to class j
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 K = length(P.c); % number of classes
 NB=length(P.clg); % number of body parts NB=10 
 
 ClassProb = zeros(N,K);

 loglikelihoodIter =0.0;
 for i=1:N

      LogPKI=zeros(1,K);
      for cl=1:K
             LogPK_NI=log(P.c(cl));
             % P(C=k, O1=o1,,,)=P(C=k)*Prod_j(P(O_j|P_Opj,k)
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
      LogNormConst=logsumexp(LogPKI);
      LogNormPKI=bsxfun(@minus, LogPKI, LogNormConst);
      ClassProb(i,:) = exp(LogNormPKI);
      loglikelihoodIter = loglikelihoodIter + LogNormConst;
 end
  

end


function [log_prob] = lognormpdf(x,mu,sigma)

% LOGNORMPDF Natural logarithm of the normal probability density function (pdf)
% Y = lognormpdf(X,MU,SIGMA) returns the log of the pdf of the normal
% distribution parameterized by mean MU and standard deviation SIGMA evaluated
% at each value in the vector X. Thus, the size of the return
% vector Y is the size of X. 
% 
% MU and X should have the same dimensions.

log_prob = -log(sigma*sqrt(2*pi))-(x-mu).^2 ./ (2*sigma.^2);
end
