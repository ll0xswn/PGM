function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NP=length(P.clg); % number of body parts NP=10 

% samples
 for i=1:N

      %classes
      LogPKI=zeros(1,K);
      for cl=1:K

             LogPK_OI=log(P.c(cl));
             % P(C=k, O1=o1,,,)=P(C=k)*Prod_j(P(O_j|P_Opj,k)
             % body parts
             for j=1:NP
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

             LogPK_OI= LogPK_OI + logP_Oj_POj_k;            
             end
             LogPKI(cl)=LogPK_OI;
      end
     %LogPKI
     PKI=exp(LogPKI);
     POI=sum(PKI);
     logPOI=log(POI);
     loglikelihood=loglikelihood+logPOI;
 end
