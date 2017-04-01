function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

%body parts
NB=size(dataset,2);

loglikelihood = 0;
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
    NK=sum(labels(:,i)==1);
    P.c(i)=NK/N;
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

        nk=find(labels(:,cl) == 1);
	yj=dataset(nk,j,1);
        xj=dataset(nk,j,2);
        alphaj=dataset(nk,j,3);

        if (G_Head !=0)
           ypj=dataset(nk,PJ,1);
           xpj=dataset(nk,PJ,2);
           alphapj=dataset(nk,PJ,3);
           U=[ypj xpj alphapj];
           [theta_y sigma_y] = FitLinearGaussianParameters(yj, U);
           [theta_x sigma_x] = FitLinearGaussianParameters(xj, U);
           [theta_a sigma_a] = FitLinearGaussianParameters(alphaj, U);

           P.clg(j).theta(cl,:)=[theta_y(end) theta_y(1:end-1)' theta_x(end) theta_x(1:end-1)' theta_a(end) theta_a(1:end-1)'];
          
           P.clg(j).sigma_y(cl)=sigma_y;
           P.clg(j).sigma_x(cl)=sigma_x;
           P.clg(j).sigma_angle(cl)=sigma_a;
        else
	%head of graph
           [mu_y sigma_y] = FitGaussianParameters(yj);
           [mu_x sigma_x] = FitGaussianParameters(xj);
           [mu_a sigma_a] = FitGaussianParameters(alphaj);
           P.clg(j).mu_y(cl)=mu_y;
           P.clg(j).mu_x(cl)=mu_x;
           P.clg(j).mu_angle(cl)=mu_a;
           P.clg(j).sigma_y(cl)=sigma_y;
           P.clg(j).sigma_x(cl)=sigma_x;
           P.clg(j).sigma_angle(cl)=sigma_a;
        end

    end
 end

loglikelihood = ComputeLogLikelihood(P, G, dataset);
fprintf('log likelihood: %f\n', loglikelihood);

