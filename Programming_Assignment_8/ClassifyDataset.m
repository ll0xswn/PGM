function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
accuracy = 0.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 K = length(P.c); % number of classes
 NB=length(P.clg); % number of body parts NB=10 
 predLabels=zeros(N,2);

 for i=1:N

      LogPKI=zeros(1,K);
      for cl=1:K

             LogPK_OI=log(P.c(cl));
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

             LogPK_OI= LogPK_OI + logP_Oj_POj_k;            
             end
             LogPKI(cl)=LogPK_OI;
      end
      [mc midx] = max(LogPKI);
      predLabels(i,midx)=1;
 end
   %nCorrect=sum(predLabels(:,1) == labels(:,1))  
   nCorrect=length(find(sum( predLabels == labels,2) == size(labels,2)));
   accuracy=nCorrect/N;
fprintf('Accuracy: %.2f\n', accuracy);
