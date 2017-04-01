% You should put all your code for recognizing unknown actions in this file.
% Describe the method you used in YourMethod.txt.
% Don't forget to call SavePrediction() at the end with your predicted labels to save them for submission, then submit using submit.m

function [predicted_labels] = RecognizeUnknownActions_1(datasetTrain,  datasetTest, G, maxIter)

% INPUTS
% datasetTrain: dataset for training models, see PA for details
% datasetTest: dataset for testing models, see PA for details
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 nActionLabels=size(datasetTrain,2)
  P_A={}; ClassProb_A={};
  PairProb_A = {};

  for i=1:nActionLabels
        % !!!TODO CHANGE
       trainActionData=datasetTrain(i).actionData(1:18);
      %  [d1 d2 InitClassProb] = EM_cluster(datasetTrain(i).poseData, G, datasetTrain(i).InitialClassProb, 20);

 KClass=size(datasetTrain(i).InitialClassProb,2);
       %InitClassProb =  datasetTrain(i).InitialClassProb;
        InitClassProb = (1/KClass).*ones(size(datasetTrain(i).InitialClassProb));
        InitPairProb =  datasetTrain(i).InitialPairProb;
        [P loglikelihood ClassProb PairProb] = EM_HMM(trainActionData, datasetTrain(i).poseData, G,  InitClassProb,  InitPairProb, maxIter);
         P_A{i}=P;
      ClassProb_A{i}=ClassProb;
  PairProb_A{i}=PairProb;
  end  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Classify each of the instances in datasetTrain
% Compute and return the predicted labels and accuracy
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

accuracy = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %  testActionData=
   % testPoseData=

 %  logEmissionProbPerAction={};
 % for i = 1:nActionLabels
  %  [lep] = ComputeEmissionFactors(testPoseData, P_A{i}, G);
 %   logEmissionProbPerAction{i}=lep;
 % end

   % for each action in test set
   %  nData=size(testActionData,2)
   %  predicted_labels = zeros(nData,1);
   
%temp  
   predicted_labels = zeros(36,1);
   for  ll=1:3
        testActionData=datasetTrain(ll).actionData(19:30);
        testPoseData=datasetTrain(ll).poseData;
        nData=size(testActionData,2)

        logEmissionProbPerAction={};
        for ie = 1:nActionLabels
          [lep] = ComputeEmissionFactors(testPoseData, P_A{ie}, G);
          logEmissionProbPerAction{ie}=lep;
        end

      for i=1:nData
          %ActionClasses
          loglikelihoodNK=zeros(1,nActionLabels);
          for ki=1:nActionLabels
          
           P=P_A{ki};
           logEmissionProb=logEmissionProbPerAction{ki};
           K = size(P.c,2);
          % create factors
           numFactors= length(testActionData(i).marg_ind)+ length(testActionData(i).pair_ind) + 1;
           factors = repmat(struct('var', [], 'card', [], 'val', []), 1, numFactors);
           
           % Initial State prior factor
           factors(1).var=[1];
           factors(1).card=[K];
           factors(1).val=log(P.c);
           
           % Transition factors
           transIndx=testActionData(i).pair_ind;
           for j=1:length(transIndx)  
              factors(j+1).var=[j j+1];
              factors(j+1).card=[K K]; 
              factors(j+1).val= log(P.transMatrix(:))';              
           end
           
           %Emission Factors P(PJ|S)
           offset=length(transIndx) + 1;
           stateIndx=testActionData(i).marg_ind;
           for j=1:length(stateIndx)
	       factors(j+offset).var=[j];
	       factors(j+offset).card=[K];
	       factors(j+offset).val= logEmissionProb(stateIndx(j),:);              
           end
         
           [M, PCalibrated] = ComputeExactMarginalsHMM(factors);
              
           % unnormalized Clique Tree
           loglikelihoodNK(ki)=logsumexp(PCalibrated.cliqueList(1).val);
          end
           loglikelihoodNK
           [mll mid]=max(loglikelihoodNK);
           predicted_labels((ll-1)*12+i)=mid
      end

%temp
     end
     nCorrect=sum([ones(1,12) 2*ones(1,12) 3*ones(1,12)]'==predicted_labels)
     accuracy=nCorrect/nData
     
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

