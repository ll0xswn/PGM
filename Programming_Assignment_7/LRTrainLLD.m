function thetaOpt = LRTrainLLD(dataX, dataY, theta, modelParams)

   maxIter=220;
   numFeatures = 2366;

    load Part2FullDataset.mat;

   %theta=zeros(1,numFeatures);
  for i=1:5
    for k=1:maxIter 
       X=trainData(k).X;
     Y=trainData(k).y;

      [cost gradL]= InstanceNegLogLikelihood(X, Y, theta, modelParams);
      %alpha=0.1/(1+ sqrt(k));
       alpha=1/(1+ 0.05*k);
      theta =  theta - alpha*gradL;
      cost
   end
 end
   thetaOpt=theta;

end
