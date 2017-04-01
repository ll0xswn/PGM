function [datasetTrain] =  setCombinedTrainSet()
      load PA9Data.mat;
      
      datasetTrain(1).actionData=[datasetTrain1(1).actionData datasetTrain2(1).actionData datasetTrain3(1).actionData];
      datasetTrain(2).actionData=[datasetTrain1(2).actionData datasetTrain2(2).actionData datasetTrain3(2).actionData];
      datasetTrain(3).actionData=[datasetTrain1(3).actionData datasetTrain2(3).actionData datasetTrain3(3).actionData];

     
    %  datasetTrain(2).poseData=[datasetTrain1(2).poseData datasetTrain2(2).poseData datasetTrain3(2).poseData];
    %  datasetTrain(1).poseData=[datasetTrain1(1).poseData datasetTrain2(1).poseData datasetTrain3(1).poseData];

   %   datasetTrain(1).InitialClassProb=[datasetTrain1(1).InitialClassProb  datasetTrain2(1).InitialClassProb datasetTrain3(1).InitialClassProb];
    %  datasetTrain(2).InitialClassProb=[datasetTrain1(2).InitialClassProb  datasetTrain2(2).InitialClassProb datasetTrain3(2).InitialClassProb];
    %  datasetTrain(3).InitialClassProb=[datasetTrain1(3).InitialClassProb  datasetTrain2(3).InitialClassProb datasetTrain3(3).InitialClassProb];

    %  datasetTrain(1).InitialPairProb=[datasetTrain1(1).InitialPairProb  datasetTrain2(1).InitialPairProb  datasetTrain3(1).InitialPairProb];
    %  datasetTrain(2).InitialPairProb=[datasetTrain1(2).InitialPairProb  datasetTrain2(2).InitialPairProb  datasetTrain3(2).InitialPairProb];
    %  datasetTrain(3).InitialPairProb=[datasetTrain1(3).InitialPairProb  datasetTrain2(3).InitialPairProb  datasetTrain3(3).InitialPairProb];
        
     for  j=1:3 
  
         datasetTrain(j).poseData=zeros(size(datasetTrain1(j).poseData,1) + size(datasetTrain2(j).poseData,1) + size(datasetTrain3(j).poseData,1), 
                                        size(datasetTrain1(j).poseData,2) , size(datasetTrain1(j).poseData,3));
         s1=size(datasetTrain1(j).poseData,1)
         for i=1:s1
                datasetTrain(j).poseData(i,:,:)=datasetTrain1(j).poseData(i,:,:);
         end

        s2=size(datasetTrain2(j).poseData,1)
        for i=1:s2
                datasetTrain(j).poseData(s1+i,:,:)=datasetTrain2(j).poseData(i,:,:);
        end

        s3=size(datasetTrain3(j).poseData,1)
        for i=1:s3
                datasetTrain(j).poseData(s2+i,:,:)=datasetTrain3(j).poseData(i,:,:);
        end
        
        nData= size(datasetTrain1(j).actionData,2) + size(datasetTrain2(j).actionData,2) + size(datasetTrain3(j).actionData,2);
        datasetTrain(j).actionData=repmat(struct('action', [], 'marg_ind', [], 'pair_ind',[]), 1, nData);

        for i=1:size(datasetTrain1(j).actionData,2)
          datasetTrain(j).actionData(i)=datasetTrain1(j).actionData(i); 
        end
           j
                  
        a1=size(datasetTrain1(j).actionData,2);
        t1=size(datasetTrain1(j).InitialPairProb,1);
        for i=1:size(datasetTrain2(j).actionData,2)
           datasetTrain(j).actionData(a1+i)=datasetTrain2(j).actionData(i);
           mi = datasetTrain2(j).actionData(i).marg_ind +  s1;
           datasetTrain(j).actionData(a1+i).marg_ind = mi;

          pind = datasetTrain2(j).actionData(i).pair_ind +  t1;
          datasetTrain(j).actionData(a1+i).pair_ind = pind;
        end

         

        a2=size(datasetTrain2(j).actionData,2);
        t2=size(datasetTrain2(j).InitialPairProb,1);
        for i=1:size(datasetTrain3(j).actionData,2)
          datasetTrain(j).actionData(a1+a2+i)=datasetTrain3(j).actionData(i);

          datasetTrain(j).actionData(a1+a2+i).marg_ind = datasetTrain3(j).actionData(i).marg_ind +  s1+ s2;
          datasetTrain(j).actionData(a1+a2+i).pair_ind = datasetTrain3(j).actionData(i).pair_ind +  t2 + t2;
        end

        nIP=size(datasetTrain1(j).InitialClassProb,1) + size(datasetTrain2(j).InitialClassProb,1) + size(datasetTrain3(j).InitialClassProb,1)
        datasetTrain(j).InitialClassProb=zeros(nIP,size(datasetTrain1(j).InitialClassProb,2));
        datasetTrain(j).InitialPairProb=zeros(size(datasetTrain1(j).InitialPairProb,1) + size(datasetTrain2(j).InitialPairProb,1) + size(datasetTrain3(j).InitialPairProb,1));
   end
end 
