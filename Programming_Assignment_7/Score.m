 function [charAcc, wordAcc] =  Score(dataX, dataY, theta, modelParams)
  
   nSamples=size(dataX,2)
   
   nCorrectChars=0;
   nCorrectWords=0;
   for i=1:nSamples
     [pred] = PredictWord(dataX(i).X, dataY(i).y, theta, modelParams);
     nCorrectChars=nCorrectChars + sum(pred);
     if (all(pred))
        nCorrectWords= nCorrectWords+1;

     end
   end
     nCorrectChars
     nCorrectWords
 
     charAcc=nCorrectChars/(nSamples*size(dataX(1).X,1));
     wordAcc=nCorrectWords/nSamples;
 end
