% T

% ffd

function PW = PredictWord(word, imageModel, pairwiseModel, tripletList )

 factors = BuildOCRNetwork(word, imageModel, pairwiseModel,tripletList);
 
 for i=1:length(factors) 
    factors(i).val=factors(i).val';
 end

 maxMarginals = ComputeExactMarginalsBP(factors,[],1);
 MAPAssignment = MaxDecoding(maxMarginals);
 DecodedMarginalsToChars1(MAPAssignment);  
 PW=MAPAssignment;
end

function CRS = DecodedMarginalsToChars1(decodedMarginals)
    chars = 'abcdefghijklmnopqrstuvwxyz';
    fprintf('%c', chars(decodedMarginals));
    fprintf('\n');
    CRS=chars(decodedMarginals);
end
