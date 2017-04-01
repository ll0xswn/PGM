function featureCounts = GetFeatureCounts(theta, Y, featureSet) 

  featureCounts=zeros(1,length(theta));
  for i=1:length(theta)
     fIdx=find([featureSet.features.paramIdx] == i);
     featuresForTheta=featureSet.features(fIdx);
     for j=1:length(featuresForTheta)
          match=all(featuresForTheta(j).assignment== Y(featuresForTheta(j).var));
          if (match)
            featureCounts(i) = featureCounts(i) + 1;
          end
     end
       %featureCounts(i)
  end

end
