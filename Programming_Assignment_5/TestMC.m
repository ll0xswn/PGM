%
%
%

%

function TestMC()
load('exampleIOPA5.mat');

printf(' MCMCInference (part 2)');
 randi('seed',1);
  for iter = 1:20
      iter
       A = MHSWTrans(exampleINPUT.t11a1{iter}, exampleINPUT.t11a2{iter}, exampleINPUT.t11a3{iter}, ...
            exampleINPUT.t11a4{iter});
       result=isequal(A, exampleOUTPUT.t11{iter})
       if(result < 1)
           printf('WRONG!!!!');
       end 
    end

 printf(' BlockLogDistribution');
    LogBS = BlockLogDistribution(exampleINPUT.t6a1,exampleINPUT.t6a2, exampleINPUT.t6a3, exampleINPUT.t6a4);
    
    result = isequal(LogBS, exampleOUTPUT.t6)

    printf('GibbsTrans')
    randi('seed',1);
    result = 1;
    for iter = 1:10
       iter
        A = GibbsTrans(exampleINPUT.t7a1{iter},exampleINPUT.t7a2{iter}, exampleINPUT.t7a3{iter});
       
        result = result && isequal(A, exampleOUTPUT.t7{iter})
    end

    printf('MCMCInference')
    % second iteration works if exampleINPUT.t8a4{2} = MHGibbs
    result = 1;
    randi('seed',1);
   
       iter=1
        [M, all_samples] = MCMCInference(exampleINPUT.t8a1{1},exampleINPUT.t8a2{1}, exampleINPUT.t8a3{1}, ...
            exampleINPUT.t8a4{1}, exampleINPUT.t8a5{1}, exampleINPUT.t8a6{1}, exampleINPUT.t8a7{1}, ...
            exampleINPUT.t8a8{1});
        result = isequal(M, exampleOUTPUT.t8o1{1,1}) && isequal(all_samples, exampleOUTPUT.t8o2{1})
   
       exampleINPUT.t8a4{2}='MHGibbs';
        iter=2
        [M, all_samples] = MCMCInference(exampleINPUT.t8a1{2},exampleINPUT.t8a2{2}, exampleINPUT.t8a3{2}, ...
            exampleINPUT.t8a4{2}, exampleINPUT.t8a5{2}, exampleINPUT.t8a6{2}, exampleINPUT.t8a7{2}, ...
            exampleINPUT.t8a8{1});
        result = isequal(M, exampleOUTPUT.t8o1{1,2}) && isequal(all_samples, exampleOUTPUT.t8o2{2})

end
   
