function fit=fitnessdctpso(position,classnum,totsize)
global classmeanarray totalmeanarray
for i=1:totsize
  modct(i)=double(position(i)).*double(totalmeanarray(i));               %array containing features of the entire database for testing or training
end
sum=0;
for i=1:classnum                                                             
   str = int2str(i);
       for j=1:totsize
        midct(j)=double(position(j)).*double(classmeanarray{i}(j));        %array containing features of the folder for testing or training
   end
    diff=abs(midct-modct);                                                 %calculate difference between them
    tpose=diff';
    mul=mtimes(diff,tpose);
    sum=double(sum)+double(mul);                                           %eucledian distance
end
 fit=sqrt(sum);                                                            %standard deviation
