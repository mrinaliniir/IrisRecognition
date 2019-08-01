%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Enhanced Iris Recognition using Discrete Cosine Transform and Radon Transform               %                                                                               %
%                                  Mrinalini I.R, Pratusha B.P                                            %
%                          M.S Ramaiah Institute of Technology, India                                     %
%                                                                                                         %
%                                                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



clear all                               %          
clc                                     % clears the command window and homes the cursor
close all                               % closes all the open figure windows
global classmeanarray totalmeanarray GlobalBestP;         %Gobal declaration
%% Generalised Variables
countsum=0;
percentsum=0;
locstr= 'C:\Users\irmri\OneDrive\Documents\undergrad_research\IITD\';  %reading the images in the colorferet folder                      
TotNoIter=1;                                     % Total number of iterations
DctSize= [1 625];                                % The overall DCT coefficients size is initialized to 625

%% Specifications regarding the IITD database %%
        TClasses=224;                                      % Total number of  classes (224) present in the database
        Formt='.bmp';                                      % The images are in the bitmap format 
        TImgPerClass=10;                                   % Total number of images per class is 10
        NoOfTrImg=8;                                       % Number of training images per class is initialized to 8, according to the ratio required 8:2
TTrImg=NoOfTrImg*TClasses;                                 % Overall number of training images is given by the product shown
XDct=DctSize(1); YDct=DctSize(2);                          % coeff along x &y coordinates.
TDctSize=XDct*YDct;                                        % TDctSize is the total size of the face gallery


for NoOfTimes=1:TotNoIter                                  % runs the code starting from 1 to the max number of iterations provided. More number of iterations more is the accuracy.
    fprintf(1, '\n');                                      % displays the iteration number
    disp(strcat('Iteration number:',num2str(NoOfTimes)));  
    
    tic                                                     % Tic and toc is used to calculate the training time and testing time
    TTotal=zeros(DctSize(1),DctSize(2));                    %initialize Ttotal to 0 , coeff of all folders 
    k=1;
    for TotClasses=1:TClasses
        b{TotClasses}=randperm(TImgPerClass,NoOfTrImg);     % by random permutation it selects the number randomly
        TSum=zeros(DctSize(1),DctSize(2));                  % coeff in each folder initialize to 0
        for TotTrImg = 1:NoOfTrImg
            
            IR=imread(strcat(locstr,num2str(TotClasses),'\',num2str(b{TotClasses}(TotTrImg),'%02i'),Formt)); % reading the images
  IR= imresize(IR,0.125);
  IR= rgb2gray(IR);                                         % converting the images from rgb to gray scale
 [IR or] = canny(IR, 1,10,4,7); % canny edge detector markes the boundary isolating the iris from the eye image
   %  IR = edge(IR,'canny');
         IR=radon(IR);                                      % used for curve detection and eliminates illumination variations
         IR=abs(IR);                                        % takes absolute value of the complex number
         IR=dct2(IR);
         
         K=IR(1:25,1:25);                                  %resizing an image
         K=flipdim(K,1);                                   % flips the square DCT from down to up
         K=tril(K,-10);                                    % Divides the square DCT into a triangle with diagonal at -10
        
         DctToFace=reshape(K,1,TDctSize);
         IrisGal{k}=DctToFace;                            %store it in FaceGal
         k=k+1;                                           %increment
         TSum=double(TSum)+double(DctToFace);             %update
         TTotal=double(TTotal)+double(DctToFace);
            
        end
        
        Avg=(TSum./(NoOfTrImg));                             % mean of the coeff in each folder 
        classmeanarray{TotClasses}=reshape(Avg(1:XDct,1:YDct),1,TDctSize);
        TrainingTime(NoOfTimes)=toc;
    end
    TotAvg=TTotal./TTrImg;                                     % mean of total coeff in all the folders
    totalmeanarray=reshape(TotAvg(1:XDct,1:YDct),1,TDctSize);
    
    
%%%% BPSO %%%%%
    
    NumofParticles = 30;    %         
    NOPar = TDctSize;       %assigning the size of the gallery to NOPar 
    FitFn='fitnessc';       %fitness function
    GlobalBestP = rand(1,NOPar);  %it selects a random particle
    GlobalBestC = 0;              %initialized to 0
    MaxIterations =100;
    C2 = 2;
    C1 = 2;
      iner=0.6;   
    for m = 1:NumofParticles
        for NOP=1:NOPar
            PartPos(m,NOP)=randi([0,1]);     % particle position it randomly select
            PartVel(m,NOP)=0;                %particle velocity is initialized to 0
            PartBest(m,NOP)= PartPos(m,NOP); % initially assumes particle best as the partile position randomly selected
        end
        FBestP(m)= feval(FitFn,PartPos(m,:),TClasses,NOPar);  %evaluating the fitness function
    end
    [FGBestP,Index] = max(FBestP);
    GlobalBestP = PartBest(Index,:);
    
  
    
    for t = 1:MaxIterations                          % each loop runs 100 times 
        iner=iner.^t;
        for p = 1:NumofParticles                     % for loop runs 1 to 30
            FValue=feval(FitFn,PartPos(p,:),TClasses,NOPar); % again evaluates the fitness function
            if (FValue>FBestP(p))                         % compares 
                FBestP(p)=FValue;                         % if Fvalue is greater than FBestP then assigns to FBestP
                PartBest(p,:)=PartPos(p,:);               %particle Best is assigned the current particle pos value
            end
            
            [FGBestP,Index] = max(FBestP);
            GlobalBestP  = PartBest(Index,:);
            
            for tp=1:NOPar
                PartVel(p,tp)=(iner)*(PartVel(p,tp)) + C1 * (PartBest(p,tp) - PartPos(p,tp))* rand + ... % evaluating the particle velocity equation with the 
                    C2 * (GlobalBestP(1,tp) - PartPos(p,tp))* rand;                                       % contant values provided C1 &C2
                PartPos(p,tp) = (rand) < (1/(1 + exp(-PartVel(p,tp))));                                 % new particle pos value is calculated
            end
        end
    end
    
   
    
    Count=0;opti=0;
    for q=1:TDctSize
        if GlobalBestP(q)==1                    %if the value is one then there's a feature existing
            Count=Count+1;
            opti(Count)=q;
        end
    end
    disp(strcat('Number of selected features:',num2str(Count)));        %displays the number of features
    temp(NoOfTimes)=Count;
    
    for t= 1:TTrImg
        EyeGallery{t}(1:TDctSize)=0;
        for bpopti=1:Count
            EyeGallery{t}(opti(bpopti))= IrisGal{t}(opti(bpopti)).*GlobalBestP(opti(bpopti)); %if it evaluates to 0 then that feature is eliminated
            EyeGallery_size{t}(bpopti)=EyeGallery{t}((opti(bpopti)));                         %if not it will be selected and put to the optimized FaceGal
        end
         size(cell2mat(EyeGallery));
    end
   
    clear FaceGal

  ticID2  =  tic;
    rec=0;
    RImages=(TImgPerClass-NoOfTrImg);            %obtaining the testing images now
    tests=TClasses*RImages;                      %total testing images is the remaining images mul with Total classes
    c=1:TClasses;
    
    for tst=1:TClasses
        
        b2=[1:TImgPerClass];                     %calculates difference between training and testing images
        b1=setdiff(b2,b{c(tst)});
        
        for h12=1:RImages
          IR=imread(strcat(locstr,num2str(tst),'\',num2str(b1(h12),'%02i'),Formt)); % reading the testing images
      %% Sequential flow of  image pre-processing steps %%
      IR=imresize(IR,0.125);
      IR= rgb2gray(IR); 
      [IR or] = canny(IR, 1,10,4,7);
      IR=radon(IR);
      IR=abs(IR);
      IR=dct2(IR);
      K=IR(1:25,1:25);
      K=flipdim(K,1);
      K=tril(K,-10);
      DctToFace=reshape(K,1,TDctSize);                   % after reshaping these testing images again store in a gal Pic_dct
      Pic_dct=DctToFace;
                 
          TstImgGal=Pic_dct.*GlobalBestP;
        
        for bpopti=1:Count
            TstImgGal(bpopti)=Pic_dct(opti(bpopti));           % that is assigned to test imag gallery
        end
        d=zeros(1,TTrImg);
        for p=1:TTrImg
            for bpopti=1:Count
                d(p)=d(p)+((TstImgGal(bpopti)-EyeGallery_size{p}(bpopti)).^2);
            end
            d(p)=sqrt(d(p));                       %calculating the minimum mean square error
        end
        [val,index]=min(d);
        if((ceil(index/(NoOfTrImg)))==c(tst))
            rec=rec+1;                            % increment the recognition rate
                                
         end 
     end
 end
    TestingTime(NoOfTimes)=toc;
    percent=(rec/tests)*100;
    disp(strcat('Recognition rate:',num2str(percent)));
    percentsum(NoOfTimes)=percent;
    
end
fprintf(1, '\n');
disp(strcat('Average number of selected features: ',num2str(sum(temp)/max(NoOfTimes))));
disp(strcat('Average Recognition Rate: ',num2str(sum(percentsum)/max(NoOfTimes))));
disp(strcat('testing time: ',num2str((sum(TestingTime)/( max(NoOfTimes)*tests))/(2*224))));
disp(strcat('training time: ',num2str(sum(TrainingTime)/( max(NoOfTimes)*TClasses* NoOfTrImg))));
