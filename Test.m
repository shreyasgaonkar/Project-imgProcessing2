clc;
close all;
clear all;
B = load('C:\Users\Shreyas\Desktop\Image 2\KL_norm_train.dat');
C = load('C:\Users\Shreyas\Desktop\Image 2\KL_norm_train_2.dat');
Z = load('C:\Users\Shreyas\Desktop\Image 2\Test_image.dat');
A = [B,C];
%Initialized the Temp variables to zero%
temp5=0;
distvector=0;
temp6=0;
temp7=0;
temp9=0;
m=0;
c=0;

figure;    
   a = reshape(A,40,40,18); %Reshaping the original Image in 40x40 format for display
for n= 1:18
   subplot(3,6,n);
   imagesc(a(:,:,n));
   colormap(gray); %This function is used to convert the colored imaged into grayscale image 
end

for n=1:18
  m=m+A(:,n);   
end

%Displaying Average Faces%
avg = m/18;
figure;
suptitle('Average Face');
imagesc(reshape(avg,40,40));
colormap(gray); %This function is used to convert the colored imaged into grayscale image 

for n=1:18
    cov=(A(:,n)-avg)*(A(:,n)-avg)';
    c=c+cov;
end

[eigenvector,eigenvalue] = eig(c);

for n=1:18
    temp(:,n)=eigenvector(:,1582+n);
end

figure;
suptitle('Eigen Faces');

temp1 = reshape(temp(:,(1:18)),40,40,18);

for n= 1:18
       subplot(3,6,n);
       imagesc(temp1(:,:,n));
       colormap(gray); %This function is used to convert the colored imaged into grayscale image 
   
end

flip = fliplr(eigenvector(:,1589:1600));

for n=1:18
    
   weight(:,n)=(flip'*(A(:,n)-avg));
   recimg(:,n)=((flip*weight(:,n))+avg);
    
end

figure;
suptitle('Reconstructed Image');
temp2 = reshape(recimg(:,(1:18)),40,40,18); %Reconstructed Image is stored in temp2 just to reshape it while displaying

for n= 1:18
       subplot(3,6,n);
       imagesc(temp2(:,:,n));
       colormap(gray); %This function is used to convert the colored imaged into grayscale image 
   
end

figure;
suptitle('Test Image');
imagesc(Z);
colormap(gray); %This function is used to convert the colored imaged into grayscale image 
  
for i=1:121 %for1
    for j=1:121 %for2

%Window%
window(:,:)=Z(i:i+39,j:j+39);
M = mean(mean(window)); %Mean of the window
diff=window-M;
SD = sqrt(sum(sum(diff.^2)));
NW = diff/SD;
%Normalized Window will have DC Value of 0 and variance of 1

output=(reshape(NW,1600,1)-avg);
%y
for k=1:12
y(i,j,k)=dot(flip(:,k),output);
end

%error square
temp8=zeros(1,12);
for k=1:12
temp8(:,k)=(y(i,j,k).^2);

end
 temp9=sum(temp8); %temp9 is the sumation of y(i)^2 of 12 values   
errorsq=((norm(output).^2)-temp9);

%Lamba i for 12 values
temp3=eigenvalue(1583:1600,1583:1600);
temp4=zeros(1,18);
for i1=1:18
    for j1=1:18
        if i1 == j1
           temp7(:,i1)=temp3(i1,j1); %Lambda i
        end
    end
end

temp5=sum(temp7(:,13:18)); %Summation Lambda i for last 6 values
rho=(1/6)*(temp5);

%distance vector
temp12=(y(i,j,:).^2);

for k=1:12
    temp11(i,j)=sum((temp12)/fliplr(temp7(:,k)));
end

distvector(i,j)=abs(temp11(i,j)+(errorsq/rho));
    end% for1 ends
end% for2 ends

figure;
suptitle('Distance Vector');
imagesc(distvector);
colormap(gray); %This function is used to convert the colored imaged into grayscale image 


n=20;      
for i=1:121-n+1 %for1
    for j=1:121-n+1 %for2
window1(:,:)=distvector(i:i+n-1,j:j+n-1);
M1 = min(min(window1)); %Mean of the window
for i1=i:i+n-1
    for j1=j:j+n-1
        if (distvector(i1,j1)>M1)
            distvector(i1,j1)=255;
        end
    end
end
   end
end

figure;
suptitle('Co-ordinates');
imagesc(distvector);
colormap(gray); %This function is used to convert the colored imaged into grayscale image 

[row, col] = find(distvector<255);
rowcol=[row,col];
rowarranged=[23;21;21;62;61;62;102;101;101];
colarranged=[24;62;103;24;63;103;22;63;101];
rowcolarrange=[rowarranged,colarranged];

for i=1:9
img=Z(rowarranged(i):rowarranged(i)+39,colarranged(i):colarranged(i)+39);
M2=mean(mean(img));
diff1=img(:,:)-M2;
SD2=sqrt(sum(sum(diff1.^2)));
NW2(:,:,i)=diff1/SD2;
subplot(3,3,i);
imagesc(NW2(:,:,i));
colormap(gray);
end

for j2=1:9
oldweight(:,:,j2) = (A(:,j2)-avg)'*(fliplr(eigenvector(:,1595:1600)));
end
oldweight=squeeze(oldweight);
for j2=1:9
    newweight(j2,:)=((reshape(NW2(:,:,j2),1600,1))-avg)'*(fliplr(eigenvector(:,1595:1600)));         
    for k2=1:9
            final(j2,k2)=norm(newweight(j2,:)-(oldweight(:,k2))');
    end
    
end

disp(newweight);
disp(final);
figure;

for i1=1:9
    for j1=1:9
        subplot(3,3,i1);
        plot(final(i1,:)); %Distance of Test Image with Training Image
    end
end


