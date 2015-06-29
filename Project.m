clc;
clear all;
close all;
%% Load data from external files
training_faces_col = [reshape(load('C:\Users\Shreyas\Desktop\Image 2\KL_norm_train.dat'),1600,9) 
reshape(load('C:\Users\Shreyas\Desktop\Image 2\KL_norm_train_2.dat'),1600,9)];
training_faces_sqr = reshape(training_faces_col,40,40,18);
training1 = reshape(load('C:\Users\Shreyas\Desktop\Image 2\KL_norm_train.dat'),1600,9);
figure;
hold on;
suptitle('Eighteen Faces');
for i = 1:18

 

subplot(3,6,i);

 

colormap('gray');

 

imagesc(training_faces_sqr(:,:,i));

 

axis off;

 

hold on;

 

end

 

%% Calculate Mean Vector and display average face

 

m_x = mean(training_faces_col,2);

 

I_avg = reshape(m_x,40,40);

 

figure;

 

colormap('gray');

 

imagesc(I_avg);

 

suptitle('Average Face');

 

%% Find Difference Faces

 

difference_faces = training_faces_sqr - repmat(I_avg,[1,1,18]);

 

figure;

 

for i = 1:18

 

subplot(3,6,i);

 

colormap('gray');

 

imagesc(difference_faces(:,:,i));

 

axis off;

 

hold on;

 

end;

 

suptitle('Difference Faces');

 

%% Calculate Covariance Matrix

 

C_x = zeros(1600,1600);

 

for i = 1:18

 

C_x = C_x + 1/18 * (training_faces_col(:,i) - m_x) * (training_faces_col(:,i)- m_x)';

 

end;

 

%% Determine Eigenvalue and Eigenvectors, and reform faces

 

[eigen_vector,eigen_value] = eig(C_x);

 

eigen_vector = eigen_vector(:,1:18);

 

eigen_value = diag(eigen_value(1:18,1:18));

 

eigen_vector_sqr = reshape(eigen_vector,40,40,18);

 

kleig = reshape(eigen_vector,1600,18);

 

[U,D,V] = svd(C_x,0);

 

%% Plot Eigenfaces with full dynamic range

 

figure;

 

for i = 1:18

 

subplot(3,6,i);

 

colormap('gray');

 

imagesc(abs(eigen_vector_sqr(:,:,i)));

 

axis off;

 

hold on;

 

end;

 

suptitle('EigenFaces');

 

%% Reconstruct training faces with 6 of the 9 eigenvectors

 

% the reconstruction of the images is carried out by finding the KL

 

% coefficients, from difference images and the Eigen images and values here

 

% we are using only 6 Eigen images

 

num_eigen = 6;

 

for i=1:18

 

Coef_Mat(i,:) = reshape(difference_faces(:,:,i),1600,1)' * reshape(eigen_vector_sqr,1600,18);

 

new_face = zeros(40,40);

 

for j=1:num_eigen

    

klref(j,i) = kleig(:,j)' * reshape(difference_faces(:,:,i),1600,1);

 

new_face = new_face + (eigen_vector_sqr(:,:,j)*(Coef_Mat(i,j)));

 

end;

 

rebuilt_face(:,:,i) = new_face + I_avg;

 

end;

 

KLcoefficients = transpose(klref);

 

figure;

 

for i=1:18

 

subplot(3,6,i);

 

colormap('gray');

 

axis off;

 

imagesc(rebuilt_face(:,:,i));

 

end;

 

suptitle('Reconstructed Images');

 

%% Load test image from external file

 

test_image = load('Test_image.dat');

 

figure;

 

suptitle('Test Image');

 

colormap('gray');

 

imagesc(test_image);

 

%% Scale test image with 40x40 window & Detect faces within the normalized 

window

 

num_eigen = 6;

 

y = zeros(120,120,num_eigen);

 

d = zeros(120,120);

 

d_total = zeros(120,120);

 

figure;

 

test_image_DC = mean2(test_image);

 

test_image_std = std2(test_image-test_image_DC);

 

test_image_norm = (test_image - test_image_DC)/test_image_std;

 

 

for l = 1:160-39 %test image rows

 

for m = 1:160-39 %test image cols

 

test_image_window_norm = test_image_norm(l:(l+39),m:(m+39));

 

for i = 1:num_eigen

 

y(l,m,i) = dot(eigen_vector(:,i),reshape(test_image_window_norm-I_avg,1600,1));

 

end;

 

epsilon_sqrd = abs(norm((test_image_window_norm - I_avg).^2) - sum(y(l,m,:).^2));

 

p = 1/(18-num_eigen) * sum(abs(eigen_value((num_eigen+1):end)));

 

d(l,m) = sum(abs(squeeze(y(l,m,:))).^2 ./ abs(eigen_value(1:num_eigen)));

 

d(l,m) = d(l,m) + epsilon_sqrd/p;

 

end;

 

end

 

axis off;

 

imagesc(d);

suptitle('Distance Image');

colormap('bone');

 

%% Locate face coordinates

 

figure;

 

small_window_length = 25;

 

d_total = d;

 

suptitle('Face Coordinates');

 

for i = 1:(121-small_window_length)

 

for j = 1:(121-small_window_length)

 

d_window = reshape(d_total(i:i+small_window_length,j:j+small_window_length),(small_window_length+1)^2,1);

 

wmin= min(d_window);

 

for l = i:i+small_window_length

 

for m = j:j+small_window_length

 

if d_total(l,m)> wmin

 

d_total(l,m) = 100;

 

end;

 

end;

 

end;

 

end;

 

end;

 

axis off;

 

imagesc(d_total);

 

colormap('bone');

 

%% Display found faces

 

[x_d,y_d] = find(d_total < 100);

 

[junk,sort_index] = sort(x_d);

 

x_d = x_d(sort_index);

 

y_d = y_d(sort_index);

 

[junk,sort_index] = sort(y_d(1:3));

 

x_d(1:3) = x_d(sort_index);

 

y_d(1:3) = y_d(sort_index);

 

[junk,sort_index] = sort(y_d(4:6));

 

x_d(4:6) = x_d(sort_index+3);

 

y_d(4:6) = y_d(sort_index+3);

 

[junk,sort_index] = sort(y_d(7:9));

 

x_d(7:9) = x_d(sort_index+6);

 

y_d(7:9) = y_d(sort_index+6);

 

figure;

 

found_face = zeros(40,40,9);

 

for i = 1:9

 

subplot(3,3,i);

 

test_image_window = test_image(x_d(i) + (0:39) , y_d(i) + (0:39));

 

test_image_window_DC = mean2(test_image_window);

 

test_image_window_std = std2(test_image_window);

 

found_face(:,:,i) = (test_image_window - test_image_window_DC)/test_image_window_std;

 

colormap('gray');

 

imagesc(found_face(:,:,i));

 

axis off;

 

end;

 

suptitle('found faces');

 

%% calculating the difference matrix for the detected images.

 

diff_face = zeros(40,40,9);

 

figure;

 

for i=1:9

 

diff_face(:,:,i)= found_face(:,:,i) - I_avg/std2(I_avg);

 

subplot(3,3,i);

 

colormap('gray');

 

imagesc(diff_face(:,:,i));

 

axis off;

 

end;

 

suptitle('Difference images of the detected faces');

 

%% KL transform faces

 

determine_face = zeros(9,9);

 

for i=1:9

 

determine_face(i,:) = reshape(diff_face(:,:,i),[1600,1])' * squeeze(training1);

 

end;