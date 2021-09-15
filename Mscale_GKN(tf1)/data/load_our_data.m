clc
clear all
close all
data = load('res258_a4_f1_train.mat');
ceof = data.meshA;
ACE = reshape(ceof(10,:,:),[258,258]);
New_ACE = reshape(ACE, [1,258*258]);

solu = data.meshU;
Asolu = reshape(solu(1,:,:),[258,258]); 

XYpoints = data.meshXY;
Xcoords = XYpoints(1,:);
Ycoords = XYpoints(2,:);
X = reshape(Xcoords, [258, 258]);
Y = reshape(Ycoords, [258, 258]);

figure('name','meshU')
surf(X, Y, Asolu)
hold on

figure('name', 'meshA')
surf(X, Y, ACE)
hold on

% figure('name', 'log_meshA')
% surf(X, Y, log(ACE))
% hold on

% πÈ“ªªØAeps
mean_value = mean(mean(ACE));
std_value = std2(ACE);

Gauss_ACE = (ACE-mean_value)./(std_value*std_value);

figure('name', 'Normalize_meshA')
surf(X, Y, Gauss_ACE)
hold on

