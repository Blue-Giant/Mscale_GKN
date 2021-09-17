clc
clear all
close all
N = 130;
data = load('res130_a4_f1_test.mat');
ceof = data.meshA;
solu = data.meshU; 

XYpoints = data.meshXY;
Xcoords = XYpoints(1,:);
Ycoords = XYpoints(2,:);
X = reshape(Xcoords, [N, N]);
Y = reshape(Ycoords, [N, N]);

figure('name','meshU')
surf(X, Y, solu)
hold on

figure('name', 'meshA')
surf(X, Y, ceof)
hold on

% figure('name', 'log_meshA')
% surf(X, Y, log(ACE))
% hold on

% πÈ“ªªØAeps
mean_value = mean(mean(ceof));
std_value = std2(ceof);

Gauss_ceof = (ceof-mean_value)./(std_value*std_value+5.0);

figure('name', 'Normalize_meshA')
surf(X, Y, Gauss_ceof)
hold on

