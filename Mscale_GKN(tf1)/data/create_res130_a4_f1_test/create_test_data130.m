clc;
clear all
close all
q = 7;
N = 2^q+1;
M = N + 1;

xpoints = -1:2/N:1;
ypoints = -1:2/N:1;
[meshX, meshY] = meshgrid(xpoints, ypoints);
Xcoords = reshape(meshX,[1,M*M]);
Ycoords = reshape(meshY,[1,M*M]);
xy_points = [Xcoords; Ycoords];
original_index = linspace(1,M*M,M*M);
disorder_index = reorder_index(original_index, M);
meshXY = xy_points;

data2solu = load('utrue7.mat');
order_solu = data2solu.utrue;

meshU = reshape(order_solu, [M, M]);
figure('name', 'meshU')
surf(meshX, meshY, meshU)
hold on

data2eps = load('Aeps.mat');
uniform_A = data2eps.Aeps;
meshA = reshape(uniform_A, [M,M]);
figure('name', 'meshA')
surf(meshX, meshY, meshA)
hold on

disorderA = reorder_index(uniform_A, M);
mesh_disA = reshape(disorderA, [M, M]);
figure('name', 'mesh_disA')
surf(meshX, meshY, mesh_disA)
hold on

save('res130_a4_f1_test.mat','meshXY','meshA', 'meshU');

save('disorder_index130.mat','disorder_index');