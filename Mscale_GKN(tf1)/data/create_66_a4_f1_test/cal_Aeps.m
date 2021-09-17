clc
clear all
close all
q = 6;
N = 2^q+1;

xpoints = -1:2/N:1;
ypoints = -1:2/N:1;
[meshX, meshY] = meshgrid(xpoints, ypoints);
Xcoords = reshape(meshX, [1, (N+1)*(N+1)]);
Ycoords = reshape(meshY, [1, (N+1)*(N+1)]);
xy_points = [Xcoords; Ycoords];
Aeps = LXA_calA(xy_points, q, 4);