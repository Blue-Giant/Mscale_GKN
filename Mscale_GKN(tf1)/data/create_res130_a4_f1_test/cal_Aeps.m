clc
clear all
close all
q = 7;
N = 2^q+1;

xpoints = -1:2/N:1;
ypoints = -1:2/N:1;
[meshX, meshY] = meshgrid(xpoints, ypoints);
Xcoords = reshape(meshX, [1, (N+1)*(N+1)]);
Ycoords = reshape(meshY, [1, (N+1)*(N+1)]);
meshXY = [Xcoords; Ycoords];
Aeps = LXA_calA(meshXY, q, 4);
save('Aeps.mat','Aeps');

save('meshXY.mat','meshXY');