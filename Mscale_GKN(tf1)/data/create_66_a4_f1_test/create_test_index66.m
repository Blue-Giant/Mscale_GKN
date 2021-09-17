clc;
clear all
close all
q = 6;
N = 2^q+1;
M = N+1;

original_index = int32(linspace(0,M*M-1,M*M));
disorder_index = int32(reorder_index(original_index, M));


save('disorder_index66.mat','disorder_index');