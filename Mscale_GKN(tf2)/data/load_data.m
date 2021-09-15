clc
% clear all
% close all
data = load('piececonst_r241_N1024_smooth1.mat');
ceof = data.coeff;
ACE = reshape(ceof(1,:,:),[241,241]);

solu = data.sol;
Asolu = reshape(solu(1,:,:),[241,241]); 

Kcoef = data.Kcoeff;
AK = reshape(Kcoef(1,:,:),[241,241]);

KceofX = data.Kcoeff_x;
AKX = reshape(KceofX(1,:,:),[241,241]); 