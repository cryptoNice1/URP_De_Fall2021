clc
clear
close all

%% Import data
load XY_Nov1.mat
X1 = X;
Y1 = Y2;
load XY_Nov6.mat
X2 = X;
Y2_2 = Y2;
load XY_Nov8.mat
X3 = X;
Y3 = Y2;
load XY_Oct1.mat
X4 = X;
Y4 = Y2;
load XY_Oct4.mat
X5 = X;
Y5 = Y2;
load XY_Oct18.mat
X6 = X;
Y6 = Y2;
load XY_Oct22.mat
X7 = X;
Y7 = Y2;
load XY_Oct23.mat
X8 = X;
Y8 = Y2;
load XY_Oct25.mat
X9 = X;
Y9 = Y2;
load XY_Oct30.mat
X10 = X;
Y10 = Y2;
load XY_Sep25.mat
X11 = X;
Y11 = Y2;
load XY_Sep27.mat
X12 = X;
Y12 = Y2;

% Putting data in the same burn groups together
X_200F10s = [X11, X12];
Y_200F10s = [Y11, Y12];

X_200F20s = [X4, X5];
Y_200F20s = [Y4, Y5];

X_200F30s = [X6, X7];
Y_200F30s = [Y6, Y7];

X_200F40s = [X8, X9, X10];
Y_200F40s = [Y8, Y9, Y10];

X_200F50s = [X1, X2, X3];
Y_200F50s = [Y1, Y2_2, Y3];

%% Polynomial background correction
[~,b1] = size(X_200F10s);
[~,d1] = size(X_200F10s);

lb1=round(min(X_200F10s(:,1)));
ub1=round(max(X_200F10s(:,1)));
l1=length(X_200F10s(:,1));

for i = 1:b1
    BG1 = backcor(linspace(lb1,ub1,l1)',Y_200F10s(:,i)',5,0.001,'atq');
    region1_bg(:,i) = BG1;
    Y_200F10s_rmbg_split(:,i) = Y_200F10s(:,i) - BG1;
end

[~,b2] = size(X_200F20s);
for i = 1:b2
    BG2 = backcor(linspace(lb1,ub1,l1)',Y_200F20s(:,i)',5,0.001,'atq');
    Y_200F20s_bg(:,i) = BG2;
    Y_200F20s_rmbg_split(:,i) = Y_200F20s(:,i) - BG2;
end

[~,b3] = size(X_200F30s);
for i = 1:b3
    BG3 = backcor(linspace(lb1,ub1,l1)',Y_200F30s(:,i)',5,0.001,'atq');
    Y_200F30s_bg(:,i) = BG3;
    Y_200F30s_rmbg_split(:,i) = Y_200F30s(:,i) - BG3;
end
    
[~,b4] = size(X_200F40s);
for i = 1:b4
    BG4 = backcor(linspace(lb1,ub1,l1)',Y_200F40s(:,i)',5,0.001,'atq');
    Y_200F40s_bg(:,i) = BG4;
    Y_200F40s_rmbg_split(:,i) = Y_200F40s(:,i) - BG4;
end    

[~,b5] = size(X_200F50s);
for i = 1:b5
    BG5 = backcor(linspace(lb1,ub1,l1)',Y_200F50s(:,i)',5,0.001,'atq');
    Y_200F50s_bg(:,i) = BG5;
    Y_200F50s_rmbg_split(:,i) = Y_200F50s(:,i) - BG5;
end

%% Normalize data.
Y_200F10s_rmbg_norm = Y_200F10s_rmbg_split./repmat(sqrt(sum(Y_200F10s_rmbg_split.^2)),length(X_200F10s(:,1)),1);
Y_200F20s_rmbg_norm = Y_200F20s_rmbg_split./repmat(sqrt(sum(Y_200F20s_rmbg_split.^2)),length(X_200F20s(:,1)),1);
Y_200F30s_rmbg_norm = Y_200F30s_rmbg_split./repmat(sqrt(sum(Y_200F30s_rmbg_split.^2)),length(X_200F30s(:,1)),1);
Y_200F40s_rmbg_norm = Y_200F40s_rmbg_split./repmat(sqrt(sum(Y_200F40s_rmbg_split.^2)),length(X_200F40s(:,1)),1);
Y_200F50s_rmbg_norm = Y_200F50s_rmbg_split./repmat(sqrt(sum(Y_200F50s_rmbg_split.^2)),length(X_200F50s(:,1)),1);

%% Reading data into another file
F10s = Y_200F10s_rmbg_norm.';
F10s = [ones(size(F10s,1),1) F10s];

F20s = Y_200F20s_rmbg_norm.';
F20s = [ones(size(F20s,1),1) F20s];

F30s = Y_200F30s_rmbg_norm.';
F30s = [ones(size(F30s,1),1) F30s];

F40s = Y_200F40s_rmbg_norm.';
F40s = [ones(size(F40s,1),1) F40s];

F50s = Y_200F50s_rmbg_norm.';
F50s = [ones(size(F50s,1),1) F50s];

writematrix([F10s;F20s;F30s;F40s;F50s],'RamanData.txt')
