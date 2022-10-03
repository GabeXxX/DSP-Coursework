%% section 1
%First version, rho fixed at start
M = 10;
mu = zeros(M,1);
rho_fin = linspace(0,1,10);
k = linspace(10,200,11);
mse = zeros(length(rho_fin),1);

s=zeros(length(rho_fin), 20);
rho=0.5;
m = rho*ones(M-1,1);
c = cat(1,1,m);
C = toeplitz(c);

 for kk = 1:length(k) 
 
    for ii=1:length(rho_fin)
        rho=rho_fin(ii);
        m = rho*ones(M-1,1);
        c = cat(1,1,m);
        Ci = toeplitz(c);      
        w_k = mvnrnd(mu,Ci,k(kk));
        C_est = cov(w_k);
        mse(ii) = mean((C-C_est).^2, 'all'); 
    end
    figure(1); hold on;
    plot(rho_fin, mse);
        
 end

%% section 2
%Second version, rho NOT fixed at start
M = 10;
mu = zeros(M,1);
rho_fin = linspace(0,1,10);
k = linspace(10,200,11);
mse = zeros(length(rho_fin),1);
s=zeros(length(rho_fin), 20);

 for kk = 1:length(k) 
 
    for ii=1:length(rho_fin)
        rho=rho_fin(ii);
        m = rho*ones(M-1,1);
        c = cat(1,1,m);
        C = toeplitz(c);      
        w_k = mvnrnd(mu,C,k(kk));
        C_est = cov(w_k);
        mse(ii) = mean((C-C_est).^2, 'all'); 
    end
    
    figure(1); hold on;
    plot(rho_fin, mse);
        
 end
 

 %% 1.1 − Noise Generation
 clc;
 clear all;
 
 M=4;
 MC = 10; % Montecarlo Iterations 
 MSE_H = zeros(10,10,MC);
 for iter = 1:MC % iterate through Montecarlo 
     i=1;
     u = randn ([M,200]); % returns a Mx200 matrix containing ”random” values from normal distrib 
     for L = floor ( linspace (10 ,200 ,10)) % iterate through L
     k=1;
     for rho = linspace (0 , 0.99 ,10) %iterate through values of rho 
         Cw_true = toeplitz ([1 ,rho ,rho ,rho ]);
         % true Covariance Matrix , it makes like sum of diags in a toeplitz form 
         w = chol(Cw_true, 'lower')*u(:,1:L);
         % noise , chol gives a lower triangular matrix drawn from Cw true
         Cw_est = cov(w.');
         % estimate of the Covariance Matrix (we put the transpose since for cov each column is a variable , each 
         MSE_H(k, i, iter) = mean(mean(Cw_true - Cw_est).^2); % .ˆ since element wise,
         %mean of mean compute the mean of the vector of the mean
         k = k + 1; % to increment index associated to rho 
     end
         i = i + 1; % to increment index associated to L 
    end
 end
     MSE_H = mean(MSE_H,3); % mean for each combination of x and y of MSE
     figure ()
     bar3(MSE_H);
     xlabel('L');
     ylabel('\rho');
     zlabel('MSE') % 3D PLOT
     figure ()
     for l = 1:10 % iterate through L, to have different lines of MSE v. rho wrt L
         plot(linspace(0, 0.99, 10), MSE_H(:,l))
         legend('L=10','L=31','L=52','L=73','L=94','L=115','L=136','L=157','L=178','L=200')
         hold on
         
     end
xlabel('\rho');
ylabel('MSE')
title('MSE vs \rho')
figure ()
for rho = 1:10 % iterate through rho, to have different lines of MSE v. L wrt rho 
    plot ( floor ( linspace (10 ,200 ,10)) , MSE_H(rho , :))
    legend('\rho = 0', '\rho = 0.11','\rho = 0.22','\rho = 0.33','\rho = 0.44','\rho = 0.55','\rho = 0.66','\rho = 0.77','\rho = 0.88','\rho = 0.99')
         
    hold on;
end
xlabel('L');
ylabel('MSE')
title('MSE vs L')