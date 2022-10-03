%% section 1
clc
clear all

N = 4;
M = 4;
K = 1;
P = 200;
Q = [5,13, 33, 63, 102, 151, 199]; %Q = ceil(linspace(13, P, 20)); % Q>N*K+1-K 
ALPHA = 0.5;
SNR = [-10 : 2:  30]; % SNR in dB
ITER = 10; % monte-carlo iteration
BETA = 0.5; %[0.9, 0.5, 0.1]; % BETA = 0.5;
RHO=0.1;

m = RHO*ones(M-1,1);
c = cat(1,1,m);
C = toeplitz(c);

sigma_X2 = 10.^(SNR/10);

% build the vector h and reshape it in column h_col
h = zeros(M,N,K);
for k=1:K
    for i=1:M
        for j = 1:N
            h(i,j,k) = ALPHA^(abs(i-j))*BETA^(k-1);
        end
    end
end            
j=1;
h_col = zeros(M*N*K,1);
for i = 1:M
    for l =1:N
        h_col(j:j+K-1) = h(i,l,:);
        j = j+K;
    end
end

mu = zeros(N,1);
sigma_x = 1;
In = eye(N);

x=mvnrnd(mu,sigma_x*In,P);

H_conv = kron(h, eye(P));

%x_vect = reshape(x,M*(P),1);
%y = H_conv*x_vect
nexttile
for iQ = 1:length(Q)
    q = Q(iQ);
    
    Iq = eye(q+K-1);
    
    Ip_q = eye(N*(P-q));
    
    x_q = x(1:q, :);
    x_q_p = x(q+1: end, :);
    
    % build the convolution matrix X with only the X(1)...X(Q)
    X_conv_q = zeros(q+K-1, N*K);
    X = zeros(q+K-1, K);
    l=1;
    for i=1:N
        xi = x_q(:,i);
        for j = 1:K
            X(j:j+q-1, j) = xi;
        end
        X_conv_q(:,l: l+K-1) = X;
        l = l+K;
    end
    X_conv_q = kron(eye(M), X_conv_q);
            
    MSE_mle = zeros(size(SNR));
    MSE_mmse = zeros(size(SNR));

    for iITER = 1: ITER %montecarlo
        
        for iSNR = 1: length(SNR)
            C = toeplitz(c)/sigma_X2(iSNR);
            Cw = kron(C, Iq);

            Ic1 = eye(P-q);
            Cw1 = kron(C, Ic1);
            
            
            %w = chol(C) * sqrt(1/iSNR)*randn(M*(q+K-1),q)';
            w = sqrt(1/iSNR)*mvnrnd(zeros(M*(q+K-1),1),Cw,1)';
            y_q = X_conv_q*h_col + w;
            h_est = inv((X_conv_q'*inv(Cw)*X_conv_q))*X_conv_q'*inv(Cw)*y_q;
            
            H_est = zeros(M,N,K);
            j=1;
            for i = 1:M
                for l =1:N
                    H_est(i,l,:) = h_est(j:j+K-1);
                    j = j+K;
                end
            end
            
            w = mvnrnd(zeros(M*(P-q),1),Cw1,1)';
            H_est_conv = kron(H_est, eye(P-q));
            y_q_p = H_est_conv*reshape(x_q_p,M*(P-q),1)+ w(1:M*(P-q));
            
            x_est_mle = inv(H_est_conv'*inv(Cw1)*H_est_conv)*H_est_conv'*inv(Cw1)*y_q_p;
            x_est_mmse = inv(H_est_conv'*inv(Cw1)*H_est_conv + inv(Ip_q))*H_est_conv'*inv(Cw1)*y_q_p;
            
            err_mle = x_est_mle-reshape(x_q_p,M*(P-q),1);
            err_mmse = x_est_mmse-reshape(x_q_p,M*(P-q),1); % metric evaluation
                    
            MSE_mle(iSNR)= MSE_mle(iSNR)+(err_mle'*err_mle)/(N*(P-q))/ITER;
            MSE_mmse(iSNR)= MSE_mmse(iSNR)+(err_mmse'*err_mmse)/(N*(P-q))/ITER;
           
                                  
        end
        
    end
    CRB=10.^(-SNR/10)/(P-q);  % asymptotic CRB
    figure(1);
    title('MMSE vs MLE');
    %semilogy(SNR,CRB,'-.',SNR,MSE_mle,'-')
    %semilogy(SNR,CRB,'-.',SNR,MSE_mle,'-',SNR,MSE_mmse,'--');
    semilogy(SNR,MSE_mle,'-',SNR,MSE_mmse,'--');
    hold on;
    xlabel('SNR [dB]'); ylabel('MSE channel estimate');
    legend('MLE (Q=5)','MMSE (Q=5)','MLE (Q=13)','MMSE (Q=13)','MLE (Q=33)','MMSE (Q=33)','MLE (Q=63)','MMSE (Q=63)','MLE (Q=102)','MMSE (Q=102)','MLE (Q=151)','MMSE (Q=151)','MLE (Q=199)','MMSE (Q=199)');
    
end


hold off
