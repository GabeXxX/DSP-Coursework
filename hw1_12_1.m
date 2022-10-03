%% section 1
clc
clear all

N = 4;
M = 4;
K = 4;
Q = 50%ceil(linspace(13, 50, 5)); % Q>N*K+1-K 
ALPHA = 0.99%linspace(0, .99, 5);
SNR = -10 : 2:  30; % SNR in dB
SNRlin = 10.^(SNR/10);
ITER = 100; % monte-carlo iteration
BETA = [0.9, 0.5, 0.1]; % BETA = 0.5;
RHO=0.5;

m = RHO*ones(M-1,1);
c = cat(1,1,m);
C = toeplitz(c); % covariance matrix for noise

sigma_x = 1;
mu = zeros(N,1);
In = eye(N);

Im = eye(M);

nexttile
for iBETA = 1: length(BETA)
    beta = BETA(iBETA);
    
    for iALPHA = 1:length(ALPHA)
        alpha=ALPHA(iALPHA);
        
        % build the vector h and reshape it in column h_col
        h = zeros(M,N,K);
        for k=1:K
            for i=1:M
                for j = 1:N
                    h(i,j,k) = alpha^(abs(i-j))*beta^(k-1);
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
        
                       
        for iQ = 1:length(Q)
            q = Q(iQ);
            
            % build covariance matrix for noise
            Ic = eye(q+K-1);
            Cw = kron(C, Ic);
            
            CRBmat = zeros(M*N*K , N*M*K);
            MSE = zeros(size(SNR));
            for iSNR = 1: length(SNR)
                
                snr=SNR(iSNR);
                x=10^(snr/20)*mvnrnd(mu,sigma_x*In,q); 
                
                % build the convolution matrix X
                X_conv = zeros(q+K-1, N*K);
                X = zeros(q+K-1, K);
                l=1;
                for i=1:N
                    xi = x(:,i);
                    for j = 1:K
                        X(j:j+q-1, j) = xi; 
                    end 
                    X_conv(:,l: l+K-1) = X; 
                    l = l+K;
                end
                X_conv = kron(eye(M), X_conv);
               
                % start monte-carlo iterations
                for iITER = 1: ITER
                    w = mvnrnd(zeros(M*(q+K-1),1),Cw,1)';
                    y = X_conv*h_col+ w;  % signal generation 
                    h_est = inv((X_conv'*inv(Cw)*X_conv))*X_conv'*inv(Cw)*y; % estimation
                    err = h_est-h_col; % metric evaluation
                    MSE(iSNR)=MSE(iSNR)+(err'*err)/(N*M*K)/ITER; 
                end
              
            end
            figure(1);
            CRB=10.^(-SNR/10)/q;  % asymptotic CRB           
%             title('MSE vs SNR (alpha = 0.99, Q varies, , beta=0.9)');
%             semilogy(SNR,CRB,'-.',SNR,MSE,'-');
%             xlabel('SNR [dB]'); ylabel('MSE channel estimate');
%             legend('CRB (Q=13)','MSE (Q=13)','CRB (Q=23)','MSE (Q=23)','CRB (Q=32)','MSE (Q=32)','CRB (Q=41)','MSE (Q=41)','CRB (Q=50)','MSE (Q=50)');
%             hold on;
        end
        %hold off;
        %nexttile;         
%             title('MSE vs SNR (alpha varies, Q= 50, beta=0.9)');
%             semilogy(SNR,CRB,'-.',SNR,MSE,'-');
%             hold on;
%             xlabel('SNR [dB]'); ylabel('MSE channel estimate');
%             legend('CRB (alpha=0)','MSE (alpha=0)','CRB (alpha=0.2475)','MSE (alpha=0.2475)','CRB (alpha=0.4950)','MSE (alpha=0.4950)','CRB (alpha=0.7425)','MSE (alpha=0.7425)','CRB (alpha=0.9900)','MSE (alpha=0.9900)');
    end
    %hold off
    title('MSE vs SNR (alpha = 0.99, Q=50, beta varies)');
            semilogy(SNR,CRB,'-.',SNR,MSE,'-');
            xlabel('SNR [dB]'); ylabel('MSE channel estimate');
            legend('CRB (beta=0.90)','MSE (beta=0.90)','CRB (beta=0.50)','MSE (beta=0.50)','CRB (beta=0.10)','MSE (beta=0.10)');
            hold on;
end
hold off