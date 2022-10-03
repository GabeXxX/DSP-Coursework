
%% section 1
N = 4;
M = 4;
K = 2;
Q = ceil(linspace(13, 50, 5)); % Q>N*K+1-K 
ALPHA = linspace(0, .99, 5);
SNR = [-10 : 2:  30]; % SNR in dB
ITER = 100; % monte-carlo iteration
BETA = 0.5; %[0.9, 0.5, 0.1]; % BETA = 0.5;
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
        
        % build the vector h
        h = zeros(M,N,K);
        for k=1:K
            for i=1:M
                for j = 1:N
                    h(i,j,k) = alpha^(abs(i-j))*beta^(k-1);
                end
            end
        end
        h = reshape(h, [M*N*K,1]);       
        
        for iQ = 1:length(Q)
            q = Q(iQ);
            
            % build covariance matrix for noise
            Ic = eye(q+K-1);
            Cw = kron(Ic, C);
            
            MSE = zeros(size(SNR));
            for iSNR = 1: length(SNR)
                
                snr=SNR(iSNR);
                x=10^(snr/20)*mvnrnd(mu,sigma_x*In,q);
                
                % build the convolution matrix X
                X_conv = zeros(M*(q+K-1), M*N*K);
                tmp = zeros(M,1);
                tmp(1) = 1;
                x0 = kron(x, tmp);
                x0 = x0(1:size(x0, 1)-(M-1), :);
                j = 1;
                for i=1:M*K
                    X_conv(i:i+M*q-(M), j:j+N-1)= x0;
                    j = j + N;
                end
               
                % start monte-carlo iterations
                for iITER = 1: ITER
                    w = mvnrnd(zeros(M*(q+K-1),1),Cw,1)';
                    y = X_conv*h+ w;  % signal generation 
                    h_est = inv((X_conv'*inv(Cw)*X_conv))*X_conv'*inv(Cw)*y; % estimation
                    err = h_est-h; % metric evaluation
                    MSE(iSNR)=MSE(iSNR)+(err'*err)/(ITER*M*N*K); %MSE(iSNR)= MSE(iSNR)+(err'*err); % MSE(iSNR)=MSE(iSNR)+(err'*err)/(ITER*M*N*K);%
                end
                %MSE(iSNR)= MSE(iSNR)/ITER;
            end
            
            CRB=10.^(-SNR/10)/q;  % asymptotic CRB
            figure(1);
            title('Alpha',alpha);
            semilogy(SNR,CRB,'-.',SNR,MSE,'-');
            hold on;
            xlabel('SNR [dB]'); ylabel('MSE channel estimate');
        end
        nexttile;
    end  
    
end
hold off

