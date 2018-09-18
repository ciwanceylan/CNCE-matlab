function [ U, logPU ] = gNCEnoise( X, nu )
%GNCENOISE Function which generates noise based on Gaussian iid noise for NCE.

[N, D] = size(X);
Xmean = mean(X, 1);
Xcov = cov(X);
Y = mvnrnd(Xmean, Xcov, N*nu);
U = cat(1, X, Y);

partFun = -(D/2) * log(2*pi*det(Xcov));
Up = bsxfun(@minus, U, Xmean);
logPU = mrdivide(Up, Xcov);
logPU = -0.5 * dot(logPU, Up, 2); 
logPU = bsxfun(@plus, logPU, partFun);
end

