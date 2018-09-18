function [ sqEstimError] = calcErr_ica(Btrue,Bhat)
%CALCERR_ICA Calculates the error for the ICA model
%   Because the GT parameters for the ICA model is ambigious in sign and 
%   permutation of the rows.
%   To resolve  the ambiguity we find the permutation matrix P which 
%   minimises ||Btrue - P*Bhat||_2

    % Get non-permutation matrix solution D
    % D = Bhat*inv(Btrue);
    D = Bhat/Btrue;
    
    % Trun D into a row permutation matrix including sign info
    [~, index]=max(abs(D));
    dim = size(D,1);
    P = zeros(dim);
    for k=1:dim
        signum = sign(D(index(k),k));
        P(k,index(k)) = 1*signum;
    end

    % resolve order and sign ambiguity
    BhatPerm = P*Bhat;

    % calculate error
    estimError = Btrue-BhatPerm;
    sqEstimError = sum(estimError(:).^2);
end  

