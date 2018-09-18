function R = genRingData(N, D, mu, sig, cutoff)
%GENRINGDATA Generates data on a lower-dimensional manifold
%   Data is Gaussian distributed along the radial direction with a
%   non-zero mean "mu" and std "sig"
%   To avoid negative values for the radius, the Gaussian is truncated
%   at a small positive value "cutoff"

% Sample normal distributed data    
R = randn(N, D);
% Project onto sphere to create uniform distr over sphere
R = bsxfun(@rdivide, R, sqrt(sum(R.^2, 2)));
% Samples radii from a Gaussian distribution
r = sig * randn(N, 1) + mu;

% While there are radii which are too small, resample these
rPosInx = find(r < cutoff);
while ~isempty(rPosInx)
    r(rPosInx) = sig * randn(length(rPosInx), 1) + mu;
    rPosInx = find(r < cutoff);
end

% Assign new radii to the points on the sphere
R = bsxfun(@times, r, R);

end

