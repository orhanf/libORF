function numgrad = computeNumericalGradient(obj, J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

nParams = length(theta);
epsilon = 1e-4;

for i=1:nParams
    
    disp(['compute numerical gradient for theta(' num2str(i) ')']);
    
    e = zeros(nParams,1); e(i) = 1;
    theta_plus  = theta + epsilon .* e;
    theta_minus = theta - epsilon .* e;
    
    numgrad(i) = ( J(theta_plus) - J(theta_minus) ) ./ (2 * epsilon);
        
end

end
