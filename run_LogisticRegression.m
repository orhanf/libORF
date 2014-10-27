%% load data

x = load('data/ex4x.dat');
y = load('data/ex4y.dat');

% find returns the indices of the
% rows meeting the specified condition
pos = find(y == 1); neg = find(y == 0);

% Assume the features are in the 2nd and 3rd
% columns of x
plot(x(pos, 1), x(pos,2), '+'); hold on
plot(x(neg, 1), x(neg,2), 'o')


%% initialtize logistic regressor, apply feature scaling 

logReg1 = LogisticRegressor(struct('x',x,'y',y,'nIter',150,'alpha',1));
 
logReg2 = LogisticRegressor(struct('x',x,'y',y,'nIter',15));


%% test libORF for gradient descent without regularization

[J theta_gd] = logReg1.gradient_descent();

theta_gd

logReg1.predict_samples([1;20;80], theta_gd)


%% apply newton's method without regularization

[J theta_newton] = logReg2.newtons_method();

theta_newton

logReg2.predict_samples([1;20;80],logReg1.theta_newton)

% Plot Newton's method result 
% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(x(:,1))-2,  max(x(:,1))+2];

% Calculate the decision boundary line
plot_y = (-1./theta_newton(3)).*(theta_newton(2).*plot_x +theta_newton(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off
pause

% Plot J
figure
plot(0:14, J, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J')
pause


%% test libORF for L2 regularized logistic regression
clear all, close all, clc;

addpath(genpath('misc'));

x = load('data/ex5Logx.dat'); 
y = load('data/ex5Logy.dat');

u = x(:,1);
v = x(:,2);
x = map_feature(u, v);

logReg  = LogisticRegressor(struct('x',x,'y',y,'addBias',false,'alpha',1,'silent',true));

logReg.lambda = 0;
logReg.gradient_descent_L2();
norm(logReg.theta_gd)

logReg.lambda = 1;
logReg.gradient_descent_L2();
norm(logReg.theta_gd)

logReg.lambda = 10;
logReg.gradient_descent_L2();
norm(logReg.theta_gd)


%% test libORF for L2 regularized logistic regression using Newtons method
clear all, close all, clc;

x = load('data/ex5Logx.dat'); 
y = load('data/ex5Logy.dat');
x_orj = x;

u = x(:,1);
v = x(:,2);
x = map_feature(u, v);

logReg  = LogisticRegressor(struct('x',x,'y',y,'nIter',15,'addBias',false,'alpha',1));
logReg.lambda = 0;
logReg.newtons_method_L2();

% visualise result
pos = find(y); neg = find(y == 0);
u = linspace(-1, 1.5, 200);
v = linspace(-1, 1.5, 200);
figure
plot(x_orj(pos, 1), x_orj(pos, 2), '+','MarkerEdgeColor','k','MarkerFaceColor','k'), hold on
plot(x_orj(neg, 1), x_orj(neg, 2), 'o','MarkerEdgeColor','k','MarkerFaceColor','g')
z = zeros(length(u), length(v));
for i = 1:length(u)
    for j = 1:length(v)   
        z(j,i) = map_feature(u(i), v(j))*logReg.theta_newton;
    end
end
contour(u,v,z, [0, 0], 'LineWidth', 2), hold off
pause


logReg.lambda = 1;
logReg.newtons_method_L2();

% visualise result    
figure
plot(x_orj(pos, 1), x_orj(pos, 2), '+','MarkerEdgeColor','k','MarkerFaceColor','k'), hold on
plot(x_orj(neg, 1), x_orj(neg, 2), 'o','MarkerEdgeColor','k','MarkerFaceColor','g')
z = zeros(length(u), length(v));
for i = 1:length(u)
    for j = 1:length(v)   
        z(j,i) = map_feature(u(i), v(j))*logReg.theta_newton;
    end
end
contour(u,v,z, [0, 0], 'LineWidth', 2), hold off
pause    
    

logReg.lambda = 10;
logReg.newtons_method_L2();

% visualise result    
figure
plot(x_orj(pos, 1), x_orj(pos, 2), '+','MarkerEdgeColor','k','MarkerFaceColor','k'), hold on
plot(x_orj(neg, 1), x_orj(neg, 2), 'o','MarkerEdgeColor','k','MarkerFaceColor','g')
z = zeros(length(u), length(v));
for i = 1:length(u)
    for j = 1:length(v)   
        z(j,i) = map_feature(u(i), v(j))*logReg.theta_newton;
    end
end
contour(u,v,z, [0, 0], 'LineWidth', 2), hold off
pause


%% test libORF for stochastic gradient descent 
clear all, close all, clc;

addpath(genpath('misc'));

x = load('data/ex5Logx.dat'); 
y = load('data/ex5Logy.dat');

u = x(:,1);
v = x(:,2);
x = map_feature(u, v);

logReg  = LogisticRegressor(struct('x',x,'y',y,'addBias',false,'alpha',1,'nIter',15,'shuffle',true,'trackJ',true));

[J0 theta0] = logReg.stochastic_gradient_descent();
J_hat0 = logReg.J_hat;
figure,plot(1:length(J0), J0)
figure,plot(1:length(J_hat0), J_hat0)
pause


%% test libORF for L2 regularized stochastic gradient descent 
clear all, close all, clc;

x = load('data/ex5Logx.dat'); 
y = load('data/ex5Logy.dat');

u = x(:,1);
v = x(:,2);
x = map_feature(u, v);

logReg  = LogisticRegressor(struct('x',x,'y',y,'addBias',false,'alpha',1,'nIter',15,'shuffle',true,'trackJ',true));

logReg.lambda = 1;
[J1 theta1] = logReg.stochastic_gradient_descent_L2();
J_hat1 = logReg.J_hat;
figure,plot(1:length(J1), J1)
figure,plot(1:length(J_hat1), J_hat1)



