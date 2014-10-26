%% load data

x = load('data/ex2x.dat');
y = load('data/ex2y.dat');


%% initialize linear regressor, apply gradient descent, perform prediction

linReg1     = LinearRegressor(struct('x', x, 'y', y));
[J theta1]  = linReg1.gradient_descent();
labels1     = linReg1.predict_samples([1 3.5; 1 7]);
linReg1.theta

% Plot the training data and linear fit
figure, plot(x, y, 'o'), ylabel('Height in meters'), xlabel('Age in years'),hold on;
x_prime = [ones(size(x,1), 1) x]; % Add a column of ones to x
plot(x_prime(:,2), x_prime*theta1, '-')
legend('Training data', 'Linear regression w/o feature scaling'), hold off
pause;


%% initialize linear regressor, apply gradient descent, perform prediction

linReg2    = LinearRegressor(struct('x', x, 'y', y, 'scaleX',true));
[J theta2] = linReg2.gradient_descent();
labels2    = linReg2.predict_samples([1 3.5; 1 7]);
linReg2.theta

% Plot the training data and linear fit
figure, plot(PreProcessing.normalize_zero_mean_unit_var(x'), y, 'o'), ylabel('Height in meters'), xlabel('Age in years'),hold on;
x_prime = [ones(size(x,1), 1) PreProcessing.normalize_zero_mean_unit_var(x')']; % Add a column of ones to x
plot(x_prime(:,2), x_prime*theta2, '-')
legend('Training data', 'Linear regression with feature scaling'), hold off
pause;


%% initialize linear regressor, apply normal equation, perform prediction

linReg3 = LinearRegressor(struct('x', x, 'y', y));
theta3  = linReg3.normal_equation();
labels3 = linReg3.predict_samples([1 3.5; 1 7]);
linReg3.theta

% Plot the training data and linear fit
figure, plot(x, y, 'o'), ylabel('Height in meters'), xlabel('Age in years'),hold on;
x_prime = [ones(size(x,1), 1) x]; % Add a column of ones to x
plot(x_prime(:,2), x_prime*theta3, '-')
legend('Training data', 'Linear regression using Normal Eq.'), hold off
pause;

clear all

%% load data

x = load('data/ex3x.dat');
y = load('data/ex3y.dat');


%% initialtize linear regressor, apply feature scaling

linReg = LinearRegressor(struct('x',x,'y',y,'scaleX',true, 'nIter',100, 'alpha',0.1));


%% apply gradient descent

[J_gd theta_gd] = linReg.gradient_descent();

theta_gd

% Plot the first 50 J terms
figure,plot(0:49, J_gd(1:50), 'g', 'LineWidth', 2);
legend('Alpha(step size):0.1')
xlabel('Number of iterations')
ylabel('Cost J')


%% apply normal equation

theta_normal = linReg.normal_equation()


%% test libORF for L2 regularized linear regression with varying lambdas
clear all;

lambdas = [0, 1, 10];

for lambda = lambdas

    x = load('data/ex5Linx.dat');
    y = load('data/ex5Liny.dat');
    
    % Plot the training data
    figure;
    plot(x, y, 'o', 'MarkerFacecolor', 'r', 'MarkerSize', 8);
    
    x = [x, x.^2, x.^3, x.^4, x.^5];
    
    linReg1     = LinearRegressor(struct('x', x, 'y', y,'lambda',lambda));
    
    [J theta_gd_L2] = linReg1.gradient_descent_L2();
    theta_normal_L2 = linReg1.normal_equation_L2();
    
    theta_gd_L2
    theta_normal_L2
    
    % Plot the linear fit
    hold on;
    x_vals = (-1:0.05:1)';
    features = [ones(size(x_vals)), x_vals, x_vals.^2, x_vals.^3,...
        x_vals.^4, x_vals.^5];
    plot(x_vals, features*theta_gd_L2, '--', 'LineWidth', 2)
    legend('Training data', ['5th order fit, \lambda=' num2str(lambda)])
    hold off
    
end





