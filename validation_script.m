%% Simulación
clc
close
load('resulting_weights.mat')
load('validation_set.mat')

r_out=zeros(100,6);
for n=1:2000
    %dato entrada
    data_vent=datos_validacion(n:n+100-1,:)';
    data_simul=reshape(data_vent,1,3*100);

    test_input=data_simul;
    
    prueba=reshape(test_input,3,100)';
    figure(1)
    subplot(2,1,1)
    plot(prueba);
    ylim([-50/100,50/100])
    x_out = test_input;
    %x_out = (x_out - min) / (max - min)

    for i = 1:HN
        sumtemp = 0.0;
        for j = 1:InputN
            sumtemp = sumtemp + w(j,i) * x_out(j);
        end
        hn_out(i) = sigmoid(sumtemp);		% sigmoid serves as the activation function
    end
    for i = 1:OutN
        sumtemp = 0.0;
        for j = 1:HN
            sumtemp = sumtemp + v(j,i) * hn_out(j);
        end
        y_out(i) = sigmoid(sumtemp);
    end
    %y_out'
    subplot(2,1,2)
    bar(y_out);
    pause(0.00001);
end