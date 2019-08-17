%%
%clear
clc
load('resulting_weights.mat')

%% Declaracion de variables
InputN=3*cant_muestras                                  % number of neurons in the input layer
HN=100                                                  % number of neurons in the hidden layer
OutN=num_gestos                                         % number of neurons in the output layer
datanum=num_data_por_gesto*num_gestos                   % number of training samples


x_out=zeros(InputN,1);		% input layer
hn_out=zeros(HN,1);			% hidden layer
y_out=zeros(OutN,1);         % output layer
y=zeros(OutN,1);				% expected output layer
w=zeros(InputN,HN);		% weights from input layer to hidden layer
v=zeros(HN,OutN);			% weights from hidden layer to output layer

deltaw=zeros(InputN,HN);
deltav=zeros(HN,OutN);

hn_delta=zeros(HN,1);		% delta of hidden layer
y_delta=zeros(OutN,1)		% delta of output layer

errlimit = 0.001
gamma = 0.1;
alpha_inicial = 0.9;
loop = 0;
times = 50000;

%Delta bar delta
k = 0.05, theta = 0.08, phi = 0.1;
%para v
alphav=zeros(HN,OutN)
media_deltav=zeros(HN,OutN)
%para w
alphaw=ones(InputN,HN), media_deltaw=zeros(InputN,HN);

epocas=0
errores=0

%% Generate data samples
datainput= entradas;
datateach= salidas;

w = rand(InputN,HN);
alphaw=ones(InputN,HN)*alpha_inicial;

v = rand(HN,OutN);
alphav=ones(HN,OutN)*alpha_inicial;

%% Entrenamiento
	while loop < times
		loop=loop+1;
		error = 0.0;

		for m = 1:datanum
			% Feedforward

			max = 1;
			min = 0;
            
			x_out= datainput(m,:);
            y = datateach(m,:);
            
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
			% Backpropagation
			for i = 1:OutN
				errtemp = y(i) - y_out(i);

				y_delta(i) = -errtemp * sigmoid(y_out(i)) * (1.0 - sigmoid(y_out(i)));

				error = error+errtemp * errtemp;
            end
			% Stochastic gradient descent
			
					for i = 1:OutN
						for j =1:HN
							%Delta Bar Delta						
							delta = y_delta(i) * hn_out(j);

							deltav(j,i) = gamma * deltav(j,i) + alphav(j,i) * delta;
							v(j,i) = v(j,i)-deltav(j,i);

							if (media_deltav(j,i) * delta > 0)
								delta_alpha = k;
							else 
								delta_alpha = -phi*alphav(j,i);
							end

							alphav(j,i) = alphav(j,i) + delta_alpha;
							media_deltav(j,i) = (1 - theta)*delta + theta*media_deltav(j,i);

						end
					end
				
					for i =1:HN
						errtemp = 0.0;
						for j =1:OutN
							errtemp = errtemp + y_delta(j) * v(i,j);
                        end
						hn_delta(i) = errtemp * (1.0 + hn_out(i)) * (1.0 - hn_out(i));
                    end
                    
					for i =1:HN
						for j =1:InputN
							%Delta Bar Delta	
							delta = hn_delta(i) * x_out(j);

							deltaw(j,i) = gamma * deltaw(j,i) + alphaw(j,i) * delta;
							w(j,i) = w(j,i)-deltaw(j,i);

							if (media_deltaw(j,i) * delta > 0) 
								delta_alpha = k;
							else 
								delta_alpha = -phi*alphaw(j,i);
							end

							alphaw(j,i) = alphaw(j,i) + delta_alpha;
							media_deltaw(j,i) = (1 - theta)*delta + theta*media_deltaw(j,i);

						end
                    end
			
		end

		% Global error 
		error = error / 2;
        
		if (error < errlimit)
			break;
        end

		fprintf('El %d avo entrenamiento, error: %f\n', loop, error);
        epocas=[epocas,loop];
        errores=[errores, error];
    end
    
	%Impresión de resultados
	fprintf('\nError final: %f\n', error);
    plot(epocas, errores)
    title('Error vs Época')
     ylabel('Error')
      xlabel('Época')
        grid on
        hold on
    
%% Validación
    %Test Feedforward
	max = 1;
	min = -1;
    n=15
	test_input=entradas(n,:)
    salidas(n)
    
    figure(2);
    prueba=reshape(test_input,3,100)';
    plot(prueba);
	fprintf('\nEntrada: ');
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
    y_out'