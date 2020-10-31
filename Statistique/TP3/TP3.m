

clear variables;
close all;


% %% nuage de points
% Data= importdata("gmm2d.asc");
% [h,w]=size(Data);
% 
% figure(1);
% plot(Data(:,1),Data(:,2),'.');
% title 'Donées origine'
% 
% N_K_classe = 5;
% 
% %msize = numel(data_array(:,1));
% idx = randperm(h);
% K_center=Data(idx(1:N_K_classe),:);
% %rand_Data=randperm(Data)
% 
% 
% distance = zeros(h,N_K_classe);
% couleurs = ['b.', 'g.', 'r.', 'c.', 'm.']
% 
% for i=0:10
%     for k=1:N_K_classe
%         distance(:,k) = (Data(:,1) - K_center(k,1)).^2 + (Data(:,2) - K_center(k,2)).^2 ;
%     end
%     [Min,Label] = min(distance,[],2);
%     for k=1:N_K_classe
%         K_center(k,:)= mean(Data((Label==k),:))
%         
%         figure(i+1);
%         hold on;
%         plot(Data((Label==k),1),Data((Label==k),2),couleurs(k))
% 
%     end
% end
    

%% image


im1 = im2double(imread('peppers.bmp'));
[h_im,w_im,l]=size(im1);

N_K_classe2 = 5;

%msize = numel(data_array(:,1));
idx_h = randperm(h_im);
idx_w = randperm(w_im);
K_center2=zeros(N_K_classe2,3);
K_center2(:,1)=diag(im1(idx_h(1:N_K_classe2),idx_w(1:N_K_classe2),1));
K_center2(:,2)=diag(im1(idx_h(1:N_K_classe2),idx_w(1:N_K_classe2),2));
K_center2(:,3)=diag(im1(idx_h(1:N_K_classe2),idx_w(1:N_K_classe2),3));
%rand_Data=randperm(Data)
K_center2
% 
distance2 = zeros(h_im,w_im,N_K_classe2);

% 
for i=1:30
    
    for k=1:N_K_classe2
        
        distance2(:,:,k) = (im1(:,:,1) - K_center2(k,1)).^2  +  (im1(:,:,2) - K_center2(k,2)).^2  +  (im1(:,:,3) - K_center2(k,3)).^2;
    end
    [Min2,Label2] = min(distance2,[],3);
    for k=1:N_K_classe2
        moyennes = zeros(N_K_classe2,3);
        for l=1:h_im
            for m=1:w_im
                moyennes(Label2(l,m),1) = moyennes(Label2(l,m),1) + im1(l,m,1);
                moyennes(Label2(l,m),2) = moyennes(Label2(l,m),2) + im1(l,m,2);
                moyennes(Label2(l,m),3) = moyennes(Label2(l,m),3) + im1(l,m,3);
            end
        end
        k;
        s=size(im1((Label2==k)));
        moyennes(k,:)/s(1);
        if s(1) ~= 0
            disp("caca")
            moyennes(k,:)=moyennes(k,:)/s(1);
        else
            moyennes(k,:)=0;
        end
        
        %t=im1(find(Label2==k))
        %K_center2(k,:)= mean(im1(find(Label2==k),:));
        K_center2(k,:)= moyennes(k,:);
%         
        figure(40);
        subplot(211)
        imshow(Min2,[])
        subplot(212)
        imshow(Label2,[])
    end
end

image_segmentee=zeros(h_im,w_im,3);

for l=1:h_im
    for m=1:w_im
        image_segmentee(l,m,:) = K_center2(Label2(l,m),:);

    end
end

figure(50)
imshow(image_segmentee,[])


%% EM




%% nuage de points
clear variables;
close all;

Data= importdata("gmm2d.asc");
[h,w]=size(Data);

figure(1);
plot(Data(:,1),Data(:,2),'.');
title 'Donées origine'

d=2;
Data=Data';
k=5;



Data_index_k = randi(k,h,1);

Card_k = zeros(k,1);

for j=1:k
    Card_k(j)=size(Data(Data_index_k(:,1)==j),1);
end

pi_k = Card_k./h;
old_pi_k = zeros(k);

%mu_array(:,1)

%msize = numel(data_array(:,1));
idx = randperm(h);
mu_array=Data(:,idx(1:k))';
mu_array= [0,10;4,14;11.5,14;11,8;11,0.5];

% mu_array = zeros(2,k)
% for j=1:k
%     mu_array(:,j) = mean(Data(:,Data_index_k(:,1)==j),2)';
% end

%mu_array=mean(Data');

%rand_Data=randperm(Data)
% 

sigma_array = zeros(2,2,k);

for j=1:k
    %Data_j=Data(:,Data_index_k(:,1)==j);
    temp_sigma = 0;
    for i=1:h
        temp_sigma = temp_sigma + (Data(:,i)-mu_array(j,:))*(Data(:,i)-mu_array(j,:))';
    end
    sigma_array(:,:,j) = temp_sigma./Card_k(j);
end

result_Q = zeros(h,k);
result_Q_somme = zeros(h,1);

for j=1:k
    %Data_j=Data(:,Data_index_k(:,1)==j)
    %Idx=find(Data_index_k(:,1)==j)
    for i=1:h
        temp = pi_k(j)*1/( sqrt (det (sigma_array(:,:,j)) ) *( sqrt ((2*pi)^d) ) );
        temp = temp * exp( -1 * (Data(:,i)- mu_array(j,:)')' * (sigma_array(:,:,j)^(-1)) * (Data(:,i)-mu_array(j,:)') / 2 );
        temp;
        result_Q(i,j) = temp;

    end
    result_Q_somme = result_Q_somme + result_Q(:,j);
end


figure(2)
plot3(Data(1,:),Data(2,:),result_Q_somme,'.')


result_Q_norm=result_Q./result_Q_somme;

%[U,Data_index_k] = max(result_Q_norm,[],2);


figure(3)
plot3(Data(1,:),Data(2,:),result_Q_norm(:,:),'.')

mean(result_Q)
mean(result_Q_norm)
mean(result_Q_somme)

mean_pi_diff = 10;


while mean_pi_diff>0.0001

    
    for j=1:k

        
        pi_k(j)=mean(result_Q_norm(:,j));


        mu_array(j,:) = [mean( Data(1,:).*result_Q_norm(:,j)' )/pi_k(j) ,mean( (Data(2,:).*result_Q_norm(:,j)') )/(pi_k(j))];

        temp_sigma = 0;
        for i=1:h
            temp_sigma = temp_sigma + (Data(:,i)-mu_array(j,:)')*(Data(:,i)-mu_array(j,:)')'*result_Q_norm(i,j);
        end
        sigma_array(:,:,j) = temp_sigma/(h*pi_k(j));

    end

    PI_DIFF = pi_k - old_pi_k
    mean_pi_diff = mean(abs(PI_DIFF))
    result_Q = zeros(h,k);
    result_Q_somme = zeros(h,1);
    
    for j=1:k
        %Data_j=Data(:,Data_index_k(:,1)==j);
        %Idx=find(Data_index_k(:,1)==j);
        for i=1:h
            temp = pi_k(j)*1/( sqrt (det (sigma_array(:,:,j)) ) *( sqrt ((2*pi)^d) ) );
            temp = temp * exp( -1 * (Data(:,i)-mu_array(j,:)')' * (sigma_array(:,:,j)^(-1)) * (Data(:,i)-mu_array(j,:)') / 2 );
            temp;
            result_Q(i,j) = temp;

        end
        result_Q_somme = result_Q_somme + result_Q(:,j);
    end
    
    
%     
%     for j=1:k
% 
%         for i=1:h
%             temp = pi_k(j)*1/(sqrt(det(sigma_array(:,:,j))*(sqrt(2*pi)^d)));
%             temp = temp * exp( -1 * (Data(:,i)-mu_array(:,j))' * (sigma_array(:,:,j)^(-1)) * (Data(:,i)-mu_array(:,j)) / 2 );
% 
%             result_Q(i,j) = temp;
% 
%         end
%         result_Q_somme = result_Q_somme + result_Q(:,j);
%     end
% 


    result_Q_norm=result_Q./result_Q_somme;
    old_pi_k = pi_k;

end

% 
% 

figure()
plot3(Data(1,:),Data(2,:),result_Q_norm,'.')
[U,Data_index_k] = max(result_Q_norm,[],2);
result_Q;
mean(result_Q)
figure()
plot3(Data(1,:),Data(2,:),result_Q_somme,'.')

mean(result_Q)
mean(result_Q_norm)
mean(result_Q_somme)


Card_k = zeros(k,1);

% for j=1:k
%     Card_k(j)=size(Data(Data_index_k(:,1)==j),1)
% end

figure()
hold on
for j=1:k
    Data_j=Data(:,Data_index_k(:,1)==j);
    Idx=find(Data_index_k(:,1)==j);
    plot3(Data(1,Idx),Data(2,Idx),result_Q_norm(Idx,j),'.')

end




%% nuage de points 3d
clear variables;
close all;

Data= importdata("gmm3d.asc");
[h,w]=size(Data);

figure(1);
plot3(Data(:,1),Data(:,2),Data(:,3),'.');
title 'Donées origine'

d=2;
Data=Data';
k=3;



Data_index_k = randi(k,h,1);

Card_k = zeros(k,1);

for j=1:k
    Card_k(j)=size(Data(Data_index_k(:,1)==j),1);
end

pi_k = Card_k./h;
old_pi_k = zeros(k);

%mu_array(:,1)

%msize = numel(data_array(:,1));
idx = randperm(h);
mu_array=Data(:,idx(1:k))';
%mu_array= [0,10;4,14;11.5,14;11,8;11,0.5];

% mu_array = zeros(2,k)
% for j=1:k
%     mu_array(:,j) = mean(Data(:,Data_index_k(:,1)==j),2)';
% end

%mu_array=mean(Data');

%rand_Data=randperm(Data)
% 

sigma_array = zeros(3,3,k);

for j=1:k
    %Data_j=Data(:,Data_index_k(:,1)==j);
    temp_sigma = 0;
    for i=1:h
        temp_sigma = temp_sigma + (Data(:,i)-mu_array(j,:))*(Data(:,i)-mu_array(j,:))';
    end
    sigma_array(:,:,j) = temp_sigma./Card_k(j);
end

result_Q = zeros(h,k);
result_Q_somme = zeros(h,1);

for j=1:k
    %Data_j=Data(:,Data_index_k(:,1)==j)
    %Idx=find(Data_index_k(:,1)==j)
    for i=1:h
        temp = pi_k(j)*1/( sqrt (det (sigma_array(:,:,j)) ) *( sqrt ((2*pi)^d) ) );
        temp = temp * exp( -1 * (Data(:,i)- mu_array(j,:)')' * (sigma_array(:,:,j)^(-1)) * (Data(:,i)-mu_array(j,:)') / 2 );
        temp;
        result_Q(i,j) = temp;

    end
    result_Q_somme = result_Q_somme + result_Q(:,j);
end


figure(2)
plot3(Data(1,:),Data(2,:),result_Q_somme,'.')


result_Q_norm=result_Q./result_Q_somme;

%[U,Data_index_k] = max(result_Q_norm,[],2);


figure(3)
plot3(Data(1,:),Data(2,:),result_Q_norm(:,:),'.')

mean(result_Q)
mean(result_Q_norm)
mean(result_Q_somme)

mean_pi_diff = 10;


while mean_pi_diff>0.00001

    
    for j=1:k

        
        pi_k(j)=mean(result_Q_norm(:,j));


        mu_array(j,:) = [mean( Data(1,:).*result_Q_norm(:,j)' )/pi_k(j) ,mean( (Data(2,:).*result_Q_norm(:,j)') )/(pi_k(j)), mean( (Data(3,:).*result_Q_norm(:,j)') )/(pi_k(j))];

        temp_sigma = 0;
        for i=1:h
            temp_sigma = temp_sigma + (Data(:,i)-mu_array(j,:)')*(Data(:,i)-mu_array(j,:)')'*result_Q_norm(i,j);
        end
        sigma_array(:,:,j) = temp_sigma/(h*pi_k(j));

    end

    PI_DIFF = pi_k - old_pi_k
    mean_pi_diff = mean(abs(PI_DIFF))
    result_Q = zeros(h,k);
    result_Q_somme = zeros(h,1);
    
    for j=1:k
        %Data_j=Data(:,Data_index_k(:,1)==j);
        %Idx=find(Data_index_k(:,1)==j);
        for i=1:h
            temp = pi_k(j)*1/( sqrt (det (sigma_array(:,:,j)) ) *( sqrt ((2*pi)^d) ) );
            temp = temp * exp( -1 * (Data(:,i)-mu_array(j,:)')' * (sigma_array(:,:,j)^(-1)) * (Data(:,i)-mu_array(j,:)') / 2 );
            temp;
            result_Q(i,j) = temp;

        end
        result_Q_somme = result_Q_somme + result_Q(:,j);
    end
    
    
%     
%     for j=1:k
% 
%         for i=1:h
%             temp = pi_k(j)*1/(sqrt(det(sigma_array(:,:,j))*(sqrt(2*pi)^d)));
%             temp = temp * exp( -1 * (Data(:,i)-mu_array(:,j))' * (sigma_array(:,:,j)^(-1)) * (Data(:,i)-mu_array(:,j)) / 2 );
% 
%             result_Q(i,j) = temp;
% 
%         end
%         result_Q_somme = result_Q_somme + result_Q(:,j);
%     end
% 


    result_Q_norm=result_Q./result_Q_somme;
    old_pi_k = pi_k;

end

% 
% 

figure()
plot3(Data(1,:),Data(2,:),result_Q_norm,'.')
[U,Data_index_k] = max(result_Q_norm,[],2);
result_Q;
mean(result_Q)
figure()
plot3(Data(1,:),Data(2,:),result_Q_somme,'.')

mean(result_Q)
mean(result_Q_norm)
mean(result_Q_somme)


Card_k = zeros(k,1);

% for j=1:k
%     Card_k(j)=size(Data(Data_index_k(:,1)==j),1)
% end

figure()
hold on
for j=1:k
    Data_j=Data(:,Data_index_k(:,1)==j);
    Idx=find(Data_index_k(:,1)==j);
    %plot3(Data(1,Idx),Data(2,Idx),result_Q_norm(Idx,j),'.')
    plot3(Data(1,Idx),Data(2,Idx),Data(3,Idx),'.')
end




%% Image


clear variables;
close all;

%im = im2double(imread('text0.png'));
im = im2double(imread('peppers.bmp'));

[h_im,w_im,p_im]=size(im);

Data = zeros(h_im*w_im,3);


for i=1:h_im
    for j=1:w_im
        Data((j-1)+w_im*(i-1)+1,:)=im(i,j,:);
    end
end

%Data= importdata("gmm3d.asc");
[h,w]=size(Data);

figure(1);
plot3(Data(:,1),Data(:,2),Data(:,3),'.');
title 'Donées origine'

d=2;
Data=Data';
k=3;



Data_index_k = randi(k,h,1);

Card_k = zeros(k,1);

for j=1:k
    Card_k(j)=size(Data(Data_index_k(:,1)==j),1);
end

pi_k = Card_k./h;
old_pi_k = zeros(k);

%mu_array(:,1)

%msize = numel(data_array(:,1));
idx = randperm(h);
mu_array=Data(:,idx(1:k))';
%mu_array= [0,10;4,14;11.5,14;11,8;11,0.5];

% mu_array = zeros(2,k)
% for j=1:k
%     mu_array(:,j) = mean(Data(:,Data_index_k(:,1)==j),2)';
% end

%mu_array=mean(Data');

%rand_Data=randperm(Data)
% 

sigma_array = zeros(3,3,k);

for j=1:k
    %Data_j=Data(:,Data_index_k(:,1)==j);
    temp_sigma = 0;
    for i=1:h
        temp_sigma = temp_sigma + (Data(:,i)-mu_array(j,:))*(Data(:,i)-mu_array(j,:))';
    end
    sigma_array(:,:,j) = temp_sigma./Card_k(j);
end

result_Q = zeros(h,k);
result_Q_somme = zeros(h,1);

for j=1:k
    %Data_j=Data(:,Data_index_k(:,1)==j)
    %Idx=find(Data_index_k(:,1)==j)
    for i=1:h
        temp = pi_k(j)*1/( sqrt (det (sigma_array(:,:,j)) ) *( sqrt ((2*pi)^d) ) );
        temp = temp * exp( -1 * (Data(:,i)- mu_array(j,:)')' * (sigma_array(:,:,j)^(-1)) * (Data(:,i)-mu_array(j,:)') / 2 );
        temp;
        result_Q(i,j) = temp;

    end
    result_Q_somme = result_Q_somme + result_Q(:,j);
end


figure(2)
plot3(Data(1,:),Data(2,:),result_Q_somme,'.')


result_Q_norm=result_Q./result_Q_somme;

%[U,Data_index_k] = max(result_Q_norm,[],2);


figure(3)
plot3(Data(1,:),Data(2,:),result_Q_norm(:,:),'.')

mean(result_Q)
mean(result_Q_norm)
mean(result_Q_somme)

mean_pi_diff = 10;


while mean_pi_diff>0.001

    
    for j=1:k

        
        pi_k(j)=mean(result_Q_norm(:,j));


        mu_array(j,:) = [mean( Data(1,:).*result_Q_norm(:,j)' )/pi_k(j) ,mean( (Data(2,:).*result_Q_norm(:,j)') )/(pi_k(j)), mean( (Data(3,:).*result_Q_norm(:,j)') )/(pi_k(j))];

        temp_sigma = 0;
        for i=1:h
            temp_sigma = temp_sigma + (Data(:,i)-mu_array(j,:)')*(Data(:,i)-mu_array(j,:)')'*result_Q_norm(i,j);
        end
        sigma_array(:,:,j) = temp_sigma/(h*pi_k(j));

    end

    PI_DIFF = pi_k - old_pi_k
    mean_pi_diff = mean(abs(PI_DIFF))
    result_Q = zeros(h,k);
    result_Q_somme = zeros(h,1);
    
    for j=1:k
        %Data_j=Data(:,Data_index_k(:,1)==j);
        %Idx=find(Data_index_k(:,1)==j);
        for i=1:h
            temp = pi_k(j)*1/( sqrt (det (sigma_array(:,:,j)) ) *( sqrt ((2*pi)^d) ) );
            temp = temp * exp( -1 * (Data(:,i)-mu_array(j,:)')' * (sigma_array(:,:,j)^(-1)) * (Data(:,i)-mu_array(j,:)') / 2 );
            temp;
            result_Q(i,j) = temp;

        end
        result_Q_somme = result_Q_somme + result_Q(:,j);
    end
    
    
%     
%     for j=1:k
% 
%         for i=1:h
%             temp = pi_k(j)*1/(sqrt(det(sigma_array(:,:,j))*(sqrt(2*pi)^d)));
%             temp = temp * exp( -1 * (Data(:,i)-mu_array(:,j))' * (sigma_array(:,:,j)^(-1)) * (Data(:,i)-mu_array(:,j)) / 2 );
% 
%             result_Q(i,j) = temp;
% 
%         end
%         result_Q_somme = result_Q_somme + result_Q(:,j);
%     end
% 


    result_Q_norm=result_Q./result_Q_somme;
    old_pi_k = pi_k;

end

% 
% 

figure()
plot3(Data(1,:),Data(2,:),result_Q_norm,'.')
[U,Data_index_k] = max(result_Q_norm,[],2);
result_Q;
mean(result_Q)
figure()
plot3(Data(1,:),Data(2,:),result_Q_somme,'.')

mean(result_Q)
mean(result_Q_norm)
mean(result_Q_somme)


Card_k = zeros(k,1);

% for j=1:k
%     Card_k(j)=size(Data(Data_index_k(:,1)==j),1)
% end

Value = zeros(3,k);
Data_Result = zeros(h_im*w_im,3)';

figure()
hold on
for j=1:k
    Data_j=Data(:,Data_index_k(:,1)==j);
    Idx=find(Data_index_k(:,1)==j);
    %plot3(Data(1,Idx),Data(2,Idx),result_Q_norm(Idx,j),'.')
    plot3(Data(1,Idx),Data(2,Idx),Data(3,Idx),'.')
    Value(:,j) = mean(Data(:,Idx),2);
    h_idx=size(Idx);
    for l=1:h_idx
        Data_Result(:,Idx(l)) = Value(:,j);
    end
end


figure()
plot3(Data_Result(:,1),Data_Result(:,2),Data_Result(:,3)'.')

im2=im;
for i=1:h_im
    for j=1:w_im
        im2(i,j,:) = Data_Result(:,(j-1)+w_im*(i-1)+1);
    end
end

figure()
subplot(211)
imshow(im,[])
subplot(212)
imshow(im2,[])

