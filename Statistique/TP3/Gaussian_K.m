function result_Q = Gaussian_K(k,pi_k,mu_array,sigma_array,Data)

    
    k=5;
    pi = 1/k;

    mu_array = zeros(2,k)
    mu_array(:,1)

    sigma_array = ones(4,4,k)
    sigma_array(:,:,1)


    result_Q = [];
    somme=0;
    
    d=2;
    for k=1:k
        temp = pi_k*1/(sqrt(det(sigma_array(:,:,k))*(sqrt(2*pi)^d))
        temp = temp*exp(-1*(Data-mu_array(:,k))'*(sigma_array(:,:,k)^(-1))*(Data-mu_array(:,k)/2))
        
        somme = somme + temp;
        result_Q = [result_Q, temp];
        
        
        
    end
    

    result_Q=result_Q/somme



end


