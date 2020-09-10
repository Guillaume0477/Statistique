%% init
clear all; close all; clc;

img = im2double(imread('text0.png'));

figure()
imshow(img,[])
title('image origine')

%% patch init
[w_true,l_true,c_true] = size(img)

taille= 11%impair
point_depart=[uint64(w_true/2),uint64(l_true/2)]
patch = img((point_depart(1)-taille/2:point_depart(1)+taille/2-1),(point_depart(2)-taille/2:point_depart(2)+taille/2-1), :);
figure()
imshow(patch,[])
title('patch')


%% swp_img init
%<w_true
swp_taille = 20;
%swp_taille = 5*taille-1;
swp_img = img%((point_depart(1)-swp_taille/2:point_depart(1)+swp_taille/2-1),(point_depart(2)-swp_taille/2:point_depart(2)+swp_taille/2-1),:);
%swp_img=img(:,:,1)
[w_swp,l_swp,c_swp] = size(swp_img);

size_modelized = 1;

base_modelized = swp_img;

[w_base,l_base,c_base] = size(base_modelized);
modelized_img = zeros(w_base+size_modelized*2+taille-1,l_base+size_modelized*2+taille-1,3);
filled_matrice = zeros(w_base+size_modelized*2+taille-1,l_base+size_modelized*2+taille-1);

[w_patch,l_patch,c_patch] = size(patch);
[w_mod,l_mod,c_mod] = size(modelized_img);
compteur=0;

for k=1:w_base
   for l=1:l_base
       
       %a=modilized_img(point_depart(1) - taille/2 + k +  (taille/2)  ,point_depart(2) - taille/2 + l +   (taille/2)   ,:)
       %b=patch(k,l,:)
       modelized_img( (taille-1)/2 + size_modelized + k   , (taille-1)/2 + size_modelized + l ,: ) = swp_img(k,l,:);
       filled_matrice( (taille-1)/2 + size_modelized + k  , (taille-1)/2 + size_modelized + l  ) = 1;
       %ab=modilized_img(point_depart(1) - taille/2 + k +  (taille/2)  ,point_depart(2) - taille/2 + l +   (taille/2)   ,:)
       compteur=compteur+1;
       
   end
end

figure()
subplot(131)
imshow(swp_img,[])
title('image swp')
subplot(132)
imshow(modelized_img,[])
title('image modelis�')
subplot(133)
imshow(filled_matrice,[])
title('matrice filled elements')



% boucle de creation de pixels
%while compteur<w_true*l_true
fin=(w_base+2*size_modelized)*(l_base+2*size_modelized)
while compteur < fin
%%% find good pixel
 


    max_neighboorood=0;
    % boucle sur 
    for i=(((taille-1)/2)+1):w_mod-((taille-1)/2)
        for j=(((taille-1)/2)+1):l_mod-((taille-1)/2)
            neighboorood = 0;

            for k=1:w_patch
               for l=1:l_patch
                   neighboorood = neighboorood + filled_matrice(i+k- (w_patch+1)/2,j+l- (l_patch+1)/2);
               end
            end
            if neighboorood > max_neighboorood && filled_matrice(i,j) == 0
               good_i=i;
               good_j=j;
               max_neighboorood = neighboorood;
            end
        end
    end
    max_neighboorood;
    good_i
    good_j

    %%% distance

    % taille/2 = w_patch = l_patch

    %for i=(taille/2):w_true+(taille/2)
    %   for j=(taille/2):l_true+(taille/2)




    Min_dist_patch = 255;
    good_a=-1;
    good_b=-1;
    for a=((taille-1)/2+1):w_swp-(taille-1)/2
        for b=((taille-1)/2+1):l_swp-(taille-1)/2
           Dist_patch = 0;
           G=0;
           %mean(mean())
           for k=1:w_patch
               for l=1:l_patch
                   %patch(k,l);
                   %patch(k,l) - modilized_img(i+k - w_patch/2 ,j+l - l_patch/2);
                   %swp_img(k+a - (w_patch+1)/2,l+b - (l_patch+1)/2);
                   %modelized_img(  good_i + k - (w_patch+1)/2  ,  good_j + l - (l_patch+1)/2  );
                   %Dist_patch = ((swp_img(k+a - (w_patch+1)/2,l+b - (l_patch+1)/2) - modilized_img(good_i+k- (w_patch+1)/2,good_j+l- (l_patch+1)/2))^2)*filled_matrice(good_i+k- (w_patch+1)/2,good_j+l- (l_patch+1)/2) + Dist_patch;
                   %res1 = swp_img(k+a - (w_patch+1)/2,l+b - (l_patch+1)/2)
                   %res2 = modilized_img(good_i+k- (w_patch+1)/2,good_j+l- (l_patch+1)/2)
                   %res3 = int64(swp_img(k+a - (w_patch+1)/2,l+b - (l_patch+1)/2)) - int64(modilized_img(good_i+k- (w_patch+1)/2,good_j+l- (l_patch+1)/2))
                   if filled_matrice(good_i+k- (w_patch+1)/2,good_j+l- (l_patch+1)/2)
                       Dist_patch = ( (base_modelized( k+a-(w_patch+1)/2 , l+b-(l_patch+1)/2 , :) - modelized_img( good_i+k-(w_patch+1)/2 , good_j+l-(l_patch+1)/2 ,:)).^2 )/max_neighboorood + Dist_patch;
                       G = G + filled_matrice(good_i+k- (w_patch+1)/2,good_j+l- (l_patch+1)/2);
                   end
               end
           end
           max_neighboorood;
           G;
           Dist_patch_med = (Dist_patch(:,:,1)+Dist_patch(:,:,2)+Dist_patch(:,:,3))/3;
           if Dist_patch_med < Min_dist_patch
               good_a=a;
               good_b=b;
               Min_dist_patch = Dist_patch_med; 
           end
        end
    end
    good_a;
    good_b;
    Min_dist_patch; 
    modelized_img(good_i,good_j,:)=swp_img(good_a,good_b,:);
    filled_matrice(good_i,good_j)=1;
    compteur = compteur +1
 
end


       %Dist_img(i,j)= Dist_patch;
       %Dist_img(i,j)= Dist_patch/(w_patch*l_patch);
%   end
%end 


image_final = modelized_img((taille-1)/2+1:w_swp-(taille-1)/2 , (taille-1)/2+1:l_swp-(taille-1)/2 , :);

figure()
subplot(131)
imshow(swp_img,[])
title('image swp')
subplot(132)
imshow(modelized_img,[])
title('image modelis�')
subplot(133)
imshow(filled_matrice,[])
title('matrice filled elements')

figure()
imshow(image_final,[])
title('image_final')