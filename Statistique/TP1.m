%% read image
clear all; close all; clc;

img = im2double(imread('text0.png'));

figure()
imshow(img,[])
title('image origine')

%% patch init
[w_true,l_true,c_true] = size(img)

taille= 3%impair
point_depart=[uint64(w_true/2),uint64(l_true/2)]
patch = img((point_depart(1)-taille/2:point_depart(1)+taille/2-1),(point_depart(2)-taille/2:point_depart(2)+taille/2-1), :);
figure()
imshow(patch,[])
title('taille patch')


%% swp_img init
%<w_true

%swp_taille = 5*taille-1;
swp_img = img%((point_depart(1)-swp_taille/2:point_depart(1)+swp_taille/2-1),(point_depart(2)-swp_taille/2:point_depart(2)+swp_taille/2-1),:);
%swp_img=img(:,:,1)

%% base modelized
[w_swp,l_swp,c_swp] = size(swp_img);


base_taille = 63; % max = 
point_depart_modilized=[uint64(w_swp/2),uint64(l_swp/2)]

base_modelized = swp_img((point_depart_modilized(1)-base_taille/2:point_depart_modilized(1)+base_taille/2-1),(point_depart_modilized(2)-base_taille/2:point_depart_modilized(2)+base_taille/2-1),:);

figure()
imshow(base_modelized,[])
title('image debut')

%% image modelized



size_modelized_x = 1;
size_modelized_y = 1;
size_modelized = 8;

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
       modelized_img( (taille-1)/2 + size_modelized + k   , (taille-1)/2 + size_modelized + l ,: ) = base_modelized(k,l,:);
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
title('image modelisé')
subplot(133)
imshow(filled_matrice,[])
title('matrice filled elements')

epsilon=0.0005;

% boucle de creation de pixels
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

    Min_dist_patch = 255;
    good_a=-1;
    good_b=-1;
    %parcourt swp_img
    for a=((taille-1)/2+1):w_swp-(taille-1)/2
        for b=((taille-1)/2+1):l_swp-(taille-1)/2
           Dist_patch = 0;
           %parcourt patch
           for k=1:w_patch
               for l=1:l_patch
                   % si la valeur du pixel est connue
                   if filled_matrice(good_i+k- (w_patch+1)/2,good_j+l- (l_patch+1)/2)
                       Dist_patch = ( (swp_img( k+a-(w_patch+1)/2 , l+b-(l_patch+1)/2 , :) - modelized_img( good_i+k-(w_patch+1)/2 , good_j+l-(l_patch+1)/2 ,:)).^2 )/max_neighboorood + Dist_patch;
                   end
               end
           end
           Dist_patch_med = (Dist_patch(:,:,1)+Dist_patch(:,:,2)+Dist_patch(:,:,3))/3;
           if Dist_patch_med < Min_dist_patch
               good_a=a;
               good_b=b;
               Min_dist_patch = Dist_patch_med; 
               if Dist_patch_med < epsilon
                   disp("BREAK")

                   break;
               end
               
            end
        end
        if Dist_patch_med < epsilon
            disp("BREAK2")
            break;
        end
    end
    Min_dist_patch
    modelized_img(good_i,good_j,:)=swp_img(good_a,good_b,:);
    filled_matrice(good_i,good_j)=1;
    compteur = compteur +1;
 
end

image_finale = modelized_img((taille-1)/2+1:w_mod-(taille-1)/2 , (taille-1)/2+1:l_mod-(taille-1)/2 , :);

figure()
subplot(131)
imshow(swp_img,[])
title('image swp')
subplot(132)
imshow(modelized_img,[])
title('image modelisé')
subplot(133)
imshow(filled_matrice,[])
title('matrice filled elements')

figure()
subplot(121)
imshow(image_finale,[])
title('image_finale')

subplot(122)
imshow(base_modelized,[])
title('image_origine')