%% read image
clear all; close all; clc;

str_image='text0.png';
taille= [5,5];%impair
img = im2double(imread(str_image));
[w_true,l_true,c_true] = size(img);
taille_swp = [w_true-1,l_true-1];
base_taille = 25;
size_modelized = 30;
liste_patch=[[3,3];[5,5];[9,9];[15,15];[23,23]];
Liste_epsilon=[0,0.05,0.1,0.2,0.5];
liste_size_modelized=[];

for k=1:1 %image 
    for i=1:1 %taille patch  
        for j=1:1 %epsilon
            
            str_image=['text',num2str(k),'.png']
            img = im2double(imread(str_image));
            [w_true,l_true,c_true] = size(img);
            taille_swp = [w_true-1,l_true-1];
            base_taille = l_true - 20;
            Liste_epsilon(j)
            liste_patch(i,:)
            Efros_Leung(str_image , liste_patch(i,:), taille_swp, base_taille, size_modelized, Liste_epsilon(j))
            disp([str_image, ' patch = '])
            liste_patch(i,:)
        end
    end
end




