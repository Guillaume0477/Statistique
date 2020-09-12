%% read image
clear all; close all; clc;

str_image='text0.png';
taille= 5;%impair
base_taille = 20; % max = 
size_modelized = 2;
epsilon=0.0005;

int= Efros_Leung(str_image , taille, base_taille, size_modelized, epsilon );