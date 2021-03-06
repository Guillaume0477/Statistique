

function temps = Efros_Leung(str_image , taille, taille_swp, base_taille, size_modelized, epsilon )


    tic
    img = im2double(imread(str_image));

    figure()
    imshow(img,[])
    title('image origine')

    %% patch init
    [w_true,l_true,c_true] = size(img);


    point_depart=[uint64(w_true/2),uint64(l_true/2)];
    patch = img((point_depart(1)-taille(1)/2:point_depart(1)+taille(1)/2-1),(point_depart(2)-taille(2)/2:point_depart(2)+taille(2)/2-1), :);
%     figure()
%     imshow(patch,[])
%     title('taille patch')


    %% swp_img init
    %<w_true

    swp_img = img((point_depart(1)-taille_swp(1)/2:point_depart(1)+taille_swp(1)/2-1), (point_depart(2)-taille_swp(2)/2:point_depart(2)+taille_swp(2)/2-1), :);

    %img%((point_depart(1)-swp_taille/2:point_depart(1)+swp_taille/2-1),(point_depart(2)-swp_taille/2:point_depart(2)+swp_taille/2-1),:);
    %swp_img=img(:,:,1)

    %% base modelized
    [w_swp,l_swp,c_swp] = size(swp_img);


    point_depart_modilized=[uint64(w_swp/2),uint64(l_swp/2)];

    base_modelized = swp_img((point_depart_modilized(1)-base_taille/2:point_depart_modilized(1)+base_taille/2-1),(point_depart_modilized(2)-base_taille/2:point_depart_modilized(2)+base_taille/2-1),:);

%     figure()
%     imshow(base_modelized,[])
%     title('image debut')

    %% image modelized



    size_modelized_x = 1;
    size_modelized_y = 1;


    [w_base,l_base,c_base] = size(base_modelized);
    modelized_img = zeros(w_base+size_modelized*2+taille(1)-1,l_base+size_modelized*2+taille(2)-1,3);
    filled_matrice = zeros(w_base+size_modelized*2+taille(1)-1,l_base+size_modelized*2+taille(2)-1);

    [w_patch,l_patch,c_patch] = size(patch);
    [w_mod,l_mod,c_mod] = size(modelized_img);
    compteur=0;

    for k=1:w_base
       for l=1:l_base

           %a=modilized_img(point_depart(1) - taille/2 + k +  (taille/2)  ,point_depart(2) - taille/2 + l +   (taille/2)   ,:)
           %b=patch(k,l,:)
           modelized_img( (taille(1)-1)/2 + size_modelized + k   , (taille(2)-1)/2 + size_modelized + l ,: ) = base_modelized(k,l,:);
           filled_matrice( (taille(1)-1)/2 + size_modelized + k  , (taille(2)-1)/2 + size_modelized + l  ) = 1;
           %ab=modilized_img(point_depart(1) - taille/2 + k +  (taille/2)  ,point_depart(2) - taille/2 + l +   (taille/2)   ,:)
           compteur=compteur+1;

       end
    end

    image_initiale = modelized_img((taille(1)-1)/2+1:w_mod-(taille(1)-1)/2 , (taille(2)-1)/2+1:l_mod-(taille(2)-1)/2 , :);

%     figure()
%     subplot(131)
%     imshow(swp_img,[])
%     title('image swp')
%     subplot(132)
%     imshow(modelized_img,[])
%     title('image modelis�')
%     subplot(133)
%     imshow(filled_matrice,[])
%     title('matrice filled elements')



    % boucle de creation de pixels
    fin=(w_base+2*size_modelized)*(l_base+2*size_modelized);
    while compteur < fin
    %%% find good pixel
        max_neighboorood=0;
        % boucle sur 
        for i=(((taille(1)-1)/2)+1):w_mod-((taille(1)-1)/2)
            for j=(((taille(2)-1)/2)+1):l_mod-((taille(2)-1)/2)
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
        for a=((taille(1)-1)/2+1):w_swp-(taille(1)-1)/2
            for b=((taille(2)-1)/2+1):l_swp-(taille(2)-1)/2
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
               if Dist_patch_med < Min_dist_patch*(1+epsilon)
                   good_a=a;
                   good_b=b;
                   Min_dist_patch = Dist_patch_med; 
%                    if Dist_patch_med < epsilon
%                        disp("BREAK")
% 
%                        break;
%                    end

                end
            end
%             if Dist_patch_med < epsilon
%                 disp("BREAK2")
%                 break;
%             end
        end
        % min
        %Min_dist_patch
        
        modelized_img(good_i,good_j,:)=swp_img(good_a,good_b,:);
        filled_matrice(good_i,good_j)=1;
        compteur = compteur +1;

    end

    image_finale = modelized_img((taille(1)-1)/2+1:w_mod-(taille(1)-1)/2 , (taille(2)-1)/2+1:l_mod-(taille(2)-1)/2 , :);
    [dim_x,dim_y,dim_z]=size(image_finale);

    time = toc;
    
%     figure()
%     subplot(131)
%     imshow(swp_img,[])
%     title('image swp')
%     subplot(132)
%     imshow(modelized_img,[])
%     title('image modelis�')
%     subplot(133)
%     imshow(filled_matrice,[])
%     title('matrice filled elements')

    figure()
    subplot(121)
    imshow(image_initiale,[])
    title('image origine')
    subplot(122)
    imshow(image_finale,[])
    title('image finale')
    saveas(gcf,[ 'figures_saved/figure 2 ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized),' image = ',  str_image, ' ).png' ], 'png')
    saveas(gcf,[ 'figures_saved/figure 2 ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized),' image = ',  str_image, ' ).fig' ], 'fig')



    figure()
    subplot(221)
    imshow(patch,[])
    title([ 'taille patch (', num2str(taille(1)), 'x',num2str(taille(2)) ,  ')'])
    subplot(222)
    imshow(swp_img,[])
    title(['image swp (', num2str(w_swp), 'x',num2str(l_swp) ,  ')'])
    subplot(223)
    imshow(image_initiale,[])
    title(['image origine (', num2str(w_base), 'x',num2str(l_base) ,  ')'])
    subplot(224)
    imshow(image_finale,[])
    title(['image modelis� final (', num2str(dim_x), 'x',num2str(dim_y) ,' en ',num2str(time),  's)'])
    saveas(gcf,[ 'figures_saved/all figure ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized),' image = ',  str_image, ' ).png' ], 'png')
    saveas(gcf,[ 'figures_saved/all figure ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized),' image = ',  str_image, ' ).fig' ], 'fig')



    %figure('position',get(0,'Screensize'))
    figure()
    imshow(img,[])
    title(['image origine'])
    saveas(gcf,[ 'figures_saved/image origine ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized),' image = ',  str_image, ' ).png' ], 'png')
    saveas(gcf,[ 'figures_saved/image origine ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized),' image = ',  str_image, ' ).fig' ], 'fig')
    figure()
    imshow(patch,[])
    title(['image patch'])
    saveas(gcf, ['figures_saved/image patch ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized), ' image = ', str_image, ').png' ], 'png')
    saveas(gcf, ['figures_saved/image patch ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized), ' image = ', str_image, ').fig' ], 'fig')
    figure()
    imshow(swp_img,[])
    title(['image swp'])
    saveas(gcf, ['figures_saved/image swp ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized), ' image = ', str_image, ').png' ], 'png')
    saveas(gcf, ['figures_saved/image swp ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized), ' image = ', str_image, ').fig' ], 'fig')
    figure()
    imshow(base_modelized,[])
    title(['image initiale'])
    saveas(gcf, ['figures_saved/image initiale ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized), ' image = ', str_image, ').png' ], 'png')
    saveas(gcf, ['figures_saved/image initiale ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized), ' image = ', str_image, ').fig' ], 'fig')
    figure()
    imshow(image_finale,[])
    title(['image modelis� final (', num2str(dim_x), 'x',num2str(dim_y) ,  ')'])
    saveas(gcf, ['figures_saved/image modelisee finale ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized), ' image = ', str_image, ').png' ], 'png') 
    saveas(gcf, ['figures_saved/image modelisee finale ( taille patch = ', num2str(taille),' epsilon = ', num2str(epsilon),' taille debut = ', num2str(base_taille), ' taille modelis�e = ',num2str(size_modelized), ' image = ', str_image, ').fig' ], 'fig') 

    
    temps=time;

end

