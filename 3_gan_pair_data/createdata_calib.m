mkdir('testa');
delete('testa\*');
emo = 0;
emol = [0 1 3];
cnt = 1;
cnt2 = 1; % number of images 1:200 
labelc = -1;
labelt = -1;

for c=1:1 % all emotions
    
name = sprintf('data_train/%d/',emo(c));
fileList = getAllFiles(name);

for ii=1:100%ntrain:length(fileList)/2  % all persons
    
  listid = {};
  listid2 = {};  
    
    % calibrate with respect to    
  for d=1:size(emol,2) % all calibrations
    if emol(d)~=2
    name = sprintf('calibration/parax%d.txt',emol(d));
    par_x = load(name);
    name = sprintf('calibration/paray%d.txt',emol(d));
    par_y = load(name);
    end
    
    check = 1;
    
    try
    img = imread(fileList{ii});
    img = imresize(img,[48 48]);
    cmtname = strrep(fileList{ii},'data','cmt');
    cmtname = strrep(cmtname,'.png','.txt');
    
    newface1 = sprintf('testa//PID%d_CLEAN1_IID%d.jpg',cnt,cnt2);
    cnt2 = cnt2 + 1;
    newland1 = sprintf('testa//PID%d_CLEAN0_IID%d.jpg',cnt,cnt2);
    cnt2 = cnt2 + 1;
    cnt = cnt + 1;
    
    listid{d} = newface1;
    listid2{d} = newland1;
    
    cmtxy = load(cmtname);
    
    minx = min(cmtxy(:,1));
    miny = min(cmtxy(:,2));
    maxx = max(cmtxy(:,1));
    maxy = max(cmtxy(:,2));  
    
    img2 = img(miny:maxy,minx:maxx,:);
    img2 = imresize(img2,[200 200]);
   
    if emol(d)~=2
    % calibrate the smile cmt
    % do calibration for img 1
    cmtxy_new = cmtxy;

    for i=1:size(cmtxy,1)
   
       cur = [cmtxy(i,2)^2 cmtxy(i,1)^2 cmtxy(i,2)*cmtxy(i,1) cmtxy(i,2) cmtxy(i,1) 1];
       predx = dot(par_x,cur);
       predy = dot(par_y,cur);
       cmtxy_new(i,:) = [predx predy];
       
    end
    
    cmtxy = floor(cmtxy_new);
    
    minx = min(cmtxy(:,1));
    miny = min(cmtxy(:,2));
    maxx = max(cmtxy(:,1));
    maxy = max(cmtxy(:,2));  
    
    end
    
    %create landmark image
    h = figure('visible','off');
    img3 = ones(size(img));
    imshow(img3);
    hold on
    scatter(cmtxy(:,1),cmtxy(:,2),1)
    name = sprintf('land%d_%d',cnt,cnt2);
    export_fig(h,name, '-a1', '-jpg');
    
    img4 = imread(strcat(name,".jpg"));
    img5 = imresize(img4,size(img,1:2));
    img6 = img5(miny:maxy,minx:maxx,:);
    %imshow(img6);
    img6 = imresize(img6,[200 200]);
    catch 
       for k=1:length(listid)
          delete(listid{k});
       end
       
       for k=1:length(listid2)
          delete(listid2{k});
       end
       check = 0;
       break;
    end
    if check==1
      imwrite(img2, newface1);
      imwrite(img6, newland1);
      delete("land*");    
    end   
    close all
  end

  if check == 1
     labelc = [labelc; emol'];
     labelt = [labelt; ones(size(emol,2),1)*emo];
  end
  
end

end

labelc = labelc(2:end,:);
labelt = labelt(2:end,:);

dlmwrite('alabelsc.txt',labelc);
dlmwrite('alabelst.txt',labelt);
