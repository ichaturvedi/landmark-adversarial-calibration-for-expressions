% gold standard
img_sce = imread('gold/surprise_img270.png');
cmtxy = load('gold/surprise_img270.txt');
cal_x = cmtxy(:,1);
cal_y = cmtxy(:,2);

% calibrate with a normal image
img_eye = imread('normal/surprise_img82.png');   
cmtxy = load('normal/surprise_img82.txt');
eye_x = cmtxy(:,1);
eye_y = cmtxy(:,2);

A = zeros(size(eye_x,1),6);

for i=1:size(eye_x,1)
  A(i,:) = [eye_y(i)^2 eye_x(i)^2 eye_y(i)*eye_x(i) eye_y(i) eye_x(i) 1]; 
end

[ua, da, va] = svd(A);
b1 = ua'*cal_x;
b1 = b1(1:6);
par_x = va*(b1./diag(da));
b2 = ua'*cal_y;
b2 = b2(1:6);
par_y = va*(b2./diag(da));

dlmwrite('parax_surprise.txt',par_x');
dlmwrite('paray_surprise.txt',par_y');

% do calibration for img 1
eye_x_new = eye_x;
eye_y_new = eye_y;

for i=1:size(eye_x,1)
   
    cur = [eye_y(i)^2 eye_x(i)^2 eye_y(i)*eye_x(i) eye_y(i) eye_x(i) 1];
    predx = dot(par_x,cur);
    predy = dot(par_y,cur);
    
    eye_x_new(i) = predx;
    eye_y_new(i) = predy;
end

% draw old
minx = min(eye_x(:)-5);
miny = min(eye_y(:)-5);
maxx = max(eye_x(:)+5);
maxy = max(eye_y(:)+5);  
    
%create landmark image
h = figure
img3 = ones(size(img_eye));
%imshow(img_eye);
imshow(img3);
hold on
scatter(eye_x,eye_y,'r')
% draw new
scatter(eye_x_new,eye_y_new,'g')
