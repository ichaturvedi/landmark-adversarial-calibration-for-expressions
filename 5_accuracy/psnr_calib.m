labels = load('alabelsc.txt');
labelst = load('alabelst.txt');

noclass = 3;
nosam = 36;

psnra = zeros(1,noclass*nosam);

fid = fopen('testfile.txt');
data = textscan(fid,'%s');
names = data{1,1};
fclose(fid);

for ii=1:noclass*nosam
    name1 = sprintf("samplestest\\%d.jpg",ii);
    name2 = names{ii};
    
    img = imread(name1);
    target = img(:,1:64,:);
    predicted = img(:,64*3+1:256,:);
    
    for j=1:noclass*nosam
        idx = sprintf("PID%d_",j);
        if ~isempty(strfind(name2, idx)) 
            psnra(j) = getPSNR(target,predicted)
        end
    end
    
end

a = reshape(labels,noclass,nosam);
b = reshape(psnra,noclass,nosam);
c = reshape(labelst,noclass,nosam);
bold = b;


[val, ind] = max(b);

for i=1:nosam
    pred(i) = a(ind(i),i);
end

truel = c(1,:);
sum(pred==truel)
sum(b(1,:)==0)


dlmwrite('preda.txt',pred)
