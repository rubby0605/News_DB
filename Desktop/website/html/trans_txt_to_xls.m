function trans_txt_to_xls(elot,clot,demopath)
%trans_txt_to_xls("E123","C123","Data/test/")
filename = strcat(elot,'_',clot,'.xls');
list = dir(strcat(demopath,'',elot,'*.txt'));
mat1 = [];
mat2 = [];
for ii = 1 : length(list)
    file = list(ii).name;
    fid = fopen(strcat(demopath,'/',file),'r')
    jj = 0;
    while ~feof(fid)
        line = fgetl(fid);
        mat = sscanf(line,'%d %d');
        mat1 = [mat1; mat(1)];
        mat2 = [mat2; mat(2)];
    end
    fclose(fid)
end
T = table(mat1, mat2);
fullfilename = strcat(demopath,'\',filename);
writetable(T,fullfilename,'Sheet',clot);
excelFileName = fullfilename;
excelFilePath = demopath; % Current working directory.
sheetName = 'Sheet'; % EN: Sheet, DE: Tabelle, etc. (Lang. dependent)
