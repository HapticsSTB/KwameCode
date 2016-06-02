function convertToMat(fileName)
    
    testDir = 'testdir';
    problemFiles = {};
    
    mkdir('evalMat/')
    try
        disp('Loading csv file...')
        rawData = single(csvread([ testDir, '/',  fileName]));
        disp('Saving csv file into mat...')
        save(['evalMat/' fileName(1:end-4)], 'rawData');
    catch
        disp(['Problem with ' fileName])
        problemFiles{end+1} = fileName;
    end
    clear rawData
    
    
    problemFiles
    
end