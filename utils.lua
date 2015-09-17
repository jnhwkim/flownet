function rgbdPath(task, silent)
    task = task or '360'
    fh,err = io.popen("uname 2>/dev/null","r")
    if fh:read() == 'Linux' then
        dataRoot = '/opt2/data/rgbd'
    else
        dataRoot = '/Volumes/Oculus/data/rgbd'
    end
    dataPath = dataRoot..'/rgbd_dataset_freiburg2_pioneer_'..task
    if not silent then print('Source: '..dataPath) end
    return dataPath
end