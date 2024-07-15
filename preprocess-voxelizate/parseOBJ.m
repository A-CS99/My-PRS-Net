function [vertices, faces, surf_samples] = parseOBJ(obj_path, sample_num)
    try
        % 读取obj文件的顶点和三角面
        % % vertices中元素为点坐标：[x,y,z]
        % % faces中元素为面的三个点序号: [v1,v2,v3]
        [vertices, faces, ~] = readOBJ(obj_path);
        % 从obj表面随机选取sample_num个点
        % % surf_samples中元素为点坐标: [x,y,z]
        [surf_samples, ~] = meshlpsampling(obj_path, sample_num);
    catch
        ME = MException('ParseException::FailRead', 'Fail to read obj %s', obj_path);
        throw(ME);
    end
    if (isempty(vertices) || isempty(faces))
        ME = MException('ParseException::EmptyResult', 'Get no vertex or face from obj %s', obj_path);
        throw(ME);
    end
end