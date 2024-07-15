function [] = genData()
    % 配置生成类型 ('train' or 'test')
    type = 'train';
    % 配置路径
    % 原数据集路径
    data_set_path = '../shapenet';
    % 体素化模型保存路径
    save_path = ['../datasets/', type];
    % 数据集划分信息路径
    split_path = '../data_split';
    if exist(save_path, "dir") == 0
        mkdir(save_path);
    end

    % 配置每个模型用于计算SDE的采样数量
    sample_num = 1000;
    % 配置每个种类的目标模型数 (数量不足时对原模型做变换)
    models_need = 4000;

    % 配置体素网格的大小
    volumn_size = 32;
    % 计算体素化网格的中心点位置
    % % shapeNet模型经过规范化，点位分布于[-0.5,0.5]
    voxel_size = 1 / volumn_size;
    start_pos = -0.5 + voxel_size / 2;
    end_pos = 0.5 - voxel_size / 2;
    centers_range = start_pos:voxel_size:end_pos;
    voxel_centers = ndgrid(centers_range, centers_range, centers_range);

    % 读取shapeNet分类，从第3个开始为类别文件夹 (前两个为'.'和'..')
    cats = dir(data_set_path);
    for i=3:length(cats)
        if ((i-3) ~= 1)
            continue;
        end
        cat_id = cats(i).name;
        disp([num2str(i-3), '/', num2str(length(cats)-3), ' type: ', cat_id]);
        % 读取该分类中用于训练的模型id
        split_file = fopen([split_path, '/', cat_id, '_', type, '.txt'], 'r');
        model_ids = textscan(split_file, '%s', 'Delimiter', '\n');
        fclose(split_file);
        % 从cell中读取id数据 (结果仍为cell)
        model_ids = model_ids{1};
        % 计算平均到每个模型所需的模型变种数
        models_need_each = ceil(models_need / length(model_ids));
        for j=1:length(model_ids)
            if (j > 100)
                break;
            end
            disp([num2str(j), '/', num2str(length(model_ids)), ' model: ', model_ids{j}]);
            % 读取模型obj文件，获取顶点、三角面和1000个表面样本点
            obj_file = [data_set_path, '/', cat_id, '/', model_ids{j}, '/models/model_normalized.obj'];
            try
                [vertices, faces, surf_samples] = parseOBJ(obj_file, sample_num);
            catch e
                % 解析错误则跳过该模型
                disp(e.message);
                continue;
            end
            for k=1:models_need_each
                if (k > 10)
                    break;
                end
                disp(['rotate time: ', num2str(k), '/', num2str(models_need_each)]);
                % 生成随机的规范化旋转轴
                axis = rand(1,3);
                axis = axis/norm(axis);
                % 生成随机的旋转角度
                angle = rand(1)*2*pi;
                % 合成轴角表示
                rotate_axisangle = [axis, angle];
                % 生成旋转矩阵
                rotate_matrix = axang2rotm(rotate_axisangle);
                % 旋转所有顶点 (三角面只和顶点序号相关，无需变换)
                % % 取顶点矩阵的转置，以和旋转矩阵相乘
                r_vertices = rotate_matrix*vertices';
                r_surf_samples = rotate_matrix*surf_samples;
                % 将顶点的规范化坐标系平移缩放到体素对应的位置
                volumn_vertices = volumn_size*(r_vertices + 0.5) + 0.5;
                volumn_samples = volumn_size*(r_surf_samples + 0.5) + 0.5;
                % 将顶点和三角面组合为结构体
                model = struct();
                model.vertices = volumn_vertices;
                model.faces = faces;
                volumn = model2volumn(model, volumn_size);
%                 scatter3(volumn_samples(1,:), volumn_samples(2,:), volumn_samples(3,:), 'green');
%                 hold off;
                % 将所需变量保存
                save_name = [save_path, '/', model_ids{j}, '_r', num2str(k), '.mat'];
                save(save_name, "volumn", "volumn_vertices", "volumn_samples", "faces", "rotate_axisangle", "voxel_centers");
            end
        end
    end
    disp('training ends');
end
    