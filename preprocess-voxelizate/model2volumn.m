function volumn = model2volumn(model,volumn_size)
    % volumn_size必须为整数
    volumn_size = round(volumn_size);
    volumn_size = [volumn_size, volumn_size, volumn_size];
    % 按列划分顶点和三角面
    vertices = struct();
    vertices.x = double(model.vertices(1,:));
    vertices.y = double(model.vertices(2,:));
    vertices.z = double(model.vertices(3,:));

    faces = struct();
    faces.v1 = double(model.faces(:,1));
    faces.v2 = double(model.faces(:,2));
    faces.v3 = double(model.faces(:,3));

    % 得到double类型的顶点和三角面的顶点坐标
    % % double_vertices为N×3的矩阵
    % % face_vertices为结构体{'v1': N×3, 'v2': N×3, 'v3': N×3}
    % % 上述N×3矩阵结构相同，每行为点坐标[x,y,z]
    double_vertices = [vertices.x',vertices.y',vertices.z'];
    face_vertices = struct();
    face_vertices.v1 = double_vertices(faces.v1,:);
    face_vertices.v2 = double_vertices(faces.v2,:);
    face_vertices.v3 = double_vertices(faces.v3,:);

    % 将volumn_size转换为double，以便处理
    volumn_size = double(volumn_size);
    % 体素化处理
    volumn = process_model(face_vertices, volumn_size);
end

function volumn = process_model(face_vertices, volumn_size)
    % 生成空体素
    volumn = false(volumn_size);
    % 细分三角面
    face_vertices_splited = split_faces(face_vertices);
    % 拼接顶点坐标矩阵，并去除重复行
    splited_vertices = [face_vertices_splited.v1;face_vertices_splited.v2;face_vertices_splited.v3];
    splited_vertices = unique(splited_vertices, 'rows', 'stable');
    % scatter3(splited_vertices(:,1),splited_vertices(:,2),splited_vertices(:,3));
    % 得到点坐标对应的体素坐标
    voxel_vertices = [floor(splited_vertices(:,1))+1,floor(splited_vertices(:,2))+1,floor(splited_vertices(:,3))+1];
    voxel_vertices = unique(voxel_vertices, 'rows', 'stable');
    % 分别对 voxel_vertices 的每一列应用边界限制
    voxel_vertices(:,1) = max(min(voxel_vertices(:,1), volumn_size(1)), 1);
    voxel_vertices(:,2) = max(min(voxel_vertices(:,2), volumn_size(2)), 1);
    voxel_vertices(:,3) = max(min(voxel_vertices(:,3), volumn_size(3)), 1);
    % scatter3(voxel_vertices(:,1),voxel_vertices(:,2),voxel_vertices(:,3),'filled', 'd');
    % hold on;
    % 将向量转换为线性索引，利用索引为对应体素位赋值
    linear_indices = sub2ind(size(volumn), voxel_vertices(:,1), voxel_vertices(:,2), voxel_vertices(:,3));
    volumn(linear_indices) = true;
end

function face_vertices_splited = split_faces(face_vertices)
    splited_faces = struct('v1',[],'v2',[],'v3',[]);
    while(~isempty(face_vertices.v1))
        disp(['remain faces: ', num2str(length(face_vertices.v1))]);
        % 距离矩阵dist，每列分别代表一个边距离[(v1-v2),(v2-v3),(v1-v3)]，初始化为全零矩阵
        dist = zeros(length(face_vertices.v1(:,1)),3);
        % 分别计算x^2，y^2，z^2并求和，得到距离
        for i=1:3
            dist(:,1) = dist(:,1) + (face_vertices.v1(:,i)-face_vertices.v2(:,i)).^2;
            dist(:,2) = dist(:,2) + (face_vertices.v2(:,i)-face_vertices.v3(:,i)).^2;
            dist(:,3) = dist(:,3) + (face_vertices.v1(:,i)-face_vertices.v3(:,i)).^2;
        end
        % 得到三边最大值，构成列向量
        maxdist = max(dist,[],2);
        % 得到最大边长大于0.5个体素的三角面，在can_split列向量对应行标记为1
        can_split = maxdist > 0.5;
        splited_faces.v1 = [splited_faces.v1;face_vertices.v1(~can_split,:)];
        splited_faces.v2 = face_vertices.v2(~can_split,:);
        splited_faces.v3 = face_vertices.v3(~can_split,:);
        % 更新face_vertices和dist，去除无法再分割的三角面
        face_vertices_temp = face_vertices;
        face_vertices_temp.v1 = face_vertices.v1(can_split,:);
        face_vertices_temp.v2 = face_vertices.v2(can_split,:);
        face_vertices_temp.v3 = face_vertices.v3(can_split,:);
        face_vertices = face_vertices_temp;
        dist = dist(can_split,:);
        % 计算仍能分割的三角面的三边中点
        mid_vertices = struct();
        for i=1:3
            mid_vertices.v12(:,i) = (face_vertices.v1(:,i)+face_vertices.v2(:,i))/2;
            mid_vertices.v23(:,i) = (face_vertices.v2(:,i)+face_vertices.v3(:,i))/2;
            mid_vertices.v13(:,i) = (face_vertices.v1(:,i)+face_vertices.v3(:,i))/2;
        end
        % 计算得到各边最长的三角面
        max_v12 = dist(:,1) >= dist(:,2) & dist(:,1) >= dist(:,3);
        max_v23 = dist(:,2) >= dist(:,1) & dist(:,2) >= dist(:,3) & ~max_v12;
        max_v13 = dist(:,3) >= dist(:,1) & dist(:,3) >= dist(:,2) & ~max_v12 & ~max_v23;
        % 左替换最长边结点为中点
        left_split = face_vertices;
        left_split.v1(max_v12,:) = mid_vertices.v12(max_v12,:);
        left_split.v2(max_v23,:) = mid_vertices.v23(max_v23,:);
        left_split.v1(max_v13,:) = mid_vertices.v13(max_v13,:);
        % 右替换最长边结点为中点
        right_split = face_vertices;
        right_split.v2(max_v12,:) = mid_vertices.v12(max_v12,:);
        right_split.v3(max_v23,:) = mid_vertices.v23(max_v23,:);
        right_split.v3(max_v13,:) = mid_vertices.v13(max_v13,:);
        % 拼接左右替换
        face_vertices.v1 = [left_split.v1;right_split.v1];
        face_vertices.v2 = [left_split.v2;right_split.v2];
        face_vertices.v3 = [left_split.v3;right_split.v3];
    end
    face_vertices_splited = splited_faces;
end