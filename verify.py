import numpy as np
import open3d as o3d

src = o3d.io.read_point_cloud("/home/brian/DCP/Cut/data_2_cut.pcd")
tgt = o3d.io.read_point_cloud("/home/brian/DCP/Cut/data_2.pcd")

# 把这里的 T 替换为脚本输出的矩阵
T = np.array([[ 1.00000000e+00,  2.78388423e-14, -1.21569421e-14,  1.84297022e-14],
              [-2.78388423e-14,  1.00000000e+00,  1.95399252e-14, -5.43662337e-14],
              [ 1.22957200e-14, -1.95121697e-14,  1.00000000e+00,  1.06581410e-14],
              [ 0.0, 0.0, 0.0, 1.0]])

src.transform(T)

# KDTree 查询每个 src 点到 tgt 的最近距离（可能较慢但准确）
tree = o3d.geometry.KDTreeFlann(tgt)
src_pts = np.asarray(src.points)
dists = []
for p in src_pts:
    _, idx, dist2 = tree.search_knn_vector_3d(p, 1)
    dists.append(np.sqrt(dist2[0]))
dists = np.array(dists)

print("src点数:", len(src_pts))
print("nearest dist mean/max/min:", dists.mean(), dists.max(), dists.min())

# 保存变换后的点云供本地可视化
o3d.io.write_point_cloud("src_transformed.pcd", src)
print("saved src_transformed.pcd")