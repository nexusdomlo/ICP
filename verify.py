import numpy as np
import open3d as o3d

src = o3d.io.read_point_cloud("C:\\Abandon\\PCD_Data\\data_2_cut.pcd")
tgt = o3d.io.read_point_cloud("C:\\Abandon\\PCD_Data\\data_2_cut_transformed.pcd")

# 把这里的 T 替换为脚本输出的矩阵
T = np.array([[ 1.00000000e+00,  1.00751683e-16,  9.36002697e-17, -2.85687673e-17],
               [ 5.20209367e-17, -5.96933695e-17,  1.00000000e+00, -1.63241769e-16],
               [ 8.62027715e-18, -1.00000000e+00,  9.87865859e-17, -2.81162291e-16],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

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