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




# 好的，我们来分析一下 `Python/ICP/verify.py` 这个脚本是如何对你的ICP（Iterative Closest Point，迭代最近点）算法生成的结果进行验证的。

# 这个脚本的核心思想是：**用你的ICP算法计算出的变换矩阵 `T`，将原始点云（source）进行变换，然后计算变换后的点云与目标点云（target）之间的距离误差。误差越小，说明你的ICP算法结果越精确。**

# 下面是详细的步骤分解：

# 1.  **加载数据 (L4-L5):**
#     *   `src = o3d.io.read_point_cloud(...)`: 加载你的**源点云**（source point cloud），也就是进行配准之前的原始点云。
#     *   `tgt = o3d.io.read_point_cloud(...)`: 加载**目标点云**（target point cloud），也就是你希望源点云配准过去对齐的那个点云。

# 2.  **定义变换矩阵 (L8-L12):**
#     *   `T = np.array(...)`: 这里定义了一个4x4的齐次变换矩阵。脚本中的注释明确指出，**你需要把你自己的ICP算法运行后输出的那个变换矩阵粘贴到这里**，以替换掉当前这个示例矩阵。这个矩阵 `T` 就是你整个ICP算法的成果。

# 3.  **应用变换 (L14):**
#     *   `src.transform(T)`: 将上面定义的变换矩阵 `T` 应用于源点云 `src`。执行这行代码后，`src` 点云在内存中的坐标已经被更新，理论上它现在应该已经和 `tgt` 点云对齐了。

# 4.  **计算配准误差 (L17-L23):**
#     这是整个验证过程最核心的部分。它逐点计算了变换后的 `src` 中的每个点到 `tgt` 点云的最近距离。
#     *   `tree = o3d.geometry.KDTreeFlann(tgt)`: 为目标点云 `tgt` 创建一个KD树。KD树是一种空间索引数据结构，可以极大地加速查找最近点的过程。
#     *   `for p in src_pts:`: 循环遍历变换后 `src` 中的**每一个点**。
#     *   `tree.search_knn_vector_3d(p, 1)`: 对于 `src` 中的当前点 `p`，在 `tgt` 的KD树中搜索离它最近的1个点（`k=1`）。
#     *   `dists.append(np.sqrt(dist2[0]))`: `search_knn` 返回的是距离的平方 (`dist2`)，所以这里通过 `np.sqrt` 计算出真实的欧氏距离，并存入 `dists` 列表。

# 5.  **输出量化指标 (L26):**
#     *   `print("nearest dist mean/max/min:", dists.mean(), dists.max(), dists.min())`:
#         *   `dists.mean()`: **平均距离**。这是衡量配准质量最重要的指标之一。这个值越接近0，说明整体配准效果越好。它在概念上类似于均方根误差（RMSE）。
#         *   `dists.max()`: **最大距离**。这个值代表了所有点中最大的一个误差，可以帮你发现局部配准最差的地方。
#         *   `dists.min()`: **最小距离**。这个值通常接近0，参考意义不大。

# 6.  **保存结果用于可视化 (L29-L30):**
#     *   `o3d.io.write_point_cloud("src_transformed.pcd", src)`: 将变换后的源点云保存为一个新的PCD文件。
#     *   **目的**：你可以使用点云可视化软件（如 CloudCompare 或 Open3D 的可视化工具）同时打开这个新生成的 `src_transformed.pcd` 文件和原始的 `tgt` 文件。通过肉眼观察两个点云的重合程度，可以非常直观地判断配准效果的好坏。

# ### 总结

# 总而言之，`verify.py` 通过以下两种方式对你的ICP结果进行验证：

# *   **量化分析**：计算变换后的源点云与目标点云之间的**平均/最大/最小距离**，提供具体的数字指标来评估配准精度。
# *   **可视化分析**：生成一个变换后的点云文件，让你可以在可视化工具中直观地**观察两个点云的重合情况**。

# 这是一个非常标准且有效的点云配准结果验证流程。