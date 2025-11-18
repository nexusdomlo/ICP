import open3d as o3d
import numpy as np
import sys

def calculate_average_spacing(pcd):
    """
    计算点云的平均点间距。
    """
    # Open3D 有一个内置函数来计算每个点到其最近邻的距离
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    
    print("\n========================================")
    print(f"点云的平均点间距是: {avg_dist:.6f}")
    print("========================================")
    print("\n建议:")
    print(f"这是一个很好的 `voxel_size` 的初始参考值。")
    print(f"你可以从 voxel_size = {avg_dist:.4f} 开始尝试，")
    print(f"或者设为它的几倍，例如 {avg_dist*5:.4f} 或 {avg_dist*10:.4f}，")
    print("然后根据配准效果和速度进行调整。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python calculate_spacing.py <你的点云文件.pcd>")
        sys.exit(1)
        
    pcd_path = sys.argv[1]
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        if not pcd.has_points():
            raise ValueError("点云为空。")
        print(f"成功加载点云: {pcd_path}")
    except Exception as e:
        print(f"加载点云失败: {e}")
        sys.exit(1)
        
    calculate_average_spacing(pcd)

# 计算点间距