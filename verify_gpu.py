import open3d as o3d
import open3d.core as o3c
import open3d.t.pipelines.registration as treg
import numpy as np
import time

def create_large_point_cloud(n_points=2_000_000):
    """创建一个包含两百万个点的大规模合成点云。"""
    print(f"正在生成一个包含 {n_points:,} 个点的大点云...")
    points = np.random.rand(n_points, 3)
    points[:, 2] *= 0.1  # 使其形状像一个扁平的平面
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def stress_test_gpu_icp():
    """
    在 GPU 上执行一个持续且高负载的 ICP 任务，以便在系统监视器中清晰地看到资源使用情况。
    """
    print("--- Open3D ICP 的 GPU 压力测试 ---")

    # 1. 验证 CUDA 是否可用
    if not o3c.cuda.is_available():
        print("[失败] 您的 Open3D 安装不支持 CUDA。无法继续进行 GPU 压力测试。")
        return

    device = o3c.Device("CUDA:0")
    print(f"成功找到 CUDA 设备: {device}")

    # 2. 创建大型源点云和目标点云
    src = create_large_point_cloud()
    
    # 创建一个稍微变换过的目标点云
    T_target = np.array([
        [1, 0, 0, 0.1],
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.05],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    tgt = o3d.geometry.PointCloud(src) # 制作一个副本
    tgt.transform(T_target)

    # 3. 转换为 GPU 张量
    print("正在将点云数据移动到 GPU 内存...")
    src_t = o3d.t.geometry.PointCloud.from_legacy(src, o3c.float64, device)
    tgt_t = o3d.t.geometry.PointCloud.from_legacy(tgt, o3c.float64, device)
    
    # 4. 准备开始压力测试
    print("\n准备开始 GPU 计算。请立即切换到您的任务管理器并观察 GPU 性能图表。")
    print("计算将在 5 秒后开始...")
    time.sleep(5)

    # 5. 在循环中执行高强度 ICP 计算
    print("--- 开始高强度计算！---")
    iterations = 10
    start_time = time.time()
    for i in range(iterations):
        print(f"正在进行第 {i + 1}/{iterations} 轮 ICP 计算...")
        # 每次都用一个稍微不同的初始变换来运行
        init_trans = np.eye(4, dtype=np.float64)
        init_trans[0, 3] = i * 0.01 
        
        treg.icp(
            src_t, tgt_t, 0.02, o3c.Tensor(init_trans, device=device),
            treg.TransformationEstimationPointToPoint(),
            treg.ICPConvergenceCriteria(max_iteration=5) # 每次迭代内部也做几次
        )
    end_time = time.time()
    print(f"--- 高强度计算完成！总耗时: {end_time - start_time:.2f} 秒 ---")

if __name__ == "__main__":
    stress_test_gpu_icp()

