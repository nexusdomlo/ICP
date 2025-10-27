
# import open3d as o3d
# import numpy as np
# import time
# from open3d import core as o3c
# # import open3d.t.pipelines.registration as treg

# if not o3c.cuda.is_available():
#     print("⚠️ CUDA 不可用 - 请确保安装了支持 CUDA 的 Open3D")
#     gpu_device = o3c.Device("CUDA:0")  # GPU 设备
# else:
#     print("✅ CUDA 可用 - 使用 GPU 加速计算")


# # 创建测试点云（使用 Tensor API）
# gpu_device = o3d.core.Device("CUDA:0")  # GPU 设备
# cpu_device = o3d.core.Device("CPU:0")  # CPU 设备
# source_points = np.random.rand(500000, 3).astype(np.float32)
# # 为 GPU 和 CPU 创建源点云
# source_gpu = o3d.t.geometry.PointCloud(gpu_device)
# source_gpu.point.positions = o3d.core.Tensor(source_points, device=gpu_device)
# source_cpu = o3d.t.geometry.PointCloud(cpu_device)
# source_cpu.point.positions = o3d.core.Tensor(source_points, device=cpu_device)
# # GPU 计算法向量
# print("🚀 在 GPU 上计算法向量...")
# start_time_gpu = time.time()
# source_gpu.estimate_normals(max_nn=30, radius=0.1)
# gpu_time = time.time() - start_time_gpu

# # CPU 计算法向量
# print("🚀 在 CPU 上计算法向量...")
# start_time_cpu = time.time()
# source_cpu.estimate_normals(max_nn=30, radius=0.1)
# cpu_time = time.time() - start_time_cpu
# # 输出结果
# print("\n📊 性能对比：")
# print(f"🕒 GPU 处理时间：{gpu_time:.4f} 秒")
# print(f"🕒 CPU 处理时间：{cpu_time:.4f} 秒")
# print(f"🚀 加速比(CPU/GPU):{cpu_time / gpu_time:.2f}x")

# # 验证结果：检查是否生成了法向量
# has_normals_gpu = hasattr(source_gpu.point, 'normals')
# has_normals_cpu = hasattr(source_cpu.point, 'normals')
# print(f"\nGPU 法向量计算：{'成功' if has_normals_gpu else '失败'}")
# print(f"CPU 法向量计算：{'成功' if has_normals_cpu else '失败'}")


import open3d
print(open3d.__version__)
print(hasattr(open3d, "t"))