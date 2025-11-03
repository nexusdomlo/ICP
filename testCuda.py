# import open3d as o3d
# import open3d.core as o3c
# import numpy as np

# print(f"Open3D版本: {o3d.__version__}")
# print(f"CUDA是否可用: {o3d.core.cuda.is_available()}")

# # 测试CUDA张量
# if o3d.core.cuda.is_available():
#     device = o3c.Device("CUDA:0")
#     x = o3c.Tensor([1, 2, 3], device=device)
#     print(f"张量设备: {x.device}")
#     print("CUDA加速已成功启用！")
# else:
#     print("CUDA不可用，将使用CPU模式")


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


# import open3d as o3d
# from open3d import core as o3c

# print(o3c.Device("CUDA:0"))


# import open3d as o3d
# import sys

# def check_open3d_cuda():
#     """
#     检查当前环境中的 Open3D 是否支持 CUDA。
#     """
#     print(f"Python executable: {sys.executable}")
#     print(f"Open3D version: {o3d.__version__}")

#     try:
#         # Open3D 0.10.0 及以上版本引入了基于张量的API
#         # 我们可以通过检查可用设备来判断CUDA是否支持
#         available_devices = o3d.core.Device.get_available_devices()
#         print(f"Available devices: {available_devices}")

#         is_cuda_available = any(
#             device.get_type() == o3d.core.Device.DeviceType.CUDA
#             for device in available_devices
#         )

#         if is_cuda_available:
#             print("\n[结论] 恭喜！您的 Open3D 安装已成功检测到 CUDA，可以进行 GPU 加速。")
#         else:
#             print("\n[结论] 您的 Open3D 安装未检测到 CUDA 设备。将仅使用 CPU。")
#             print("提示：如果需要 GPU 加速，请确保您已安装支持 CUDA 的 Open3D 版本，并正确配置了 NVIDIA 驱动和 CUDA Toolkit。")

#     except AttributeError:
#         # 兼容旧版本 Open3D
#         print("\n[警告] 无法使用 o3d.core.Device。您的 Open3D 版本可能较旧。")
#         print("对于旧版本，CUDA 支持通常在编译时确定。如果从源码编译时启用了 CUDA，则支持 GPU。")
#     except Exception as e:
#         print(f"\n[错误] 在检查过程中发生未知错误: {e}")

# if __name__ == "__main__":
#     check_open3d_cuda()


# import torch

# # 方法1：检查CUDA是否可用
# print(torch.cuda.is_available())

# # 方法2：检查GPU数量
# print(f"GPU数量: {torch.cuda.device_count()}")

# # 方法3：获取当前GPU名称
# if torch.cuda.is_available():
#     print(f"当前GPU: {torch.cuda.get_device_name(0)}")


# import open3d as o3d
# import sys

# def check_o3d_cuda():
#     """
#     检查 Open3D 是否能找到并使用 CUDA 设备。
#     """
#     print(f"Open3D version: {o3d.__version__}")
#     print(f"Python version: {sys.version}")

#     try:
#         # 尝试创建一个 CUDA 设备对象
#         device = o3d.core.Device("CUDA:0")
#         print(f"成功定位到 Open3D CUDA 设备: {device.get_type_name()}:{device.get_id()}")

#         # 尝试在该设备上创建一个张量
#         tensor = o3d.core.Tensor([1, 2, 3], device=device)
#         print("成功在 CUDA 设备上创建 Open3D 张量。")
#         print("\n结论：您的 Open3D 安装已正确启用 CUDA 支持！")

#     except Exception as e:
#         print("\n无法在 CUDA 设备上创建 Open3D 张量。")
#         print(f"错误信息: {e}")
#         print("\n结论：您的 Open3D 安装可能没有 CUDA 支持，或者无法访问 GPU。")
#         print("请检查您的 NVIDIA 驱动和 Open3D 安装包是否正确。")

# if __name__ == "__main__":
#     check_o3d_cuda()
# if not o3d.core.cuda.is_available():
#         raise RuntimeError("CUDA is not available. Please check your Open3D installation and CUDA setup.")


import open3d
# print(open3d.cuda.is_cuda_available())
print(open3d.core.cuda.is_available())