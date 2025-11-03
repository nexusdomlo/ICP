import open3d as o3d
import numpy as np

def verify_gpu_usage():
    """
    检查 Open3D 是否可以检测并使用 CUDA GPU。
    """
    print(f"Open3D version: {o3d.__version__}")
    
    # 1. 检查可用的 CUDA 设备
    cuda_devices = o3d.core.cuda.is_available()
    if not cuda_devices:
        print("未找到可用的 CUDA 设备。")
        print("请确认：")
        print("1. 您的系统中已正确安装 NVIDIA 驱动。")
        print("2. 您安装的 Open3D 版本是使用 CUDA 编译的。")
        print("3. 您的 GPU 具有 CUDA 计算能力。")
        return

    print(f"找到 {len(cuda_devices)} 个可用的 CUDA 设备:")
    for i, device in enumerate(cuda_devices):
        print(f"  - 设备 {i}: {device}")

    # 2. 尝试在 GPU 上创建一个张量
    try:
        device = o3d.core.Device("CUDA:0")
        print(f"\n尝试在 {device} 上创建张量...")
        
        # 创建一个简单的点云
        points = np.random.rand(100, 3)
        pcd_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32, device=device)
        
        print("成功在 GPU 上创建张量！")
        print("张量设备:", pcd_tensor.device)
        
        # 3. 演示一个简单的 GPU 操作
        print("\n在 GPU 上执行降采样操作...")
        pcd_t = o3d.t.geometry.PointCloud(pcd_tensor)
        pcd_downsampled_t = pcd_t.voxel_down_sample(voxel_size=0.1)
        
        print("GPU 降采样完成。")
        print(f"原始点数: {len(pcd_t.point.positions)}")
        print(f"降采样后点数: {len(pcd_downsampled_t.point.positions)}")

    except Exception as e:
        print("\n在尝试使用 GPU 时发生错误:")
        print(e)

if __name__ == "__main__":
    print(o3d._build_config)
    verify_gpu_usage()


# import torch

# def check_pytorch_cuda():
#     """
#     使用 PyTorch 检查 CUDA 环境是否可用。
#     """
#     print(f"PyTorch version: {torch.__version__}")
    
#     # 1. 检查 CUDA 是否对 PyTorch 可用
#     is_available = torch.cuda.is_available()
#     print(f"PyTorch CUDA available: {is_available}")
    
#     if not is_available:
#         print("\nPyTorch 未能找到可用的 CUDA 设备。")
#         print("这通常意味着您的 NVIDIA 驱动或 CUDA Toolkit 安装存在问题。")
#         print("请检查：")
#         print("1. NVIDIA 驱动是否已正确安装并正在运行。")
#         print("2. CUDA Toolkit 是否已安装，并且其路径已添加到系统环境变量中。")
#         print("3. 您安装的 PyTorch 版本是否支持 CUDA。")
#     else:
#         # 2. 如果可用，打印详细信息
#         device_count = torch.cuda.device_count()
#         print(f"Found {device_count} CUDA device(s).")
        
#         current_device_index = torch.cuda.current_device()
#         current_device_name = torch.cuda.get_device_name(current_device_index)
#         print(f"Current CUDA device: {current_device_index} - {current_device_name}")
        
#         # 3. 打印 PyTorch 编译时使用的 CUDA 版本
#         torch_cuda_version = torch.version.cuda
#         print(f"PyTorch was compiled with CUDA version: {torch_cuda_version}")

# if __name__ == "__main__":
#     check_pytorch_cuda()

