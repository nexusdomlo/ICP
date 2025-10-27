# ...existing code...
import open3d as o3d
import numpy as np
from open3d import core as o3c
import open3d.t.pipelines.registration as treg


def downsample_pcd(in_path, out_path, voxel_size=0.02):
    pcd = o3d.io.read_point_cloud(in_path)
    if pcd.is_empty():
        raise RuntimeError(f"读取点云失败或为空: {in_path}")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # 根据后缀保存 .pcd/.ply，否则保存为 npy
    if out_path.lower().endswith(('.pcd', '.ply')):
        o3d.io.write_point_cloud(out_path, pcd_down)
    else:
        np.save(out_path, np.asarray(pcd_down.points))
    print("saved", out_path, "count", len(pcd_down.points))

def load_pcd(path):
    # 支持 .pcd/.ply/.xyz/.ply 等图形文件或 .npy 点阵文件
    if path.lower().endswith(('.pcd', '.ply', '.xyz', '.xyzn', '.pts')):
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise RuntimeError(f"读取点云失败或为空: {path}")
        return pcd
    elif path.lower().endswith('.npy'):
        pts = np.load(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd
    else:
        raise ValueError("不支持的文件格式: " + path)
# ...existing code...

def crop_target_by_source(src, tgt, expand=1.2):
    aabb = src.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = aabb.get_extent() * expand
    min_b = center - extent / 2.0
    max_b = center + extent / 2.0
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
    return tgt.crop(crop_box)

def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) == 0:
        return None, None
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    return pcd_down, fpfh

def global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_icp(src, tgt, init_trans, voxel_size):
    distance_threshold = voxel_size * 0.4
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def refine_icp_gpu(src, tgt, init_trans, voxel_size):
    """
    使用 Open3D 的 Tensor API 在 GPU 上执行 ICP。
    """
    # 检查是否有可用的 CUDA 设备
    if not o3d.core.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your Open3D installation and CUDA setup.")

    device = o3c.Device("CUDA:0")
    
    # 将初始变换矩阵转换为 GPU 上的张量
    init_trans_tensor = o3c.Tensor(init_trans, device=device)
     # 将 open3d.geometry.PointCloud 转换为 open3d.t.geometry.PointCloud
    # 并将其发送到 GPU
    src_t = o3d.t.geometry.PointCloud.from_legacy(src, o3c.float64, device)
    tgt_t = o3d.t.geometry.PointCloud.from_legacy(tgt, o3c.float64, device)
    # 在 GPU 上估计法线
    src_t.estimate_normals()
    tgt_t.estimate_normals()
    # 设置 ICP 的收敛标准
    criteria = treg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30)
    # 在 GPU 上运行 ICP
    result = treg.icp(
        src_t, 
        tgt_t, 
        voxel_size * 0.4,  # 对应 CPU 版本中的 distance_threshold
        init_trans_tensor,
        treg.TransformationEstimationPointToPlane(),
        criteria
    )
    # 将结果从 GPU 张量转换回 NumPy 数组
    return result.transformation.cpu().numpy(), result.fitness, result.inlier_rmse


def register_with_prior(src_path, tgt_path, voxel_size=0.02, crop_expand=1.5, do_global=True, skip_crop=False, prior_transform=None, use_gpu=False):
    src = load_pcd(src_path)
    tgt = load_pcd(tgt_path)
    # 如果 skip_crop 为 True，表示目标已经是裁剪好的子集
    if not skip_crop:
        src_down_for_crop = src.voxel_down_sample(max(voxel_size, 0.01))
        tgt_cropped = crop_target_by_source(src_down_for_crop, tgt, expand=crop_expand)
        if len(tgt_cropped.points) == 0:
            raise RuntimeError("裁剪后目标点云为空，检查 crop_expand 或点云位置")
    else:
        tgt_cropped = tgt
    print("cropped target count:", len(tgt_cropped.points))

    # 预处理（下采样 + FPFH）
    src_down, src_fpfh = preprocess(src, voxel_size)
    tgt_down, tgt_fpfh = preprocess(tgt_cropped, voxel_size)
    if src_down is None or tgt_down is None:
        raise RuntimeError("下采样导致点云为空，请减小 voxel_size")

    # 全局配准（可选）或质心对齐作为初始变换
    if do_global:
        print("running global registration (FPFH + RANSAC)...")
        glob = global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size)
        init_trans = glob.transformation
        print("global fitness:", glob.fitness, "inlier_rmse:", glob.inlier_rmse)
    else:
        print("using centroid alignment as init")
        T = np.eye(4)
        src_center = np.asarray(src.get_center())
        tgt_center = np.asarray(tgt_cropped.get_center())
        T[:3, 3] = tgt_center - src_center
        init_trans = T

    # 精配准（ICP）
    if use_gpu:
        print("refining with ICP on GPU...")
        try:
            T_icp, fitness, rmse = refine_icp_gpu(src, tgt_cropped, init_trans, voxel_size)
            print("GPU ICP fitness:", fitness, "rmse:", rmse)
            # 为了保持返回值结构一致，我们创建一个类似 legacy API 的结果对象
            from collections import namedtuple
            ICPResult = namedtuple('ICPResult', ['transformation', 'fitness', 'inlier_rmse'])
            result_icp = ICPResult(transformation=T_icp, fitness=fitness, inlier_rmse=rmse)
        except Exception as e:
            print(f"GPU ICP failed: {e}. Falling back to CPU.")
            result_icp = refine_icp(src, tgt_cropped, init_trans, voxel_size)
            print("CPU ICP fitness:", result_icp.fitness, "rmse:", result_icp.inlier_rmse)
    else:
        print("refining with ICP...")
        result_icp = refine_icp(src, tgt_cropped, init_trans, voxel_size)
        print("ICP fitness:", result_icp.fitness, "rmse:", result_icp.inlier_rmse)

    return result_icp.transformation, tgt_cropped, src
# ...existing code...
# 示例：先下采样（可选），然后用已知小点云位置做配准
# ...existing code...
if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="register_with_prior: src tgt (supports .pcd/.ply/.npy)")
    parser.add_argument("src", help="source 点云 (.pcd/.ply/.npy)")
    parser.add_argument("tgt", help="target 点云 (.pcd/.ply/.npy) 或 已裁剪的 .pcd")
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--crop-expand", type=float, default=1.0)
    parser.add_argument("--no-global", action="store_true", help="跳过 FPFH+RANSAC，使用质心对齐作为初始变换")
    parser.add_argument("--skip-crop", action="store_true", help="目标已是裁剪子集，跳过裁剪")
    parser.add_argument("--use-gpu", action="store_true", help="使用 GPU 版本的 ICP 进行精配准")
    args = parser.parse_args()

    do_global = not args.no_global
    T, tgt_cropped, src = register_with_prior(args.src, args.tgt,
                                              voxel_size=args.voxel,
                                              crop_expand=args.crop_expand,
                                              do_global=do_global,
                                              skip_crop=args.skip_crop,
                                              use_gpu=args.use_gpu)
    print("final transformation:\n", T)
    # 可选可视化
    src_tmp = src.transform(T)
    o3d.visualization.draw_geometries([src_tmp.paint_uniform_color([1,0,0]), tgt_cropped.paint_uniform_color([0,1,0])])
