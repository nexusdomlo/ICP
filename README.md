# ICP: A method for registration of 3-D shapes
一个普通的ICP算法，但是由于我的设备时jetson agx orin Ubuntu22.04版本，CPU远远弱于GPU，所以将ICP算法做了GPU加速
要实现GPU加速就要去自己编译C++库,但是编译一堆问题，所以用回CPU吧

## 运行
```
python demo.py "C:\Abandon\PCD_Data\data\data_2_cut.pcd" "C:\Abandon\PCD_Data\data\data_2_cut_transformed.pcd" --skip-crop
第一个参数是source点云，第二个参数是target点云

--voxel             "表示降采样的程度，值越大，点越少"
--crop-expand
--no-global         "跳过FPFH+RANSAC，使用质心对齐作为初始变换"
--skip-crop         "目标已是裁剪子集，跳过裁剪"
--use-gpu           "使用 GPU 版本的 ICP 进行精配准,还在开发中" 
```

## 检验是否能够使用Cuda，即是否能够使用GPU资源加速计算
```
python testCuda.py
```
