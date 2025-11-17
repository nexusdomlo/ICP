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

## 检验open3d是否能够使用GPU资源
```
python verify_gpu.py
```

## 检验结果
```
python verify.py # 将你的矩阵答案放到verify.py中作为T变量
```
## 落地实验
### 高度5m数据使用demo.py运行结果
#### 使用命令对两个5m数据进行配准
```
python demo.py "C:\Abandon\PCD_Data\1117pcd\5m-30mlaihui.pcd" "C:\Abandon\PCD_Data\1117pcd\5m-30mlaihui1.pcd" --voxel 0.2 --skip-crop
```
#### 输出结果
```
cropped target count: 4471400
running global registration (FPFH + RANSAC)...
global init_trans:
 [[ 0.99961904  0.0090918  -0.02605973 -0.48376225]
 [-0.00934477  0.99991023 -0.00960204  0.2612823 ]
 [ 0.02597009  0.00984191  0.99961427  0.47834226]
 [ 0.          0.          0.          1.        ]]
global fitness: 0.8734565421186554 inlier_rmse: 0.14968548670326018
refining with ICP...
ICP fitness: 0.9598622333839888 rmse: 0.02697540228113096
final transformation:
 [[ 0.99956341  0.01876094 -0.0228257  -0.52530404]
 [-0.01898117  0.99977499 -0.00947021  0.2384029 ]
 [ 0.02264289  0.00989933  0.9996946   0.38887255]
 [ 0.          0.          0.          1.        ]]
[Open3D WARNING] [ViewControl] SetViewPoint() failed because window height and width are not set.
```
![alt text](image.png)


#### 使用命令对大点云和小点云（5m高度）进行一个配准
```
python demo.py "C:\Abandon\PCD_Data\1117pcd\5m-30mlaihui.pcd" "C:\Abandon\PCD_Data\1117pcd\dixingcaiji.pcd" --voxel 0.2 --skip-crop
```
#### 生成结果
```
cropped target count: 2965139
running global registration (FPFH + RANSAC)...
global init_trans:
 [[ 9.98177321e-01 -9.06007962e-04  6.03424794e-02  6.11457520e-01]
 [ 6.70832246e-04  9.99992102e-01  3.91749351e-03  6.21167739e-01]
 [-6.03455520e-02 -3.86987349e-03  9.98170045e-01  5.32165849e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
global fitness: 0.5234498459890183 inlier_rmse: 0.16664087937800262
refining with ICP...
ICP fitness: 0.7448728629212589 rmse: 0.034259410459597704
final transformation:
 [[ 0.99777648  0.01194947  0.06556902  0.56657819]
 [-0.01303002  0.99978586  0.01607673  0.41341183]
 [-0.06536287 -0.01689535  0.99771852  0.49001161]
 [ 0.          0.          0.          1.        ]]
```
 ![alt text](image-1.png)

### 高度2m使用demo.py运行结果

#### 使用命令（这里面2m-30mlaihui1这个数据是大数据，而目标这个是小数据，要使用大去对小，这样子才能得出图中的结果，小去对大可能很容易进入局部最优导致匹配错误）
```
python demo.py "C:\Abandon\PCD_Data\1117pcd\2m-30mlaihui1.pcd" "C:\Abandon\PCD_Data\1117pcd\2m-30mlaihui.pcd" --voxel 0.2 --skip-crop
cropped target count: 18627
running global registration (FPFH + RANSAC)...
[Open3D WARNING] Too few correspondences (2173) after mutual filter, fall back to original correspondences.
global init_trans:
 [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
global fitness: 0.0 inlier_rmse: 0.0
refining with ICP...
ICP fitness: 0.026563110658869012 rmse: 0.05070781715875687
final transformation:
 [[ 0.99993692  0.01042934  0.00416868  0.02097781]
 [-0.01043878  0.99994299  0.00224812 -0.1399351 ]
 [-0.004145   -0.0022915   0.99998878 -0.06008712]
 [ 0.          0.          0.          1.        ]]
[Open3D WARNING] [ViewControl] SetViewPoint() failed because window height and width are not set.
```

![alt text](image-2.png)


### 使用另一个命令（对laihui4和laihui5进行配准）

```
python demo.py "C:\Abandon\PCD_Data\1117pcd\2m-30mlaihui4.pcd" "C:\Abandon\PCD_Data\1117pcd\2m-30mlaihui5.pcd" --voxel 0.2 --skip-crop
cropped target count: 2261440
running global registration (FPFH + RANSAC)...
global init_trans:
 [[ 9.98092250e-01  1.71724397e-03  6.17163864e-02  6.23584085e-01]
 [-8.56272671e-04  9.99901990e-01 -1.39741906e-02  1.73153576e-01]
 [-6.17343347e-02  1.38946852e-02  9.97995897e-01  3.04776908e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
global fitness: 0.9786753838164605 inlier_rmse: 0.10936708418408482
refining with ICP...
ICP fitness: 0.9298794350327009 rmse: 0.034395540273941316
final transformation:
 [[ 0.99820709  0.00180221  0.05982765  0.70982265]
 [-0.00106636  0.99992343 -0.01232908  0.13832282]
 [-0.05984529  0.01224317  0.99813258  0.30797013]
 [ 0.          0.          0.          1.        ]]
[Open3D WARNING] [ViewControl] SetViewPoint() failed because window height and width are not set.
```

![alt text](image-3.png)

### 使用另一个命令（对2mlaihui1和2mlaihui进行一个配准）

```
python demo.py "C:\Abandon\PCD_Data\1117pcd\2mlaihui1.pcd" "C:\Abandon\PCD_Data\1117pcd\2mlaihui.pcd" --voxel 0.3 --skip-crop
cropped target count: 2231927
running global registration (FPFH + RANSAC)...
global init_trans:
 [[ 9.74359426e-01 -1.25499652e-02 -2.24646852e-01 -5.33347921e+00]
 [ 1.10866813e-02  9.99908321e-01 -7.77398929e-03  1.81078970e+00]
 [ 2.24723820e-01  5.08407169e-03  9.74409235e-01  6.28701414e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
global fitness: 0.5947928339615502 inlier_rmse: 0.20905190573242274
refining with ICP...
ICP fitness: 0.6068053846456831 rmse: 0.04127966828045808
final transformation:
 [[ 0.9751606  -0.01685235 -0.22085697 -5.47795198]
 [ 0.01231549  0.99968424 -0.02190306  2.03630022]
 [ 0.22115635  0.01863904  0.97506023  0.80849428]
 [ 0.          0.          0.          1.        ]]

```
![alt text](image-4.png)

warning 问题很大，可能是因为RANSAC的偏差导致的，如果我们把两个输入对调可能导致问题，可以尝试加大降采样，修改ransac或者其他方式

错误解
```
python demo.py "C:\Abandon\PCD_Data\1117pcd\2mlaihui.pcd" "C:\Abandon\PCD_Data\1117pcd\2mlaihui1.pcd" --voxel 0.1 --skip-crop
PS C:\Abandon\Code\Python\ICP> python demo.py "C:\Abandon\PCD_Data\1117pcd\2mlaihui.pcd" "C:\Abandon\PCD_Data\1117pcd\2mlaihui1.pcd" --voxel 0.1 --skip-crop
cropped target count: 2310347
running global registration (FPFH + RANSAC)...
global init_trans:
 [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
global fitness: 0.0 inlier_rmse: 0.0
refining with ICP...
ICP fitness: 0.0007383754038550544 rmse: 0.029804554825176396
final transformation:
 [[ 0.99948467  0.02999615  0.01142883 -0.255581  ]
 [-0.0301125   0.999495    0.01014832 -0.7019945 ]
 [-0.01111865 -0.01048724  0.99988319 -0.26612117]
 [ 0.          0.          0.          1.        ]]

```