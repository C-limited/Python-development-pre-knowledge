import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from sklearn import cluster, decomposition
from sklearn.datasets import fetch_olivetti_faces
from skimage import color
from skimage.transform import resize
from skimage.io import imread

rng = RandomState(0)

# 在stdout上显示进度日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
n_samples, n_features = faces.shape

# 全局中心化（聚焦于一个特征，中心化所有样本）
faces_centered = faces - faces.mean(axis=0)

# 局部中心化（聚焦于一个样本，中心化所有特征）
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("数据集包含 %d 张人脸" % n_samples)

# 加载和预处理你自己的面部图像
def load_and_preprocess_image(image_path):
    image = imread(image_path)
    image_resized = resize(image, (64, 64), anti_aliasing=True)
    image_gray = color.rgb2gray(image_resized)
    image_flattened = image_gray.flatten()
    return image_flattened.astype(np.float32)  # 确保数据类型一致

# 加载你自己的面部图像
own_face_image_path = 'path_to_your_image.jpg'  # 替换为你的面部图像路径
own_face_image = load_and_preprocess_image(own_face_image_path)

# 使用训练好的模型变换你自己的面部图像
pca_estimator = decomposition.PCA(n_components=6, svd_solver='randomized', whiten=True)
pca_estimator.fit(faces_centered)
pca_transformed = pca_estimator.transform([own_face_image.astype(faces.dtype)])

nmf_estimator = decomposition.NMF(n_components=6, tol=5e-3)
nmf_estimator.fit(faces)  # 使用原始非负数据集
nmf_transformed = nmf_estimator.transform([own_face_image.astype(faces.dtype)])

# 打印变换结果
print("PCA transformed:", pca_transformed)
print("NMF transformed:", nmf_transformed)
