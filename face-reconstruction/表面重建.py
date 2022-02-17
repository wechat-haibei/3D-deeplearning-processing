
import open3d as o3d
import numpy as np

# -------------------------- 定义点云体素化函数 ---------------------------
def get_mesh(_relative_path):
    mesh = o3d.io.read_triangle_mesh(_relative_path)
    mesh.compute_vertex_normals()
    return mesh

# -------------------- 泊松表面重建  ------------------
# 加载点云
_relative_path = "p1.ply"    # 设置相对路径
N = 5000                        # 将点划分为N个体素
pcd = get_mesh(_relative_path).sample_points_poisson_disk(N)
pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # 使现有法线无效

# 法线估计
pcd.estimate_normals()
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
pcd.orient_normals_consistent_tangent_plane(100)
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# 泊松重建
print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
print(mesh)
o3d.visualization.draw_geometries([mesh])