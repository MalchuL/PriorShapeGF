import trimesh
import trimesh.repair as repair
import trimesh.sample
import trimesh.proximity
import numpy as np


class ThreeDPreprocessor:
    @staticmethod
    def fix_mesh(mesh):
        #mesh.invert()
        mesh.remove_duplicate_faces()
        repair.fix_normals(mesh, True)
        repair.fix_winding(mesh)
        return mesh

    @staticmethod
    def _get_mesh_center(mesh):
        size = mesh.bounds
        size = size[1] + size[0]
        return size / 2

    @staticmethod
    def normalize_mesh(mesh):
        try:
            mesh.vertices -= ThreeDPreprocessor._get_mesh_center(mesh)  # mesh.vertices.mean(axis=0) #mesh.center_mass#
            mesh.vertices *= 1/mesh.scale
        except Exception as e:
            print("Can't normalize", e)
        return mesh


class ThreeDUtils:

    @staticmethod
    def sample_surface_points(mesh, count, noise, sample_even = False):
        if sample_even:
            sampled1 = trimesh.sample.sample_surface_even(mesh, count)[0]
        else:
            sampled1 = np.zeros([0,3])
        try:
            sampled2 = trimesh.sample.sample_surface(mesh, count - sampled1.shape[0])[0]
        except:
            sampled2 = np.zeros([0,3])
        sampled = np.concatenate([sampled1, sampled2], 0)[:count,:]
        sampled = sampled + np.random.uniform(-noise, noise, sampled.shape)

        return sampled

    @staticmethod
    def find_closed_triangle(mesh, points):
        points, distance, triangle_id = trimesh.proximity.closest_point(mesh, points)

        triangles = mesh.triangles[triangle_id]
        triangles_center = mesh.triangles_center[triangle_id]
        return points, triangles, triangles_center

    @staticmethod
    def calculate_sdf(mesh, points):
        query = trimesh.proximity.ProximityQuery(mesh)
        sdfs = query.signed_distance(points)
        return -sdfs #because calculation with negative sign


