import numpy as np
from functools import cmp_to_key

EPS = 0.00001
"""
PolyGen sorting strategy:
    First we sort vertices by coordinates (z,y,x), and then faces by minimal coordinated value (z,y,x). 
    We can just use vertex indexes as comparable value because vertices already sorted by needed strategy
"""



def _is_equal(x1, x2, eps=EPS):
    return abs(x1 - x2) <= eps


def _comparator(v1, v2):
    assert len(v1) == len(v2) == 3
    if v1[2] < v2[2]:
        return -1
    elif _is_equal(v1[2], v2[2]):
        if v1[1] < v2[1]:
            return -1
        elif _is_equal(v1[1], v2[1]):
            if v1[0] < v2[0]:
                return -1
            elif _is_equal(v1[0], v2[0]):
                return 0
            else:
                return 1
        else:
            return 1
    else:
        return 1


def _inverse_comparator(v1, v2):
    assert len(v1) == len(v2) == 3
    if v1[0] < v2[0]:
        return -1
    elif _is_equal(v1[0], v2[0]):
        if v1[1] < v2[1]:
            return -1
        elif _is_equal(v1[1], v2[1]):
            if v1[2] < v2[2]:
                return -1
            elif _is_equal(v1[2], v2[2]):
                return 0
            else:
                return 1
        else:
            return 1
    else:
        return 1


def _verts_sorting(verts):
    def comparator_with_ids_wrapper(x, y):
        return _comparator(x[0], y[0])

    verts_withs_ids = zip(verts, range(len(verts)))

    sorted_verts = sorted(verts_withs_ids, key=cmp_to_key(comparator_with_ids_wrapper))
    verts = [v[0] for v in sorted_verts]
    indexes = [v[1] for v in sorted_verts]

    return verts, indexes


def sort_verts(verts):
    verts, indexes = _verts_sorting(verts)

    return verts


def sort_verts_and_faces(verts, faces):
    verts, indexes = _verts_sorting(verts)

    old2new_ids = [None for _ in indexes]
    for new, old in enumerate(indexes):
        old2new_ids[old] = new

    new_faces = [
        [old2new_ids[id] for id in face] for face in faces
    ]

    new_faces = map(lambda x: sorted(x, reverse=False), new_faces)  # Sort at inverse order
    sorted_faces = sorted(new_faces, key=cmp_to_key(_inverse_comparator))

    return verts, sorted_faces


"""
if __name__ == '__main__':
    N = 10000

    verts, faces = sort_verts_and_faces(np.round(np.random.random([N, 3]) * 255),
                                        [[np.random.randint(0, N - 1) for _ in range(3)] for _ in range(N)])
    for v in faces:
        print(v)
"""
