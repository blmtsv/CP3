import numpy as np
from numba import njit, prange


def init_boids(boids: np.ndarray, asp: float, vrange: tuple[float, float]):
    """
    :param boids:
    :param asp:
    :param vrange:
    :return: an array of boids with the x and y coordinates
    filled with a uniform distribution, as well as the x and y velocities
    """
    n = boids.shape[0]
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s
    return boids


@njit()
def v_norm(a: np.ndarray):
    """
    for njit compile
    :param a: array
    :return: norm of vector
    """
    return np.sqrt(np.sum(a**2, axis=1))


@njit()
def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """
    :param boids:
    :param dt: time step
    :return: array N x (x0, y0, x1, y1) for arrow painting
    """
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))


@njit()
def vclip(v: np.ndarray, vrange: tuple[float, float]):
    """

    :param v: an array of accelerations or speeds
    :param vrange:a tuple of possible changes in magnitude
    :return: an array of velocities or accelerations in which
     those components whose norms exceeded the permissible values are normalized
    """
    norm = v_norm(v)
    mask = norm > vrange[1]
    if np.any(mask):
        v[mask] *= (vrange[1] / norm[mask]).reshape(-1, 1)


@njit()
def propagate(boids: np.ndarray,
              dt: float,
              vrange: tuple[float, float],
              arange: tuple[float, float]):
    """

    :param boids:
    :param dt: time step
    :param vrange: tuple of speed limits
    :param arange:  tuple of accelaration limits
    :return: modified array of boids after dt time
    """
    vclip(boids[:, 4:6], arange)
    boids[:, 2:4] += dt * boids[:, 4:6]
    vclip(boids[:, 2:4], vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]


@njit()
def periodic_walls(boids: np.ndarray, asp: float):
    """
    :param boids:
    :param asp: aspect ratio
    :return: Sets the position of boids with respect to periodic walls for them to not fly away
    """
    boids[:, 0:2] %= np.array([asp, 1.])


@njit()
def noise() -> np.ndarray:
    """

    :return: array of noises
    """
    arr = np.random.rand(2)
    if np.random.rand(1) > .5:
        arr[0] *= -1
    if np.random.rand(1) > .5:
        arr[1] *= -1
    return arr


@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    """
    :param boids:
    :param idx: index of boid
    :param neigh_mask: neighboors of boid
    :param perception:
    :return:where should boids go to reach the geometric center of the neighbors
    """
    arr = boids[neigh_mask, :2]
    res = np.empty(arr.shape[1], dtype=boids.dtype)
    for i in range(len(res)):
        res[i] = np.mean(arr[:, i])
    a = (res - boids[idx, :2]) / perception
    return a


@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray) -> np.ndarray:
    """
    :param boids:
    :param idx: index of boid
    :param neigh_mask: boolean mask of neighboors
    :return:coefficients for avoiding local overpopulation
    """
    neighbs = boids[neigh_mask, :2] - boids[idx, :2]
    norm = v_norm(neighbs)
    mask = norm > 0
    if np.any(mask):
        neighbs[mask] /= norm[mask].reshape(-1, 1)
    d = np.empty(neighbs.shape[1], dtype=boids.dtype)
    for i in range(len(d)):
        d[i] = np.median(neighbs[:, i])
    norm_d = np.linalg.norm(d)
    if norm_d > 0:
        d /= norm_d
    return -d


@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:
    """

    :param boids: array boid
    :param idx: index of boid
    :param neigh_mask: mask of vision
    :param vrange: tuple of speed limits
    :return:coefficients for equalizing the velocity vector by local neighbors
    """
    arr = boids[neigh_mask, 2:4]
    result = np.empty(arr.shape[1], dtype=boids.dtype)
    for i in range(arr.shape[1]):
        result[i] = np.mean(arr[:, i])
    a = (result - boids[idx, 2:4]) / (2 * vrange[1])
    return a


@njit()
def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """
    helping function
    :param edge0:
    :param edge1:
    :param x:
    :return:
    """
    x = np.clip((x - edge0) / (edge1 - edge0), 0., 1.)
    return x * x * (3.0 - 2.0 * x)


@njit()
def better_walls(boids: np.ndarray, asp: float, param: float):
    """
    returns boids to another side of window
    :param boids:
    :param asp:
    :param param:
    :return:
    """

    x = boids[:, 0]
    y = boids[:, 1]
    w = param

    a_left = smoothstep(asp * w, 0.0, x)
    a_right = -smoothstep(asp * (1.0 - w), asp, x)

    a_bottom = smoothstep(w, 0.0, y)
    a_top = -smoothstep(1.0 - w, 1.0, y)

    return np.column_stack((a_left + a_right, a_bottom + a_top))


@njit()
def normal(v: np.ndarray):
    """

    :param v: array of coordinates
    :return: normalised vector
    """
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return v
    return v / v_norm


@njit(parallel=True)
def visibility(boids: np.ndarray, perception: float, angle: float) -> np.ndarray:
    """
    find euclidian norm and angle between components of each boid and
    makes mask of vision
    :param boids: boids array
    :param perception: radius of vision
    :param angle: cos of vision angle
    :return: boolean mask of vision of every unit
    """
    vectors = boids[:, :2]
    speeds = boids[:, 2:4]
    n = vectors.shape[0]
    dist = np.zeros(shape=(n, n), dtype=np.float64)
    angles = np.zeros(shape=(n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(n):
            v = vectors[j] - vectors[i]
            d = (v @ v)
            dist[i, j] = d
            angles[i, j] = normal(v) @ normal(speeds[i])
    dist = np.sqrt(dist)
    distance_mask = dist < perception
    angle_mask = angles > angle
    mask = np.logical_and(distance_mask, angle_mask)
    np.fill_diagonal(mask, False)
    return mask



@njit()
def wall_avoidance(boids: np.ndarray, asp: float):
    """
    releases wall avoidance
    :param boids:
    :param asp: aspect ratio
    :return:
    """
    left = np.abs(boids[:, 0])
    right = np.abs(asp - boids[:, 0])
    bottom = np.abs(boids[:, 1])
    top = np.abs(1 - boids[:, 1])
    ax = 1. / left**2 - 1. / right**2
    ay = 1. / bottom**2 - 1. / top**2
    boids[:, 4:6] += np.column_stack((ax, ay))


@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             vrange: tuple,
             ang: float) -> np.ndarray:
    """
    finds boids movement with cohesion, alignment, separation, noise and walls
    :param boids: array of units
    :param perception: radius of vision
    :param coeffs:  cohesion, alignment, separation, noise and walls
    :param asp: aspect ratio
    :param vrange: tuple of speed limits
    :param ang: cos of vision angle
    :return: mask of visions for [0] boid
    """
    n = boids.shape[0]
    mask = visibility(boids, perception, ang)
    wal = better_walls(boids, asp, 0.05)
    for i in prange(n):
        if not np.any(mask[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
            ns = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            alg = alignment(boids, i, mask[i], vrange)
            sep = separation(boids, i, mask[i])
            ns = noise()
        boids[i, 4] = (coeffs[0] * coh[0]
                       + coeffs[1] * alg[0]
                       + coeffs[2] * sep[0]
                       + coeffs[3] * wal[i][0]
                       + coeffs[4] * ns[0])
        boids[i, 5] = (coeffs[0] * coh[1]
                       + coeffs[1] * alg[1]
                       + coeffs[2] * sep[1]
                       + coeffs[3] * wal[i][1]
                       + coeffs[4] * ns[0])
    return mask[0, :]
