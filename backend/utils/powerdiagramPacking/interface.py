import numpy as np
import math
from CGAL.CGAL_Kernel import *
from CGAL.CGAL_Triangulation_2 import *
from CGAL import CGAL_Convex_hull_2
from shapely.geometry import Polygon, Point, LineString
# from utils.Voronoi import Voronoi
import cv2
import ctypes
from utils.powerdiagramPacking.Box2DCircle import Box2DSimulator
from utils.powerdiagramPacking.pdutils import *
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist
import time
import matplotlib
matplotlib.use("Agg")


def computeWeightedDelaunayByConvexHull(positions, weights=None):
    N = positions.shape[0]
    if weights is None:
        weights = np.zeros(N)
    # get the convex hull
    lift = positions.copy()
    lift = np.concatenate((lift, np.zeros((lift.shape[0], 1))), axis=1)
    for i in range(N):
        lift[i, 2] = np.sum(lift[i] ** 2) - weights[i]
    pinf = np.append(np.zeros((1, 2)), 1e2)
    lift = np.vstack((lift, pinf))
    # print(lift)
    hull = ConvexHull(lift, qhull_options='QJ')
    mesh = [simp for simp in hull.simplices if N not in simp]
    return mesh

def opencv_compactness(positions, radii):
    # get the approximate compactness by opencv
    radii = np.array(radii)
    positions = np.array(positions)
    # positions += 0.5
    # rescale to center with margin
    margin = 0.05
    max_x, max_y = np.max(positions, axis=0)+margin
    min_x, min_y = np.min(positions, axis=0)-margin
    # positions = positions / (max(max_x - min_x, max_y - min_y)+0.01)
    positions = (positions - np.array([min_x, min_y])) / (max(max_x - min_x, max_y - min_y)+0.01)
    radii = radii / (max(max_x - min_x, max_y - min_y)+0.01)

    img_size = 500
    img = 255 * np.ones((img_size, img_size, 3), np.uint8)
    N = len(positions)
    for i in range(N):
        color = (0, 0, 0)
        x = int(positions[i][0] * img_size)
        y = int(positions[i][1] * img_size)
        r = int(radii[i] * img_size) + 1
        cv2.circle(img, (x, y), r, color, -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 255 - 1, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, 2)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    envelope_area = 0
    circle_area = 1
    try:
        envelope_area = cv2.contourArea(contours[0]) / img_size / img_size
        circle_area = np.sum(np.pi * radii * radii)
    except:
        # print(positions, radii)
        # draw the image
        time_stamp = time.time()
        cv2.imwrite(f'./log/image/Contour/{time_stamp}.png', img)
    return circle_area / envelope_area



circle_hull = True
def computeConvexHull(positions):
    point_set = []
    point_n = len(positions)

    for pos in positions:
        point_set.append(Point_2(float(pos[0]), float(pos[1])))
        if point_n < 4 :
            eps = 1e-5
            point_set.append(Point_2(float(pos[0] + eps), float(pos[1] + eps)))
            point_set.append(Point_2(float(pos[0] - eps), float(pos[1] + eps)))
            point_set.append(Point_2(float(pos[0] + eps), float(pos[1] - eps)))
            point_set.append(Point_2(float(pos[0] - eps), float(pos[1] - eps)))

    convex_hull = []
    CGAL_Convex_hull_2.convex_hull_2(point_set, convex_hull)
    cvp = [(v.x(), v.y()) for v in convex_hull]
    poly_hull = Polygon(cvp)
    return poly_hull

def approxConvexHull(poly_hull):
    # approximate the convex hull by a polygon with 8 vertices
    perimeter = 0
    for i in range(len(poly_hull.exterior.coords[:-1])):
        p1 = poly_hull.exterior.coords[:-1][i]
        p2 = poly_hull.exterior.coords[:-1][(i+1)%len(poly_hull.exterior.coords[:-1])]
        perimeter += np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    edgeLen = perimeter / 8
    new_hull = []
    remaining = 0
    for i in range(len(poly_hull.exterior.coords[:-1])):
        p1 = poly_hull.exterior.coords[:-1][i]
        p2 = poly_hull.exterior.coords[:-1][(i+1)%len(poly_hull.exterior.coords[:-1])]
        dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        if dist + remaining < edgeLen:
            remaining += dist
        else:
            # find the point on the edge
            ratio = (edgeLen - remaining) / dist
            new_point = (p2[0] * ratio + p1[0] * (1 - ratio), p2[1] * ratio + p1[1] * (1 - ratio))
            new_hull.append(new_point)
            remaining = 0
    # convert the new_hull to a list of list
    new_hull = [[point[0], point[1]] for point in new_hull]
    return new_hull

def rotateByOrigin(points, angle):
    if len(points) == 1:
        return points
    rad = angle * math.pi / 180
    rotation_matrix = np.array([[math.cos(rad), -math.sin(rad)],
                                [math.sin(rad), math.cos(rad)]])
    return np.dot(rotation_matrix, points.T).T

def find_transform(cell, cell_center, points, radii):
    # print(f'n_points = {len(points)}')
    time_0 = time.perf_counter()
    if len(points) == 1:
        return 1, 0
    if len(points) == 2:
        if isinstance(cell, Polygon):
            poly_cell = cell
        else:
            poly_cell = Polygon(cell)
        shifted_points = points + cell_center - np.mean(points, axis=0)
        point0 = shifted_points[0]
        point1 = shifted_points[1]
        dis = np.linalg.norm(point0 - point1)
        # print(f'point0 = {point0}, point1 = {point1}')
        # point0 and point1 are two circle centers, they are in the cell
        # so the line connecting them must intersect with the cell boundary
        # get the intersection points
        # first extend the segment to a infinite line
        point_center = np.mean(shifted_points, axis=0)
        # print(f'point_center = {point_center}')
        extend_point0 = point0 + 10/dis * (point0 - point_center)
        extend_point1 = point1 + 10/dis * (point1 - point_center)
        # print(f'extend_point0 = {extend_point0}, extend_point1 = {extend_point1}')
        line = LineString([extend_point0, extend_point1])
        # draw_cell = np.array(poly_cell.exterior.coords[:-1])
        # # print(f'cell = {draw_cell}')
        #
        # # draw the line and the cell
        # img = np.ones((1000, 1000, 3), np.uint8) * 255
        # max_x = np.max(np.array(draw_cell)[:, 0])
        # min_x = np.min(np.array(draw_cell)[:, 0])
        # max_y = np.max(np.array(draw_cell)[:, 1])
        # min_y = np.min(np.array(draw_cell)[:, 1])
        # scale = max(max_x - min_x, max_y - min_y)
        # draw_cell = np.array(draw_cell) - np.array([min_x, min_y])
        # draw_cell = draw_cell / scale * 1000
        # draw_cell = draw_cell.astype(np.int32)
        # for i in range(len(draw_cell)):
        #     cv2.line(img, (draw_cell[i][0], draw_cell[i][1]), (draw_cell[(i+1)%len(draw_cell)][0], draw_cell[(i+1)%len(draw_cell)][1]), (0, 0, 0), 2)
        # draw_extend_point0 = np.array([extend_point0[0], extend_point0[1]]) - np.array([min_x, min_y])
        # draw_extend_point0 = draw_extend_point0 / scale * 1000
        # draw_extend_point0 = draw_extend_point0.astype(np.int32)
        # draw_extend_point1 = np.array([extend_point1[0], extend_point1[1]]) - np.array([min_x, min_y])
        # draw_extend_point1 = draw_extend_point1 / scale * 1000
        # draw_extend_point1 = draw_extend_point1.astype(np.int32)
        # cv2.line(img, (draw_extend_point0[0], draw_extend_point0[1]), (draw_extend_point1[0], draw_extend_point1[1]), (0, 0, 255), 2)
        # time_stamp = time.time()
        # cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_line.png', img)

        # print(f'line = {line}')
        poly_cell_boundary = poly_cell.boundary
        # print(f'poly_cell_boundary = {poly_cell_boundary}')
        # intersection = poly_cell_boundary.intersection(line)
        intersection = line.intersection(poly_cell_boundary)
        # print(f'intersection = {intersection}')
        if intersection.geom_type == 'MultiPoint':
            intersection_points = [intersection.geoms[0], intersection.geoms[1]]
        else:
            return 1, 0
        # line = [point0, point1]
        # line = LineString(line)
        # # get the intersection of the line and the cell
        # poly_cell_boundary = poly_cell.boundary
        # intersection = poly_cell_boundary.intersection(line)
        # if intersection.geom_type == 'MultiPoint':
        #     intersection_points = [intersection.geoms[0], intersection.geoms[1]]
        # else:
        #     return 1, 0
        if len(intersection_points) == 2:
            point_ccenter_to_point0 = point0 - cell_center + radii[0]
            point_ccenter_to_point1 = point1 - cell_center + radii[1]
            point_center_to_intersection0 = np.array([intersection_points[0].x, intersection_points[0].y]) - cell_center
            point_center_to_intersection1 = np.array([intersection_points[1].x, intersection_points[1].y]) - cell_center
            # find the same direction
            min_ratio = 1e10
            if np.dot(point_ccenter_to_point0, point_center_to_intersection0) > 0:
                min_ratio = min(min_ratio, np.linalg.norm(point_center_to_intersection0) / np.linalg.norm(point_ccenter_to_point0))
                min_ratio = min(min_ratio, np.linalg.norm(point_center_to_intersection1) / np.linalg.norm(point_ccenter_to_point1))
            else:
                min_ratio = min(min_ratio, np.linalg.norm(point_center_to_intersection0) / np.linalg.norm(point_ccenter_to_point1))
                min_ratio = min(min_ratio, np.linalg.norm(point_center_to_intersection1) / np.linalg.norm(point_ccenter_to_point0))
            # print(f'min_ratio = {min_ratio}')
            return min_ratio, 0
        else:
            return 1, 0

    # print(f'points = {points}')
    # pcs = 10
    # circle_pos = np.zeros((points.shape[0]*pcs, 2))
    # for i in range(points.shape[0]):
    #     for j in range(pcs):
    #         circle_pos[i*pcs + j] = points[i] + radii[i] * np.array([math.cos(j * 2 * math.pi / pcs), math.sin(j * 2 * math.pi / pcs)])
    circle_pos = np.array(multi_interpolate(points, radii))
    # point_hull = points[ConvexHull(points).vertices]
    point_hull = circle_pos[ConvexHull(circle_pos, qhull_options='QJ').vertices]
    points_center = np.mean(points, axis=0)
    shifted_points = point_hull + cell_center - points_center
    poly_points = Polygon(shifted_points)
    if isinstance(cell, Polygon):
        poly_cell = cell
    else:
        poly_cell = Polygon(cell)
    scaling_min, scaling_max = 1, 1  # min containt, max not containt
    s = 1
    time_1 = time.perf_counter()
    count = 0
    if poly_cell.contains(poly_points):
        while count < 5:
            count += 1
            s *= 2
            new_shifted_points = (shifted_points - cell_center) * s + cell_center
            new_poly = Polygon(new_shifted_points)
            if poly_cell.contains(new_poly):
                continue
            else:
                scaling_max = s
                scaling_min = s / 2
                break
    else:
        while count < 5:
            count += 1
            s /= 2
            new_shifted_points = (shifted_points - cell_center) * s + cell_center
            new_poly = Polygon(new_shifted_points)
            if not poly_cell.contains(new_poly):
                continue
            else:
                scaling_max = s * 2
                scaling_min = s
                break
    count = 0
    while scaling_max - scaling_min > 1e-6 and count < 5:
        count += 1
        scaling_mid = (scaling_max + scaling_min) / 2
        new_shifted_points = (shifted_points - cell_center) * scaling_mid + cell_center
        new_poly = Polygon(new_shifted_points)
        if not poly_cell.contains(new_poly):
            scaling_max = scaling_mid
        else:
            scaling_min = scaling_mid

    best_scaling = scaling_min
    best_angle = 0
    time_2 = time.perf_counter()
    return best_scaling, best_angle

def minimum_circumscribed_circle(positions):
    if len(positions) == 2:
        center = Point((positions[0][0] + positions[1][0]) / 2, (positions[0][1] + positions[1][1]) / 2)
        radius = np.sqrt((positions[0][0] - positions[1][0]) ** 2 + (positions[0][1] - positions[1][1]) ** 2) / 2
    else:
        len_pos = len(positions)
        # time_before_convex_hull = time.perf_counter()
        hull = computeConvexHull(positions)
        # time_before_circle = time.perf_counter()
        positions = []
        for pos in hull.exterior.coords[:-1]:
            # if pos is not too close to any other points, add it to the positions
            add = True
            for p in positions:
                if np.sqrt((pos[0] - p[0]) ** 2 + (pos[1] - p[1]) ** 2) < 1e-5:
                    add = False
                    break
            if add:
                positions.append(pos)
        center_x, center_y, radius = find_circle(positions)
        # check if the circle contains all points
        # time_before_check_contain = time.perf_counter()
        max_dist_to_center = -1
        for pos in positions:
            dist = np.sqrt((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2)
            if dist > max_dist_to_center:
                max_dist_to_center = dist
        if max_dist_to_center > radius:
            radius = max_dist_to_center

        # img = np.ones((1000, 1000, 3), np.uint8) * 255
        # max_x, max_y = np.max(positions, axis=0)
        # min_x, min_y = np.min(positions, axis=0)
        # draw_positions = np.array(positions) - np.array([min_x, min_y])
        # draw_positions = draw_positions / (max(max_x - min_x, max_y - min_y)) * 1000
        # draw_positions = draw_positions.astype(np.int32)
        # for pos in draw_positions:
        #     cv2.circle(img, (pos[0], pos[1]), 2, (0, 0, 0), -1)
        # draw_center = np.array([center_x, center_y]) - np.array([min_x, min_y])
        # draw_center = draw_center / (max(max_x - min_x, max_y - min_y)) * 1000
        # draw_center = draw_center.astype(np.int32)
        # draw_radius = int(radius*1.1 / (max(max_x - min_x, max_y - min_y)) * 1000)
        # cv2.circle(img, draw_center, draw_radius, (0, 0, 255), 2)
        # time_stamp = time.time()
        # cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_circles.png', img)


        # print(f'center = ({center_x}, {center_y}), radius = {radius}')
        center = Point(center_x, center_y)
        # # find the center and radius of the minimum enclosing circle
        # # first, find the maximum distance between any two points on the convex hull
        # max_dist = -1
        # max_pair = None
        # for i in range(len(hull.exterior.coords[:-1])):
        #     for j in range(i + 1, len(hull.exterior.coords[:-1])):
        #         p1 = hull.exterior.coords[:-1][i]
        #         p2 = hull.exterior.coords[:-1][j]
        #         dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        #         if dist > max_dist:
        #             max_dist = dist
        #             max_pair = (p1, p2)
        # # then find the maximum distance between any point and the line connecting the two points
        # # this is the inscribed triangle of the circle
        # # the center of the circle is the circumcenter of the triangle
        # # the radius of the circle is the circumradius of the triangle
        # max_dist = -1
        # p1, p2 = max_pair
        # for p in hull.exterior.coords[:-1]:
        #     dist = np.abs((p2[1] - p1[1]) * p[0] - (p2[0] - p1[0]) * p[1] + p2[0] * p1[1] - p2[1] * p1[0]) / np.sqrt(
        #         (p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
        #     if dist >= max_dist:
        #         max_dist = dist
        #         max_pair = (p1, p2, p)
        # p1, p2, p3 = max_pair
        # # get the circumcenter and circumradius
        # D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        # center = Point(((p1[0] ** 2 + p1[1] ** 2) * (p2[1] - p3[1]) + (p2[0] ** 2 + p2[1] ** 2) * (
        #             p3[1] - p1[1]) + (p3[0] ** 2 + p3[1] ** 2) * (p1[1] - p2[1])) / D,
        #                ((p1[0] ** 2 + p1[1] ** 2) * (p3[0] - p2[0]) + (p2[0] ** 2 + p2[1] ** 2) * (
        #                            p1[0] - p3[0]) + (p3[0] ** 2 + p3[1] ** 2) * (p2[0] - p1[0])) / D)
        # radius = np.sqrt((p1[0] - center.x) ** 2 + (p1[1] - center.y) ** 2)

    # check if the circle contains all points
    # time_before_check_contain = time.perf_counter()
    # max_dist_to_center = -1
    # for pos in positions:
    #     dist = np.sqrt((pos[0] - center.x) ** 2 + (pos[1] - center.y) ** 2)
    #     if dist > max_dist_to_center:
    #         max_dist_to_center = dist
    # if max_dist_to_center > radius:
    #     radius = max_dist_to_center
    # time_before_interpolate = time.perf_counter()
    # if len_pos == 29*5:
    #     print(f'positions = {positions}')
    #     print(f'center = ({center.x}, {center.y}), radius = {radius}')
    ccp = interpolate(center.x, center.y, radius*1.1, 180)
    # draw_ccp = np.array(ccp)
    # max_x, max_y = np.max(draw_ccp, axis=0)
    # min_x, min_y = np.min(draw_ccp, axis=0)
    # draw_ccp = draw_ccp - np.array([min_x, min_y])
    # draw_ccp = draw_ccp / (max(max_x - min_x, max_y - min_y)) * 1000
    # draw_ccp = draw_ccp.astype(np.int32)
    # img = np.ones((1000, 1000, 3), np.uint8) * 255
    # for pos in draw_ccp:
    #     cv2.circle(img, (pos[0], pos[1]), 2, (0, 0, 0), -1)
    # cv2.imwrite(f'./log/image/PackingProcess/{time.time()}_circles.png', img)

    # theta = np.linspace(0, 2 * np.pi, 300)
    # x = center.x + radius * np.cos(theta)
    # y = center.y + radius * np.sin(theta)
    # time_before_array = time.perf_counter()
    # ccp = [(x[i], y[i]) for i in range(len(x))]
    # time_after_circle = time.perf_counter()
    # print(f'time_check_contain = {(time_before_interpolate - time_before_check_contain)*1000} ms')
    # print(f'time_array = {time_after_circle - time_before_array}')
    # print(f'convex_hull = {time_before_circle - time_before_convex_hull}, circle = {time_before_interpolate - time_before_circle}, interpolate = {time_after_circle-time_before_interpolate}')
    return ccp

def computeCircleHull(positions):
    ccp = minimum_circumscribed_circle(positions)
    poly_hull = Polygon(ccp)
    # # draw polygon
    # img = np.ones((1000, 1000, 3), np.uint8) * 255
    # draw_poly = np.array(poly_hull.exterior.coords[:-1])
    # max_x, max_y = np.max(draw_poly, axis=0)
    # min_x, min_y = np.min(draw_poly, axis=0)
    # draw_poly = draw_poly - np.array([min_x, min_y])
    # draw_poly = draw_poly / (max(max_x - min_x, max_y - min_y)) * 1000
    # draw_poly = draw_poly.astype(np.int32)
    # for i in range(len(draw_poly)):
    #     cv2.line(img, tuple(draw_poly[i]), tuple(draw_poly[(i + 1) % len(draw_poly)]), (0, 0, 0), 2)
    # cv2.imwrite(f'./log/image/PackingProcess/{time.time()}_poly.png', img)
    return poly_hull


def cellCentroid(cell):
    # print(cell)
    if isinstance(cell, Polygon):
        try:
            # print('polygon')
            return np.array([cell.centroid.x, cell.centroid.y])
        except:
            return None
    # print('not polygon')
    centroid = cell_centroid(cell)
    # print('cell = ', cell)
    # print('centroid = ', centroid)
    return np.array(centroid)
    # return np.array(cell_centroid(cell))
    # x, y = 0, 0
    # area = 0
    # for k in range(len(cell)):
    #     p1 = cell[k]
    #     p2 = cell[(k + 1) % len(cell)]
    #     v = p1[0] * p2[1] - p2[0] * p1[1]
    #     area += v
    #     x += (p1[0] + p2[0]) * v
    #     y += (p1[1] + p2[1]) * v
    # area *= 3
    # if area == 0:
    #     return None
    # return np.array([x / area, y / area])

def cross_product(v1, v2):
    return v1[0] * v2[1] - v2[0] * v1[1]

def cellInscribedCircleRadius(cell, site):
    if isinstance(cell, Polygon):
        cell = np.array(cell.exterior.coords[:-1])
    if len(cell) == 0:
        return 0
    return inscribed_circle_radius(cell, site)
    # r = 10000000
    # cell = np.array(cell)
    # for k in range(len(cell)):
    #     p1 = cell[k]
    #     p2 = cell[(k + 1) % len(cell)]
    #     edgeLength = np.sqrt(np.sum((p1 - p2) ** 2))
    #     if edgeLength < 1e-12:
    #         continue
    #
    #     v = cross_product(p1 - site, p2 - site)
    #     r = min(r, abs(v / edgeLength))
    # return r

def computePowerDiagramByCGAL(positions, weights=None, hull=None, return_flower=False):
    time_start_pd = time.perf_counter()
    clip_time = 0
    if weights is None:
        nonneg_weights = np.zeros(len(positions))
    else:
        nonneg_weights = weights - np.min(weights)

    rt = Regular_triangulation_2()

    v_handles = []
    v_handles_mapping = {}
    k = 0

    for pos, w in zip(positions, nonneg_weights):
        v_handle = rt.insert(Weighted_point_2(
            Point_2(float(pos[0]), float(pos[1])), float(w)))
        v_handles.append(v_handle)
        v_handles_mapping[v_handle] = k
        k += 1

    control_point_set = [
        Weighted_point_2(Point_2(-1e2, -1e2), 0),
        Weighted_point_2(Point_2(1e2, -1e2), 0),
        Weighted_point_2(Point_2(1e2, 1e2), 0),
        Weighted_point_2(Point_2(-1e2, 1e2), 0)
    ]

    for cwp in control_point_set:
        v_handle = rt.insert(cwp)
        v_handles.append(v_handle)
        v_handles_mapping[v_handle] = k
        k += 1

    cells = []
    flowers = []

    # for i, handle in enumerate(v_handles):
    i = 0
    for handle in rt.finite_vertices():
        non_hidden_point = handle.point()
        while i < len(positions) and (non_hidden_point.x() - positions[i, 0]) ** 2 + (
                non_hidden_point.y() - positions[i, 1]) ** 2 > 1e-10:
            i += 1
            cells.append(Polygon([]))
        if i >= len(positions):
            break

        f = rt.incident_faces(handle)
        done = f.next()
        cell = []
        while True:
            face_circulator = f.next()
            wc = rt.weighted_circumcenter(face_circulator)
            cell.append((wc.x(), wc.y()))
            if face_circulator == done:
                break
        # ***************************************
        # print(cell)
        poly_cell = Polygon(cell)
        time_before_clip = time.perf_counter()
        if hull is not None and not hull.contains(poly_cell):
            # try:
            poly_cell = hull.intersection(poly_cell)
            # except:
                # np_positions = np.zeros((len(hull.exterior.coords[:-1]), 2))
                # for i in range(len(hull.exterior.coords[:-1])):
                #     np_positions[i, 0] = hull.exterior.coords[:-1][i][0]
                #     np_positions[i, 1] = hull.exterior.coords[:-1][i][1]
                # # scale to [0, 1]
                # np_positions[:, 0] = (np_positions[:, 0] - np.min(np_positions[:, 0])) / (np.max(np_positions[:, 0]) - np.min(np_positions[:, 0]))
                # np_positions[:, 1] = (np_positions[:, 1] - np.min(np_positions[:, 1])) / (np.max(np_positions[:, 1]) - np.min(np_positions[:, 1]))

                # for i in range(len(hull.exterior.coords[:-1])):
                #     p1 = np_positions[i]
                #     p2 = np_positions[(i + 1) % len(hull.exterior.coords[:-1])]
                #     cv2.line(img, (int((p1[0]) * 1000), int((p1[1]) * 1000)), (int((p2[0]) * 1000), int((p2[1]) * 1000)), (0, 0, 255), 2)
                # cv2.imwrite(f'./log/image/PowerDiagram/{time.perf_counter()}.png', img)

        cells.append(poly_cell)
        time_after_clip = time.perf_counter()
        clip_time += time_after_clip - time_before_clip
        # **************************************

        if return_flower:
            v = rt.incident_vertices(handle)
            done = v.next()
            flower = []
            while True:
                vertex_circulator = v.next()
                p = vertex_circulator.point()
                flower.append(([p.x(), p.y(), p.weight()],
                              v_handles_mapping[vertex_circulator]))
                if vertex_circulator == done:
                    break
            flowers.append(flower)

        i += 1
    time_end_pd = time.perf_counter()
    # print(f'pd time = {(time_end_pd - time_start_pd)*1000} ms, clip time = {clip_time*1000} ms')
    if return_flower:
        return cells, flowers
    else:
        return cells

def computePowerDiagramBruteForce(positions, weights=None, radii=None, clipHull=False, hull=None, dummy=None,
                                  vsol=None):
    N = positions.shape[0]
    if weights is None:
        weights = np.zeros(N)

    if dummy is not None:
        M = dummy.shape[0]
        positions = np.vstack([positions, dummy])
        weights = np.hstack([weights, np.zeros(M)])
        radii = np.hstack([radii, np.zeros(M)])

    if vsol is None:
        vsol = Voronoi.Voronoi()
    vsol.clearSites()
    vsol.inputSites(positions)
    vsol.setWeight(weights)
    if radii is not None and clipHull:
        vsol.setRadius(radii)

    if hull is not None:
        vsol.setBoundary(hull)

    if clipHull:
        res = vsol.computeBruteForce(True)
    else:
        res = vsol.computeBruteForce(False)

    boundary_return = None
    if clipHull and hull is None:
        boundary_return = {
            'hull': np.array(vsol.getConvexHullVertices()),
            'edges': np.array(vsol.getConvexHullEdges()),
            'sites': np.array(vsol.generateBoundarySites())
        }

    cells, flowers = [], []
    for (cell, flower) in res:
        cells.append(np.array(cell))
        flowers.append(np.array(flower))

    preserved_pairs = [(i, j) for i in range(N) for j in flowers[i] if i < j < N]

    if boundary_return is not None:
        return cells, flowers, preserved_pairs, boundary_return
    else:
        return cells, flowers, preserved_pairs


def debug_pos(pos, radius, name_str):
    width = 1024
    height = 1024
    margin_percentage = 0.05
    def get_image_x(origin_x):
        nonlocal width
        nonlocal margin_percentage
        origin_x = (origin_x + 0.5) * (1 - 2 * margin_percentage) + margin_percentage
        return int (origin_x * width)

    def get_image_y(origin_y):
        nonlocal height
        nonlocal margin_percentage
        origin_y = (origin_y + 0.5) * (1 - 2 * margin_percentage) + margin_percentage
        return int (origin_y * height)

    circle_size = int(radius * width) - 1

    image = 255 * np.ones((height, width, 3), np.uint8)

    # draw each point
    for i in range(pos.shape[0]):
        color = (50, 50, 50)

        x = get_image_x(pos[i, 0])
        y = get_image_y(pos[i, 1])

        cv2.circle(image, (x, y), circle_size, color, -1)

    result_path = f'./log/image/PackingProcess/{name_str}.png'
    cv2.imwrite(result_path, image)

def pd_packing_interface(init_pos, sub_clusters, init_radius):
    # print(f'sub_clusters = {sub_clusters}')
    # init_pos is a list of list
    time_pd_start = time.perf_counter()
    time_before_build_pos = time.perf_counter()
    N = 0
    id_map = {}
    inv_id_map = {}
    cluster_id = {}
    for i, sub_cluster in enumerate(sub_clusters):
        for point in sub_cluster:
            if point in id_map:
                # print(f'Warning! Same point {point} in sub-clusters!')
                continue
            id_map[point] = N
            inv_id_map[N] = point
            cluster_id[N] = i
            N += 1
    cluster_indices = [[] for _ in range(len(sub_clusters))]
    for i in range(N):
        cluster_indices[cluster_id[i]].append(i)
    # pos = []
    # rad = []
    # pos = np.array(pos)
    # rad = np.array(rad)
    pos = np.zeros((N, 2))
    rad = np.zeros(N)
    # for i in range(N):
    #     pos.append(init_pos[inv_id_map[i]])
    #     rad.append(init_radius[inv_id_map[i]])
    for i in range(N):
        pos[i] = init_pos[inv_id_map[i]]
        rad[i] = init_radius[inv_id_map[i]]

    # check if any pos is not in any cluster_indices

    init_rad = rad.copy()
    time_before_build_hull = time.perf_counter()
    # pcs = 10
    # circle_pos = np.zeros((pos.shape[0]*pcs,2))
    # for i in range(pos.shape[0]):
    #     for j in range(pcs):
    #         circle_pos[i*pcs+j] = pos[i] + rad[i]*np.array([np.cos(2*np.pi*j/pcs),np.sin(2*np.pi*j/pcs)])
    # circle_pos = np.array(multi_interpolate(pos, rad))
    # if len(pos) == 29:
    #     print(f'pos = {pos}')
    #     print(f'rad = {rad}')
    time_after_interpolate = time.perf_counter()
    # top_level_hull = computeCircleHull(circle_pos)

    # time_stamp = time.time()
    # draw_pos = pos.copy()
    # draw_pos = np.array(draw_pos)
    # draw_radius = rad.copy()
    # draw_radius = np.array(draw_radius)
    # max_x = np.max(draw_pos[:, 0] + draw_radius)
    # min_x = np.min(draw_pos[:, 0] - draw_radius)
    # max_y = np.max(draw_pos[:, 1] + draw_radius)
    # min_y = np.min(draw_pos[:, 1] - draw_radius)
    # scale = max(max_x - min_x, max_y - min_y)
    # draw_pos = (draw_pos - np.array([min_x, min_y])) / scale * 1000
    # draw_radius = draw_radius / scale * 1000
    # draw_pos = draw_pos.astype(np.int32)
    # draw_radius = draw_radius.astype(np.int32)
    # img = np.zeros((1000, 1000, 3), np.uint8)
    # for i in range(draw_pos.shape[0]):
    #     cv2.circle(img, (draw_pos[i, 0], draw_pos[i, 1]), draw_radius[i], (255, 255, 255), 2)
    # # draw the top level hull
    # draw_top_level_hull = np.array(top_level_hull.exterior.coords[:-1])
    # draw_top_level_hull = (draw_top_level_hull - np.array([min_x, min_y])) / scale * 1000
    # draw_top_level_hull = draw_top_level_hull.astype(np.int32)
    # for i in range(len(draw_top_level_hull)):
    #     cv2.line(img, tuple(draw_top_level_hull[i]), tuple(draw_top_level_hull[(i + 1) % len(draw_top_level_hull)]), (255, 255, 255), 2)
    # cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_circlepos.png', img)


    # max_x_by_cluster = []
    # min_x_by_cluster = []
    # max_y_by_cluster = []
    # min_y_by_cluster = []
    # for i in range(len(sub_clusters)):
    #     selected_indices = cluster_indices[i]
    #     # print(f'selected_indices = {selected_indices}')
    #     selected_pos = pos[selected_indices]
    #     # if any two points are too close, add a small random offset
    #     for j in range(len(selected_pos)):
    #         for k in range(j + 1, len(selected_pos)):
    #             x = selected_pos[k, 0] - selected_pos[j, 0]
    #             y = selected_pos[k, 1] - selected_pos[j, 1]
    #             d = math.sqrt(x * x + y * y)
    #             eps = 1e-5
    #             if d < eps:
    #                 # print(f'same point {j}, {k}')
    #                 # add a random small offset to posi
    #                 selected_pos[j, 0] += np.random.random() * eps
    #                 selected_pos[j, 1] += np.random.random() * eps
    #                 selected_pos[k, 0] += np.random.random() * eps
    #                 selected_pos[k, 1] += np.random.random() * eps
    #     selected_radius = rad[selected_indices]
    #     max_x = np.max(selected_pos[:, 0]+selected_radius)
    #     min_x = np.min(selected_pos[:, 0]-selected_radius)
    #     max_y = np.max(selected_pos[:, 1]+selected_radius)
    #     min_y = np.min(selected_pos[:, 1]-selected_radius)
    #     max_x_by_cluster.append(max_x)
    #     min_x_by_cluster.append(min_x)
    #     max_y_by_cluster.append(max_y)
    #     min_y_by_cluster.append(min_y)
    # max_x_by_cluster = np.array(max_x_by_cluster)
    # min_x_by_cluster = np.array(min_x_by_cluster)
    # max_y_by_cluster = np.array(max_y_by_cluster)
    # min_y_by_cluster = np.array(min_y_by_cluster)
    contain_flag = False
    # check if any cluster is fully contained in another cluster
    # for i in range(len(sub_clusters)):
    #     for j in range(len(sub_clusters)):
    #         if i == j:
    #             continue
    #         if min_x_by_cluster[i] > min_x_by_cluster[j] and max_x_by_cluster[i] < max_x_by_cluster[j] and min_y_by_cluster[i] > min_y_by_cluster[j] and max_y_by_cluster[i] < max_y_by_cluster[j]:
    #             contain_flag = True
    #             break
    time_before_remove_overlap = time.perf_counter()
    # print(f'time_interpolate = {(time_after_interpolate - time_before_build_hull)*1000} ms')
    top_level_iterations = 0
    if len(sub_clusters) == 1 or contain_flag:
        top_level_iterations = 0
    top_level_res = []
    # top_level_res.append((pos.copy(), rad.copy(), [top_level_hull]))

    # # check same position point
    # for i in range(N):
    #     for j in range(i + 1, N):
    #         x = pos[j, 0] - pos[i, 0]
    #         y = pos[j, 1] - pos[i, 1]
    #         d = math.sqrt(x * x + y * y)
    #         eps = 1e-5
    #         if d < eps:
    #             # print(f'same point {i}, {j}')
    #             # add a random small offset to posi
    #             pos[i, 0] += np.random.random() * eps
    #             pos[i, 1] += np.random.random() * eps
    #             pos[j, 0] += np.random.random() * eps
    #             pos[j, 1] += np.random.random() * eps
    # time_before_build_top_input = time.perf_counter()
    # cluster_centers = []
    # for i in range(len(sub_clusters)):
    #     cluster_center = np.zeros(2)
    #     for point in sub_clusters[i]:
    #         cluster_center += init_pos[point]
    #     cluster_center /= len(sub_clusters[i])
    #     cluster_centers.append(cluster_center)
    # cluster_centers = np.array(cluster_centers)
    # cluster_radii = np.zeros(len(sub_clusters))
    # for i in range(len(sub_clusters)):
    #     for point in sub_clusters[i]:
    #         cluster_radii[i] = max(cluster_radii[i], np.linalg.norm(init_pos[point] - cluster_centers[i])+init_radius[point])

    # cluster_weights = cluster_radii ** 2
    time_top_level_start = time.perf_counter()
    # print(f'time_build_pos = {(time_before_build_hull - time_before_build_pos)*1000} ms')
    # print(f'time_build_hull = {(time_before_remove_overlap - time_before_build_hull)*1000} ms')
    # print(f'time_remove_overlap = {(time_before_build_top_input - time_before_remove_overlap)*1000} ms')
    # print(f'time_build_top_input = {(time_top_level_start - time_before_build_top_input)*1000} ms')

    for _ in range(top_level_iterations):
        time_top_level_prepare_start = time.perf_counter()
        cluster_centers = []
        cluster_weights = []
        cluster_radii = []
        if _ == 0:
            # pcs = 10
            # circle_pos = np.zeros((pos.shape[0]*pcs, 2))
            # for i in range(pos.shape[0]):
            #     for j in range(pcs):
            #         circle_pos[i*pcs + j] = pos[i] + rad[i]*1.01 * np.array([math.cos(j * 2 * math.pi / pcs), math.sin(j * 2 * math.pi / pcs)])
            circle_pos = np.array(multi_interpolate(pos, rad))
            hull = computeCircleHull(circle_pos)
        for i in range(len(sub_clusters)):
            selected_indices = cluster_indices[i]
            selected_pos = pos[selected_indices]
            selected_radius = rad[selected_indices]
            cluster_center = np.mean(selected_pos, axis=0)
            cluster_centers.append(cluster_center)
            cluster_radii.append(max([np.linalg.norm(pos[j] - cluster_center)+rad[j] for j in selected_indices]))
            cluster_weight = np.sum(selected_radius ** 2)
            cluster_weights.append(cluster_weight)
        cluster_centers = np.array(cluster_centers)
        cluster_weights = np.array(cluster_weights)
        cluster_radii = np.array(cluster_radii)

        cc = cluster_centers.copy()

        time_top_level_pd_start = time.perf_counter()
        cells = computePowerDiagramByCGAL(cluster_centers, cluster_weights, hull=hull)
        # time_stamp = time.time()
        # draw_pos = pos.copy()
        # draw_rad = rad.copy()
        # cell_points = []
        # for cell in cells:
        #     cell_points.extend(cell.exterior.coords[:-1])
        # cell_points = np.array(cell_points)
        # max_x = np.max(cell_points[:, 0])
        # min_x = np.min(cell_points[:, 0])
        # max_y = np.max(cell_points[:, 1])
        # min_y = np.min(cell_points[:, 1])
        # scale = max(max_x - min_x, max_y - min_y)
        # draw_pos = (draw_pos - np.array([min_x, min_y])) / scale
        # draw_rad = draw_rad / scale
        # img = np.zeros((1024, 1024, 3), np.uint8)
        # for i in range(draw_pos.shape[0]):
        #     cv2.circle(img, (int(draw_pos[i, 0] * 1000), int(draw_pos[i, 1] * 1000)), int(draw_rad[i] * 1000),
        #                (255, 255, 255), 2)
        #     cv2.putText(img, str(i), (int(draw_pos[i, 0] * 1000), int(draw_pos[i, 1] * 1000)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        # for cell in cells:
        #     cell = np.array(cell.exterior.coords[:-1])
        #     cell = (cell - np.array([min_x, min_y])) / scale * 1000
        #     cell = cell.astype(np.int32)
        #     for i in range(cell.shape[0]):
        #         cv2.line(img, tuple(cell[i]), tuple(cell[(i + 1) % cell.shape[0]]), (0, 0, 255), 2)
        # cv2.putText(img, str(len(draw_pos)), (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        # cv2.imwrite(f'./log/image/PackingProcess/{_}_top_{time_stamp}.png', img)

        time_top_level_cell_start = time.perf_counter()
        for i, cell in enumerate(cells):
            center = cellCentroid(cell)
            radius = cellInscribedCircleRadius(cell, center)
            cluster_centers[i] = center
            cluster_radii[i] = radius
        cluster_scale_ratio = 1e10
        # if there is nan in cluster_center, stop top level iteration
        if np.isnan(np.sum(cluster_centers)):
            break
        time_top_level_transform_start = time.perf_counter()
        # print(f'top_level_pd time = {(time_top_level_cell_start - time_top_level_pd_start)*1000} ms')
        # print(f'top_level_cell time = {(time_top_level_transform_start - time_top_level_cell_start)*1000} ms')
        for i in range(len(cluster_centers)):
            time_0 = time.perf_counter()
            selected_indices = cluster_indices[i]
            max_scaling_ratio, max_scaling_rotation_angle = find_transform(cells[i], cluster_centers[i],
                                                                           pos[selected_indices], rad[selected_indices])
            time_1 = time.perf_counter()
            pos[selected_indices] = (pos[selected_indices] - cc[i]) * max_scaling_ratio + cluster_centers[i]
            # rad[selected_indices] *= max_scaling_ratio
            # cluster_scale_ratio = min(max_scaling_ratio, cluster_scale_ratio)
            time_2 = time.perf_counter()
            # print(f'transform time = {(time_1 - time_0)*1000} ms, rotate time = {(time_2 - time_1)*1000} ms')
        top_level_res.append((pos.copy(), rad.copy(), cells))
        time_stamp = time.time()
        draw_pos = pos.copy()
        draw_rad = rad.copy()
        cell_points = []
        for cell in cells:
            cell_points.extend(cell.exterior.coords[:-1])
        cell_points = np.array(cell_points)
        max_x = np.max(cell_points[:, 0])
        min_x = np.min(cell_points[:, 0])
        max_y = np.max(cell_points[:, 1])
        min_y = np.min(cell_points[:, 1])
        scale = max(max_x - min_x, max_y - min_y)
        draw_pos = (draw_pos - np.array([min_x, min_y])) / scale
        draw_rad = draw_rad / scale
        img = np.zeros((1024, 1024, 3), np.uint8)
        for i in range(draw_pos.shape[0]):
            cv2.circle(img, (int(draw_pos[i, 0] * 1000), int(draw_pos[i, 1] * 1000)), int(draw_rad[i] * 1000),
                       (255, 255, 255), 2)
            cv2.putText(img, str(i), (int(draw_pos[i, 0] * 1000), int(draw_pos[i, 1] * 1000)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        for cell in cells:
            cell = np.array(cell.exterior.coords[:-1])
            cell = (cell - np.array([min_x, min_y])) / scale * 1000
            cell = cell.astype(np.int32)
            for i in range(cell.shape[0]):
                cv2.line(img, tuple(cell[i]), tuple(cell[(i + 1) % cell.shape[0]]), (0, 0, 255), 2)
        cv2.imwrite(f'./log/image/PackingProcess/{_}_top_{time_stamp}.png', img)

        time_top_level_transform_end = time.perf_counter()
        # print(f'top_level_prepare time = {(time_top_level_pd_start - time_top_level_prepare_start)*1000} ms')
        # print(f'top_level_pd+cells time = {(time_top_level_transform_start - time_top_level_pd_start)*1000} ms')
        # print(f'top_level_transform time = {(time_top_level_transform_end - time_top_level_transform_start)*1000} ms')

    # pos, rad, cells = top_level_res[-1]
    # scale = max(np.max(pos[:, 0]) - np.min(pos[:, 0]), np.max(pos[:, 1]) - np.min(pos[:, 1]))
    # mmin_ratio = 1e10
    # # if any two points are too close, add a small random offset
    # for i in range(len(pos)):
    #     for j in range(i + 1, len(pos)):
    #         x = pos[j, 0] - pos[i, 0]
    #         y = pos[j, 1] - pos[i, 1]
    #         d = math.sqrt(x * x + y * y)
    #         eps = 1e-5*scale
    #         if d < eps:
    #             # print(f'same point {i}, {j}')
    #             # add a random small offset to posi
    #             pos[i, 0] += np.random.random() * eps
    #             pos[i, 1] += np.random.random() * eps
    #             pos[j, 0] += np.random.random() * eps
    #             pos[j, 1] += np.random.random() * eps
    #         else:
    #             mmin_ratio = min(mmin_ratio, d / (rad[i] + rad[j]))
    # # print(f'min_ratio = {min_ratio}')
    # # # rad *= min_ratio
    # pos /= mmin_ratio



    def tesellate(indices, p, r):
        time_tesellate_start = time.perf_counter()
        iterations = min(5, (len(indices) // 3)+1)
        # iterations = 0
        pos = p[indices].copy()
        init_radius = r[indices].copy()
        # print(f'pos = {pos}')
        # init_radius = [r[0] for _ in range(len(indices))]
        # init_radius = np.array(init_radius)

        # check init_radius to avoid overlap
        init_scale = 1
        for i in range(len(init_radius)):
            for j in range(i + 1, len(init_radius)):
                x = pos[j, 0] - pos[i, 0]
                y = pos[j, 1] - pos[i, 1]
                d = math.sqrt(x * x + y * y)
                if d < init_radius[i] + init_radius[j]:
                    init_scale = min(init_scale, d / (init_radius[i] + init_radius[j]))
        if init_scale == 0: init_scale = 0.001
        radius = init_radius * init_scale
        # # print(f'init_scale = {init_scale}')
        # # print(f'init_radius = {init_radius}')
        # # print(f'radius = {radius}')
        # # pcs = 10
        # # circle_pos = np.zeros((pos.shape[0] * pcs, 2))
        # # for i in range(pos.shape[0]):
        # #     for j in range(pcs):
        # #         circle_pos[i * pcs + j] = pos[i] + radius[i] * np.array([np.cos(2 * np.pi * j / pcs),
        # #                                                                  np.sin(2 * np.pi * j / pcs)])
        # time_before_build_hull = time.perf_counter()
        circle_pos = np.array(multi_interpolate(pos, rad))
        #
        # # img = 255 * np.ones((1024, 1024, 3), np.uint8)
        # # max_x = np.max(circle_pos[:, 0])
        # # min_x = np.min(circle_pos[:, 0])
        # # max_y = np.max(circle_pos[:, 1])
        # # min_y = np.min(circle_pos[:, 1])
        # # scale = max(max_x - min_x, max_y - min_y)
        # # draw_pos = (circle_pos - np.array([min_x, min_y])) / scale
        # # for i in range(draw_pos.shape[0]):
        # #     cv2.circle(img, (int(draw_pos[i, 0] * 1000), int(draw_pos[i, 1] * 1000)), 2, (0, 0, 0), -1)
        # # # if circle_hull:
        hull = computeCircleHull(circle_pos)
        # # # else:
        # # #     hull = computeConvexHull(pos)
        # # rescaled_hull = np.array(hull.exterior.coords[:-1])
        # # rescaled_hull = (rescaled_hull - np.array([min_x, min_y])) / scale * 1000
        # # rescaled_hull = rescaled_hull.astype(np.int32)
        # # for i in range(rescaled_hull.shape[0]):
        # #     cv2.line(img, tuple(rescaled_hull[i]), tuple(rescaled_hull[(i + 1) % rescaled_hull.shape[0]]), (0, 0, 255), 2)
        # # if len(indices) > 90:
        # #     cv2.imwrite(f'./log/image/PackingProcess/{time.perf_counter()}_circles.png', img)
        #
        # time_iter_start = time.perf_counter()
        # # print(f'hull time = {(time_iter_start - time_before_build_hull)*1000} ms')
        for iter in range(iterations):
            time_before_pd = time.perf_counter()
            weights = radius ** 2
            cells = computePowerDiagramByCGAL(pos, weights,  hull=hull)
            time_after_pd = time.perf_counter()
            min_ratio = 1e9
            centroid_pos = pos.copy()
            for i, cell in enumerate(cells):
                cell = np.array(cell.exterior.coords[:-1])

                centroid = cellCentroid(cell)
                centroid_pos[i] = centroid

                max_rad = cellInscribedCircleRadius(cell, centroid)
                min_ratio = min(min_ratio, max_rad / radius[i])

            # # draw debug
            # img = 255 * np.ones((2048, 2048, 3), np.uint8)
            # max_x = np.max(pos[:, 0])
            # min_x = np.min(pos[:, 0])
            # max_y = np.max(pos[:, 1])
            # min_y = np.min(pos[:, 1])
            # scale = max(max_x - min_x, max_y - min_y)
            # draw_pos = (pos - np.array([min_x, min_y])) / scale
            # for i in range(draw_pos.shape[0]):
            #     cv2.circle(img, (int(draw_pos[i, 0] * 2048), int(draw_pos[i, 1] * 2048)), 5, (0, 0, 0), -1)
            # cv2.imwrite(f'./log/image/PackingProcess/{time.perf_counter()}_cells.png', img)

            # there is no nan in centroid_pos
            if np.isnan(np.sum(centroid_pos)):
                break
            else:
                pos = centroid_pos
            radius = radius * min_ratio
            time_after_iter = time.perf_counter()
            # print(f'pd time = {(time_after_pd - time_before_pd)*1000} ms')
            # print(f'iter time = {(time_after_iter - time_before_pd)*1000} ms')
        # time_iter_end = time.perf_counter()
        # # check overlap
        last_ratio = 1e9
        radius = init_radius
        for i in range(len(radius)):
            for j in range(i + 1, len(radius)):
                x = pos[j, 0] - pos[i, 0]
                y = pos[j, 1] - pos[i, 1]
                d = math.sqrt(x * x + y * y)
                if d > radius[i] + radius[j]:
                    last_ratio = min(last_ratio, d / (radius[i] + radius[j]))
        radius = radius * last_ratio
        scale = radius[0] / init_radius[0]
        # hull = None
        time_tesellate_end = time.perf_counter()
        # print(f'tesellate time = {(time_tesellate_end - time_tesellate_start)*1000} ms')
        # print(f'pre iter time = {(time_iter_start - time_tesellate_start)*1000} ms')
        # print(f'iter time = {(time_iter_end - time_iter_start)*1000} ms')
        # print(f'post iter time = {(time_tesellate_end - time_iter_end)*1000} ms')
        return pos, hull, scale

    # if np.min(rad) < scale/mmin_ratio * 1e-2:
    if True:
        # print('too small radius')
        scales = []
        partial_results = []

        # if np.unique(cluster).shape[0] > 1:
        selected_indices = list(range(len(pos)))
        time_low_level_start = time.perf_counter()
        partial_res = tesellate(selected_indices, pos, rad)
        time_tesselate_end = time.perf_counter()
        # print(f'tesselate time = {(time_tesselate_end - time_low_level_start)*1000} ms')
        partial_results.append((selected_indices, partial_res))

        scale = partial_res[2]
        scales.append(scale)

        # min_scale = np.min(scale)
        for scale, (indices, res) in zip(scales, partial_results):
            # print(f'scale = {scale}')
            # print(f'indices = {indices}')
            # print(f'res = {res}')
            # scaling_factor = min_scale / scale
            # scale = 1
            cluster_pos, cluster_hull, _ = res
            center = cellCentroid(cluster_hull)
            cluster_pos = (cluster_pos - center) / scale + center
            pos[indices] = cluster_pos
            # print(f'pos = {pos}')
            # print(f'indices = {indices}')


    # final_pos = [[pos[i, 0], pos[i, 1]] for i in range(N)]
    # last_ratio = 1e9
    # radius = init_radius
    # for i in range(len(radius)):
    #     for j in range(i + 1, len(radius)):
    #         x = pos[j, 0] - pos[i, 0]
    #         y = pos[j, 1] - pos[i, 1]
    #         d = math.sqrt(x * x + y * y)
    #         if d > radius[i] + radius[j]:
    #             last_ratio = min(last_ratio, d / (radius[i] + radius[j]))
    # radius = radius * last_ratio

    final_pos = init_pos
    for i in range(N):
        final_pos[inv_id_map[i]] = pos[i]
    time_overall_end = time.perf_counter()
    # print(f'overall time = {(time_overall_end - time_pd_start)*1000} ms')
    # print(f'prepare time = {(time_top_level_start - time_pd_start)*1000} ms')
    # print(f'top level time = {(time_low_level_start - time_top_level_start)*1000} ms')
    # print(f'low level time = {(time_overall_end - time_low_level_start)*1000} ms')

    return final_pos.tolist()

def post_process(positions, radii, attraction_pairs):

    if len(positions) == 1:
        edges = []
    elif len(positions) == 2:
        edges = [[0, 1]]
    else:
        # edges = attraction_pairs
        # build delaunay triangulation on positions
        # positions = np.array(positions)
        # tri = Delaunay(positions)
        edges = []
        # # get all edges in delaunay triangulation
        # for face in tri.simplices:
        #     for i in range(3):
        #         for j in range(i + 1, 3):
        #             if (face[i], face[j]) not in edges and (face[j], face[i]) not in edges:
        #                 edges.append((face[i], face[j]))
    for pair in attraction_pairs:
        if pair not in edges:
            edges.append(pair)
    return edges
    # box2d = ctypes.cdll.LoadLibrary('./utils/powerdiagramPacking/box2d.dll')
    # res = np.zeros((len(positions), 2), dtype=np.float32)
    # size_mag = 100
    # gravity_mag = 200*size_mag
    # attraction_mag = 50*size_mag
    # c_positions = positions.copy() * size_mag
    # # radii = radii * size_mag
    # c_radii = radii.copy() * size_mag
    # c_positions = c_positions.flatten()
    # c_positions = c_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # c_radii = c_radii.flatten()
    # c_radii = c_radii.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # c_size_mag = ctypes.c_double(size_mag)
    # c_gravity = ctypes.c_double(gravity_mag)
    # c_attraction = ctypes.c_double(attraction_mag)
    # c_alpha = ctypes.c_double(1)
    # c_n_iters = ctypes.c_int(200)
    # c_attractions = np.array(edges, dtype=np.int32)
    # c_attractions = c_attractions.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # c_res = res.flatten()
    # c_res = c_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # c_alpha_min = ctypes.c_double(0.1)
    # box2d_simulator = Box2DSimulator(positions, radii, edges, iterations=200)
    # box2d_simulator.run()
    # res = box2d_simulator.get_positions()
    # return res
    # print(f'radii = {radii}')
    # box2d.Simulate(len(positions), c_positions, c_radii, len(edges), c_attractions, c_size_mag, c_gravity, c_attraction, c_n_iters, ctypes.c_double(0.9), c_res, c_alpha_min)
    # #     extern void Simulate(int n, double* positions, double* radii,
    # #     int n_attraction_pairs, int* attraction_pairs, double size_mag, double gravity_mag,
    # #     double attraction_mag, int n_iters, double decay, double* out_positions, double alpha_min){
    # res = np.ctypeslib.as_array(c_res, (len(positions), 2))
    # res = res / size_mag
    # print(f'res = {res}')
    # print(f'radii = {radii}')
    # return res


def get_optimal_init_pos(sub_clusters):
    # alpha = 1
    # start_centroid = np.array([0, 0], dtype=np.float32)
    end_centroid = np.array([0, 0], dtype=np.float32)
    n = len(sub_clusters[0]['pre_packing_pos'])
    # radii = sub_clusters[0]['pre_radius']
    # radii = np.array(radii)
    N = 0
    ct_n = 0
    for sub_cluster in sub_clusters:
        n = len(sub_cluster['pre_packing_pos'])
        N += n
        # start_centroid += np.array(sub_cluster['pre_avg_start_pos']) * len(sub_cluster['pre_packing_pos'])
        end_centroid += np.array(sub_cluster['pre_avg_end_pos']) * len(sub_cluster['pre_sub_clusters_id'])
        ct_n += len(sub_cluster['pre_sub_clusters_id'])
    # start_centroid /= ct_n
    end_centroid /= ct_n
    cluster_indices = []
    init_pos = np.zeros((n, 2))
    for i, sub_cluster in enumerate(sub_clusters):
        n = len(sub_cluster['pre_sub_clusters_id'])
        cluster = []
        for j in range(n):
            # start_gap = sub_cluster['pre_avg_start_pos'] - start_centroid
            cluster.append(sub_cluster['pre_sub_clusters_id'][j])
            end_gap = sub_cluster['pre_avg_end_pos'] - end_centroid
            init_pos[sub_cluster['pre_sub_clusters_id'][j]] = sub_cluster['pre_packing_pos'][sub_cluster['pre_sub_clusters_id'][j]] + end_gap * 5
        cluster_indices.append(cluster)
    return init_pos


def new_pd_packing_interface(n_pre, pre_packing_pos, pre_avg_end_pos, pre_avg_start_pos, pre_radius, pre_sub_clusters_id, cur_sub_clusters_id):
    start_time = time.perf_counter()
    time_stamp = time.time()
    # print(f'n_pre = {n_pre}')
    # print(f'pre_sub_clusters_id = {pre_sub_clusters_id}')
    # print(f'cur_sub_clusters_id = {cur_sub_clusters_id}')
    # print(f'pre_avg_end_pos = {pre_avg_end_pos}')

    # print(f'cur time : {time_stamp}')

    n_pre = int(n_pre)
    pre_sub_clusters = []
    pre_radii = np.array(pre_radius[0])
    for i in range(n_pre):
        sub_cluster = {}
        sub_cluster['pre_radius'] = []
        sub_cluster['pre_sub_clusters_id'] = []
        sub_cluster['pre_avg_end_pos'] = pre_avg_end_pos[i]
        sub_cluster['pre_avg_start_pos'] = pre_avg_start_pos[i]
        sub_cluster['pre_packing_pos'] = pre_packing_pos[i]
        for j in range(len(pre_sub_clusters_id[i])):
            for k in range(len(pre_sub_clusters_id[i][j])):
                sub_cluster['pre_sub_clusters_id'].append(pre_sub_clusters_id[i][j][k])
                sub_cluster['pre_radius'].append(pre_radii[pre_sub_clusters_id[i][j][k]])
        pre_sub_clusters.append(sub_cluster)
        # print(f'sub_cluster = {sub_cluster}')
    # print(f'pre_sub_clusters = {pre_sub_clusters}')
    id_sub_clusters = []
    for i in range(n_pre):
        id_sub_clusters.append(pre_sub_clusters[i]['pre_sub_clusters_id'])
    # print(f'id_sub_clusters = {id_sub_clusters}')
    clusters = []
    non_zero_idx = []
    # for id_sub_cluster in id_sub_clusters:
    #     cluster = []
    #     for idx in id_sub_cluster:
    #         if idx not in non_zero_idx:
    #             non_zero_idx.append(idx)
    #             cluster.append(len(non_zero_idx) - 1)
    #     clusters.append(cluster)
    # print(f'cur_sub_clusters_id = {cur_sub_clusters_id}')
    # for id_sub_cluster in cur_sub_clusters_id:
    #     cluster = []
    #     for idx in id_sub_cluster:
    #         if idx not in non_zero_idx:
    #             non_zero_idx.append(idx)
    #             cluster.append(len(non_zero_idx) - 1)
    #     clusters.append(cluster)
    clusters = []
    cluster = []
    for idx in cur_sub_clusters_id:
        if idx not in non_zero_idx:
            non_zero_idx.append(idx)
            cluster.append(len(non_zero_idx) - 1)
    clusters.append(cluster)

        # print(f'cluster = {cluster}')
    # print(f'clusters = {clusters}')
    # print(f'non_zero_idx = {non_zero_idx}')

    # if len(non_zero_idx) == 86:
    # print(f'n_pre = {n_pre}')
    # print(f'pre_sub_clusters_id = {pre_sub_clusters_id}')
    init_pos = get_optimal_init_pos(pre_sub_clusters)
    # if there exist same points, add a small random offset to avoid them
    # pre_radius = pre_radius[0]
    # pre_radius = np.array(pre_radius)
    filter_init_pos = init_pos[non_zero_idx]
    filter_init_radius = pre_radii[non_zero_idx]
    # draw_init_pos = filter_init_pos.copy()
    # draw_init_radius = filter_init_radius.copy()
    # # print(f'filter_init_pos = {filter_init_pos}')
    # # print(f'filter_init_radius = {filter_init_radius}')
    # scale = max(np.max(draw_init_pos[:, 0]) - np.min(draw_init_pos[:, 0]), np.max(draw_init_pos[:, 1]) - np.min(draw_init_pos[:, 1]))
    # if scale == 0:
    #     scale = np.max(draw_init_radius*2)
    # draw_init_pos = (draw_init_pos - np.min(draw_init_pos, axis=0)) / scale
    # draw_init_radius = draw_init_radius / scale
    # img = np.zeros((1024, 1024, 3), np.uint8)
    # for i in range(draw_init_pos.shape[0]):
    #     cv2.circle(img, (int(draw_init_pos[i, 0] * 1000), int(draw_init_pos[i, 1] * 1000)), int(draw_init_radius[i] * 1000),
    #                (255, 255, 255), 2)
    #     cv2.putText(img, str(i), (int(draw_init_pos[i, 0] * 1000), int(draw_init_pos[i, 1] * 1000)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    # cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_pre.png', img)
    overlap_flag = False
    for i in range(len(filter_init_pos)):
        same_pos_list = [i]
        for j in range(i + 1, len(filter_init_pos)):
            x = filter_init_pos[j, 0] - filter_init_pos[i, 0]
            y = filter_init_pos[j, 1] - filter_init_pos[i, 1]
            d = math.sqrt(x * x + y * y)
            eps = 1e-4
            if d < eps:
                # print(f'same point {i}, {j}')
                # add a random small offset to posi
                # filter_init_pos[i, 0] += np.random.random() * move_eps
                # filter_init_pos[i, 1] += np.random.random() * move_eps
                # filter_init_pos[j, 0] += np.random.random() * move_eps
                # filter_init_pos[j, 1] += np.random.random() * move_eps
                same_pos_list.append(j)

            if d < (filter_init_radius[i] + filter_init_radius[j]) * 0.95:
                overlap_flag = True

        if len(same_pos_list) >= 2:
            num = len(same_pos_list)
            offset = np.zeros((num, 2))
            max_radius = np.max(filter_init_radius[same_pos_list])
            x_offset = max_radius * 2
            y_offset = max_radius * math.sqrt(3)
            num_sqt = max(int(math.sqrt(num)), 2)
            for offset_id, idx in enumerate(same_pos_list):
                x_i = offset_id % num_sqt
                y_i = offset_id // num_sqt
                offset[offset_id, 0] = x_i * x_offset + (y_i % 2) * x_offset / 2
                offset[offset_id, 1] = y_i * y_offset

            # print(f'offset = {offset}')

            offset_center = np.mean(offset, axis=0)

            for offset_id, idx in enumerate(same_pos_list):
                filter_init_pos[idx] += offset[offset_id] - offset_center

    for i, idx in enumerate(non_zero_idx):
        init_pos[idx] = filter_init_pos[i]

    # get the convexhull of the initial positions
    # hull = computeConvexHull(filter_init_pos)
    # pcs = 10
    # circle_pos = np.zeros((filter_init_pos.shape[0] * pcs, 2))
    # for i in range(filter_init_pos.shape[0]):
    #     for j in range(pcs):
    #         circle_pos[i * pcs + j] = filter_init_pos[i] + filter_init_radius[i] * np.array([np.cos(2 * np.pi * j / pcs),
    #                                                                                          np.sin(2 * np.pi * j / pcs)])

    # circle_pos = np.array(multi_interpolate(filter_init_pos, filter_init_radius))
    # hull = computeCircleHull(circle_pos)
    # area_hull = hull.area
    time_convex_hull = time.perf_counter()
    # compactness = opencv_compactness(filter_init_pos, filter_init_radius*1.05)
    time_compactness = time.perf_counter()
    # print(f'time_convex_hull = {(time_convex_hull - start_time)*1000} ms')
    # print(f'time_compactness = {(time_compactness - time_convex_hull)*1000} ms')
    # area_circles = 0
    # print(f'filter_init_pos = {filter_init_pos}')
    # print(f'filter_init_radius = {filter_init_radius}')
    # for i in range(len(filter_init_pos)):
    #     area_circles += math.pi * filter_init_radius[i] ** 2
    # print(f'compactness = {compactness}, area_circles/area_hull = {area_circles/area_hull}, overlap_flag = {overlap_flag}')
    # if 0.90 < compactness < 1.1 and 0.4 < area_circles / area_hull < 1 and not overlap_flag:
    #     # print('early stop')
    #     return init_pos.tolist(), [], []
    attraction_pairs = []

    for cluster in clusters:
        pos = filter_init_pos[cluster]
        radius = filter_init_radius[cluster]
        if len(cluster) == 1:
            continue
        if len(cluster) == 2:
            if np.linalg.norm(pos[0] - pos[1]) <= 1.1 * (radius[0] + radius[1]):
                attraction_pairs.append((cluster[0], cluster[1]))
            continue
        # use weighted delaunay triangulation to get attraction pairs
        weights = radius ** 2
        mesh = computeWeightedDelaunayByConvexHull(pos, weights)
        for face in mesh:
            for i in range(3):
                for j in range(i + 1, 3):
                    if (cluster[face[i]], cluster[face[j]]) not in attraction_pairs and (cluster[face[j]], cluster[face[i]]) not in attraction_pairs:
                        attraction_pairs.append((cluster[face[i]], cluster[face[j]]))
        # for i in range(len(cluster)):
        #     for j in range(i + 1, len(cluster)):
        #         pos1 = filter_init_pos[cluster[i]]
        #         pos2 = filter_init_pos[cluster[j]]
        #         radius1 = filter_init_radius[cluster[i]]
        #         radius2 = filter_init_radius[cluster[j]]
        #         if np.linalg.norm(pos1 - pos2) <= 1.1 * (radius1 + radius2):
        #             # TODO i,j in non-zero-idx
        #             attraction_pairs.append((cluster[i], cluster[j]))
    time_prepare = time.perf_counter()

    # draw_init_pos = filter_init_pos.copy()
    # # print(f'draw_init_pos = {draw_init_pos}')
    # scale = max(np.max(draw_init_pos[:, 0]) - np.min(draw_init_pos[:, 0]),
    #             np.max(draw_init_pos[:, 1]) - np.min(draw_init_pos[:, 1]))
    # if scale < 1e-5:
    #     scale = 1
    #
    # draw_init_pos[:, 0] = (draw_init_pos[:, 0] - np.min(draw_init_pos[:, 0])) / scale
    # draw_init_pos[:, 1] = (draw_init_pos[:, 1] - np.min(draw_init_pos[:, 1])) / scale
    # scaled_radius = np.array(filter_init_radius) / scale
    #
    # # print(f'scaled_radius = {scaled_radius}')
    # # draw
    # img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # for i in range(len(draw_init_pos)):
    #     cv2.circle(img, (int(draw_init_pos[i, 0] * 1000), int(draw_init_pos[i, 1] * 1000)),
    #                int(scaled_radius[i] * 1000), (255, 255, 255), 2)
    #     cv2.putText(img, f'{i}', (int(draw_init_pos[i, 0] * 1000), int(draw_init_pos[i, 1] * 1000)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)
    # cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_init.png', img)

    # final_pos = pd_packing_interface(init_pos, id_sub_clusters, pre_radii)
    # time_power_diagram = time.perf_counter()

    # final_pos = np.array(final_pos)
    final_pos = np.array(init_pos)

    # draw_final_pos = final_pos[non_zero_idx]
    # # print(f'draw_final_pos.shape = {draw_final_pos.shape}')

    # # scale to [0, 1]
    # scale = max(np.max(draw_final_pos[:, 0]) - np.min(draw_final_pos[:, 0]), np.max(draw_final_pos[:, 1]) - np.min(draw_final_pos[:, 1]))
    # if scale < 1e-5:
    #     scale = 1

    # draw_final_pos[:, 0] = (draw_final_pos[:, 0] - np.min(draw_final_pos[:, 0])) / scale
    # draw_final_pos[:, 1] = (draw_final_pos[:, 1] - np.min(draw_final_pos[:, 1])) / scale
    # scaled_radius = np.array(pre_radii) / scale
    # # draw
    # img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # for i in range(len(draw_final_pos)):
    #     cv2.circle(img, (int(draw_final_pos[i, 0] * 1000), int(draw_final_pos[i, 1] * 1000)), int(scaled_radius[i] * 1000), (255, 255, 255), 2)
    #     cv2.putText(img, f'{i}', (int(draw_final_pos[i, 0] * 1000), int(draw_final_pos[i, 1] * 1000)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)
    # cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_start.png', img)

    final_pos = np.array(final_pos)
    non_zero_idx = np.array(non_zero_idx)
    # print(f'final_pos_pre = {final_pos[non_zero_idx]}')
    non_zero_final_pos = final_pos[non_zero_idx]
    # print(f'non_zero_final_pos = {non_zero_final_pos}')
    non_zero_pre_radius = pre_radii[non_zero_idx]
    # non_zero_final_pos = post_process(non_zero_final_pos, non_zero_pre_radius, attraction_pairs)
    # edges = []
    edges = post_process(non_zero_final_pos, non_zero_pre_radius, attraction_pairs)
    centroid = np.mean(non_zero_final_pos, axis=0)
    non_zero_final_pos = non_zero_final_pos - centroid
    final_pos[non_zero_idx] = non_zero_final_pos

    # # time_post_process = time.perf_counter()
    # print(f'final_pos = {final_pos[non_zero_idx]}')
    # draw_final_pos = final_pos[non_zero_idx]

    # # scale to [0, 1]
    # scale = max(np.max(draw_final_pos[:, 0]) - np.min(draw_final_pos[:, 0]), np.max(draw_final_pos[:, 1]) - np.min(draw_final_pos[:, 1]))
    # if scale < 1e-5:
    #     scale = 1
    #
    # draw_final_pos[:, 0] = (draw_final_pos[:, 0] - np.min(draw_final_pos[:, 0])) / scale
    # draw_final_pos[:, 1] = (draw_final_pos[:, 1] - np.min(draw_final_pos[:, 1])) / scale
    # scaled_radius = np.array(filter_init_radius) / scale
    # # draw
    # img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # for i in range(len(draw_final_pos)):
    #     cv2.circle(img, (int(draw_final_pos[i, 0] * 1000), int(draw_final_pos[i, 1] * 1000)),
    #                int(scaled_radius[i] * 1000), (255, 255, 255), 2)
    #     cv2.putText(img, f'{i}', (int(draw_final_pos[i, 0] * 1000), int(draw_final_pos[i, 1] * 1000)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    # for edge in edges:
    #     p1 = edge[0]
    #     p2 = edge[1]
    #     cv2.line(img, (int(draw_final_pos[p1, 0] * 1000), int(draw_final_pos[p1, 1] * 1000)),
    #              (int(draw_final_pos[p2, 0] * 1000), int(draw_final_pos[p2, 1] * 1000)), (255, 255, 255), 2)
    # cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_final.png', img)

    # print(f'time_prepare = {time_prepare - start_time}')
    # print(f'time_power_diagram = {time_power_diagram - time_prepare}')
    # print('finish packing')
    # print(f'return final_pos = {final_pos.tolist()}')
    # print(f'final_pos = {final_pos.tolist()}')
    # print(f'non_zero_idx = {non_zero_idx}')
    # print(f'edges = {edges}')

    return final_pos.tolist(), non_zero_idx, edges


def sub_cluster_division(sub_clusters, sub_cluster_packing_pos, cur_sub_cluster, cur_packing_pos, radii, directions):
    # print(f'sub_clusters = {sub_clusters}')
    n_pre = len(sub_clusters)
    sub_cluster_packing_pos = np.array(sub_cluster_packing_pos)
    cur_packing_pos = np.array(cur_packing_pos)
    radii = np.array(radii)

    # get existing neighbors in sub_clusters
    sub_cluster_neighbors = []
    for i in range(n_pre):
        neighbors = []
        for j in range(len(sub_clusters[i])):
            for k in range(j+1, len(sub_clusters[i])):
                p1 = sub_clusters[i][j]
                p2 = sub_clusters[i][k]
                dist = np.linalg.norm(sub_cluster_packing_pos[i][p1] - sub_cluster_packing_pos[i][p2])
                if dist < (radii[p1] + radii[p2])*1.05:
                    neighbors.append((p1, p2))
        sub_cluster_neighbors.append(neighbors)
    cur_neighbors = []
    for i in range(len(cur_sub_cluster)):
        for j in range(i+1, len(cur_sub_cluster)):
            p1 = cur_sub_cluster[i]
            p2 = cur_sub_cluster[j]
            dist = np.linalg.norm(cur_packing_pos[p1] - cur_packing_pos[p2])
            if dist < (radii[p1] + radii[p2])*1.05:
                cur_neighbors.append((p1, p2))
    cur_adj_list = {}
    for i in range(len(cur_packing_pos)):
        cur_adj_list[i] = []
    for p1, p2 in cur_neighbors:
        cur_adj_list[p1].append(p2)
        cur_adj_list[p2].append(p1)


    # use matplotlib's cmap
    cmap = plt.get_cmap('tab10')
    # draw for debug
    time_stamp = time.perf_counter()
    for i in range(n_pre):
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        draw_pos = (sub_cluster_packing_pos[i])*10+0.5
        draw_radius = radii*10
        color = cmap(i)
        for point in sub_clusters[i]:
            cv2.circle(img, (int(draw_pos[point, 0] * 1000), int(draw_pos[point, 1] * 1000)),
                       int(draw_radius[point] * 1000), (255, 255, 255), 2)
            cv2.putText(img, f'{point}', (int(draw_pos[point, 0] * 1000), int(draw_pos[point, 1] * 1000)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        for neighbors in sub_cluster_neighbors[i]:
            p1, p2 = neighbors
            if (p1, p2) in cur_neighbors or (p2, p1) in cur_neighbors:
                cv2.line(img, (int(draw_pos[p1, 0] * 1000), int(draw_pos[p1, 1] * 1000)),
                            (int(draw_pos[p2, 0] * 1000), int(draw_pos[p2, 1] * 1000)),
                            (color[2]*255, color[1]*255, color[0]*255), 2)
        cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_sub_cluster_{i}_pre.png', img)
    sub_cluster_adj_lists = []
    for i in range(n_pre):
        adj_lists = {}
        for j in sub_clusters[i]:
            adj_lists[j] = []
        sub_cluster_adj_lists.append(adj_lists)
    for i in range(n_pre):
        neighbors = sub_cluster_neighbors[i]
        adj_lists = sub_cluster_adj_lists[i]
        for p1, p2 in neighbors:
            if (p1, p2) in cur_neighbors or (p2, p1) in cur_neighbors:
                adj_lists[p1].append(p2)
                adj_lists[p2].append(p1)
    # get connected components
    sub_cluster_connected_components = []
    for i in range(n_pre):
        adj_lists = sub_cluster_adj_lists[i]
        connected_components = []
        visited = {j: False for j in sub_clusters[i]}
        for j in sub_clusters[i]:
            if not visited[j]:
                connected_component = []
                stack = [j]
                while len(stack) > 0:
                    cur = stack.pop()
                    if not visited[cur]:
                        visited[cur] = True
                        connected_component.append(cur)
                        for neighbor in adj_lists[cur]:
                            stack.append(neighbor)
                connected_components.append(connected_component)
        sub_cluster_connected_components.append(connected_components)
    global_connected_components = []
    for i in range(n_pre):
        global_connected_components.extend(sub_cluster_connected_components[i])

    # build a map from point to connected component
    point_to_connected_component = {}
    for i in cur_sub_cluster:
        point_to_connected_component[i] = -1
    for connected_component_id, connected_component in enumerate(global_connected_components):
        for point in connected_component:
            point_to_connected_component[point] = connected_component_id

    # each point has a moving direction, we try to avoid any connected component was blocked by other connected components
    # first we need to find the connected components that are blocked, especially those small connected components
    # sort connected components by size
    connected_component_sizes = []
    for connected_component in global_connected_components:
        connected_component_sizes.append(len(connected_component))
    sorted_connected_components = np.argsort(connected_component_sizes)
    # check the neighbors of each connected component
    cc_adj_list = {}
    for i in range(len(global_connected_components)):
        cc_adj_list[i] = []
    for i in range(len(global_connected_components)):
        connected_component = global_connected_components[i]
        for point in connected_component:
            for neighbor in cur_adj_list[point]:
                if point_to_connected_component[neighbor] != i:
                    cc_adj_list[i].append(point_to_connected_component[neighbor])
    # print(f'cc_adj_list = {cc_adj_list}')
    cc_avg_direction = []
    for i in range(len(global_connected_components)):
        connected_component = global_connected_components[i]
        avg_direction = np.zeros(2)
        for point in connected_component:
            avg_direction += directions[point]
        avg_direction /= len(connected_component)
        cc_avg_direction.append(avg_direction)
    # print(f'cc_avg_direction = {cc_avg_direction}')
    cc_avg_center = []
    for i in range(len(global_connected_components)):
        connected_component = global_connected_components[i]
        avg_center = np.zeros(2)
        for point in connected_component:
            avg_center += cur_packing_pos[point]
        avg_center /= len(connected_component)
        cc_avg_center.append(avg_center)
    # categorize neighbors of each connected component into 6 categories according to the angle between the direction of the connected component and the line conneccting the center of current connected component and the neighboring circles
    # 0: 0-30, 1: 30-60, 2: 60-90, 3: 90-120, 4: 120-150, 5: 150-180
    cc_adj_category = {}
    for i in range(len(global_connected_components)):
        cc_adj_category[i] = [[], [], [], [], [], []]
    for i in range(len(global_connected_components)):
        connected_component = global_connected_components[i]
        for point in connected_component:
            # get the neighboring circles of the point
            for neighbor in cur_adj_list[point]:
                neighbor_cc = point_to_connected_component[neighbor]
                if neighbor_cc == i:
                    continue
                direction = cc_avg_direction[i]
                neighboring_direction = cc_avg_center[i] - cur_packing_pos[neighbor]
                angle = np.arccos(np.dot(direction, neighboring_direction) / (np.linalg.norm(direction) * np.linalg.norm(neighboring_direction)))
                category = int(angle / (np.pi / 6))
                cc_adj_category[i][category].append(neighbor_cc)
    # print(f'cc_adj_category = {cc_adj_category}')
    # if a connected component has a same neighbor in its 0 and 5 category, it is blocked
    blocked_connected_components = []
    for i in range(len(global_connected_components)):
        cate_0 = cc_adj_category[i][0]
        cate_5 = cc_adj_category[i][5]
        cate_1 = cc_adj_category[i][1]
        cate_4 = cc_adj_category[i][4]
        # concatenate cate_0 and cate_1
        cate_0.extend(cate_1)
        # concatenate cate_5 and cate_4
        cate_5.extend(cate_4)
        # get the intersection of cate_0 and cate_5
        intersection = list(set(cate_0).intersection(set(cate_5)))
        if len(intersection) > 0:
            blocked_connected_components.append(i)
    # print(f'blocked_connected_components = {blocked_connected_components}')
    blocker_points_list = []
    # for each blocked_connected_component, split the connected component that blocks it
    for blocked_connected_component in blocked_connected_components:
        # get the connected components that block it
        blockers = cc_adj_category[blocked_connected_component][0]
        # unique blockers
        blockers = list(set(blockers))
        for blocker in blockers:
            # find all circles in the blocker that are in the 0 category of the blocked_connected_component
            blocker_connected_component = global_connected_components[blocker]
            blocker_points = []
            for point in blocker_connected_component:
                direction = cc_avg_direction[blocked_connected_component]
                neighboring_direction = cc_avg_center[blocked_connected_component] - cur_packing_pos[point]
                angle = np.arccos(np.dot(direction, neighboring_direction) / (np.linalg.norm(direction) * np.linalg.norm(neighboring_direction)))
                category = int(angle / (np.pi / 6))
                if category <= 1:
                    blocker_points.append(point)
            blocker_points_list.append(blocker_points)

    # print(f'old sub_cluster_connected_components = {sub_cluster_connected_components}')
    # blocker_points_list records the points that should be split from the previous connected components
    # split the connected components
    # print(f'blocker_points_list = {blocker_points_list}')
    for i in range(n_pre):
        sub_cluster_ccs = sub_cluster_connected_components[i]
        for j in range(len(sub_cluster_ccs)):
            connected_component = sub_cluster_ccs[j]
            new_connected_components = []
            for blocker_points in blocker_points_list:
                new_connected_component = []
                for point in connected_component:
                    if point in blocker_points:
                        sub_cluster_ccs[j].remove(point)
                        new_connected_component.append(point)
                if len(new_connected_component) > 0:
                    new_connected_components.append(new_connected_component)
            sub_cluster_ccs.extend(new_connected_components)
        sub_cluster_connected_components[i] = sub_cluster_ccs
    # print(f'splited sub_cluster_connected_components = {sub_cluster_connected_components}')
    # for each connected component, find connected components further
    for i in range(n_pre):
        sub_cluster_ccs = sub_cluster_connected_components[i]
        new_sub_cluster_ccs = []
        for j in range(len(sub_cluster_ccs)):
            cc = sub_cluster_ccs[j]
            # find connected components in connected_component
            adj_list = {}
            for point in cc:
                adj_list[point] = []
            for point in cc:
                for neighbor in cur_adj_list[point]:
                    if neighbor in cc:
                        adj_list[point].append(neighbor)
            visited = {point: False for point in cc}
            for point in cc:
                if not visited[point]:
                    connected_component = []
                    stack = [point]
                    while len(stack) > 0:
                        cur = stack.pop()
                        if not visited[cur]:
                            visited[cur] = True
                            connected_component.append(cur)
                            for neighbor in adj_list[cur]:
                                stack.append(neighbor)
                    new_sub_cluster_ccs.append(connected_component)
        sub_cluster_connected_components[i] = new_sub_cluster_ccs
    # print(f'new sub_cluster_connected_components = {sub_cluster_connected_components}')





    # mean_pos_of_connected_components = []
    # for i in range(n_pre):
    #     connected_components = sub_cluster_connected_components[i]
    #     mean_pos_of_connected_components.append([])
    #     for connected_component in connected_components:
    #         mean_pos = np.mean(sub_cluster_packing_pos[i][connected_component], axis=0)
    #         mean_pos_of_connected_components[i].append(mean_pos)
    # main_direction = np.array(main_direction)
    # mean_proj_of_connected_components = []
    # for i in range(n_pre):
    #     mean_proj_of_connected_components.append([])
    #     for mean_pos in mean_pos_of_connected_components[i]:
    #         proj = np.dot(mean_pos, main_direction)
    #         mean_proj_of_connected_components[i].append(proj)
    # mean_proj_of_points = np.ones(len(cur_packing_pos))*-1000

    # for i in range(len(cur_packing_pos)):
    #     mean_proj_of_points[i] = mean_proj_of_connected_components[point_to_sub_cluster[i]][point_to_connected_component[i]]
    # # sort points by mean_proj
    # sorted_points = np.argsort(mean_proj_of_points)
    # # remove those points with mean_proj < -500
    # sorted_points = sorted_points[mean_proj_of_points[sorted_points] > -500]
    # # reverse the order
    # sorted_points = sorted_points[::-1]
    # each point has a moving direction, we try to avoid any connected component was blocked by other connected components
    # first we need to find the connected components that are blocked
    # for each connected_components, check its neighboring connected components

    # blocked_connected_components = []
    # for i in range(n_pre):


    # cur_edges = []
    # for i in range(n_pre):
    #     connected_components = sub_cluster_connected_components[i]
    #     for connected_component in connected_components:
    #         if len(connected_component) > 1:
    #             for j in range(len(connected_component)):
    #                 for k in range(j + 1, len(connected_component)):
    #                     p1 = connected_component[j]
    #                     p2 = connected_component[k]
    #                     if (p1, p2) in cur_neighbors or (p2, p1) in cur_neighbors:
    #                         cur_edges.append((p1, p2))
    # print(f'cur_neighbors = {cur_neighbors}')
    # print(f'cur_edges = {cur_edges}')
    # # find bridges in cur_edges using Tarjan's algorithm
    # bridges = []
    # adj_list = {}
    # for edge in cur_edges:
    #     p1, p2 = edge
    #     if p1 not in adj_list:
    #         adj_list[p1] = []
    #     if p2 not in adj_list:
    #         adj_list[p2] = []
    #     adj_list[p1].append(p2)
    #     adj_list[p2].append(p1)
    # visited = {j: False for j in range(len(cur_packing_pos))}
    # low = {j: -1 for j in range(len(cur_packing_pos))}
    # disc = {j: -1 for j in range(len(cur_packing_pos))}
    # parent = {j: -1 for j in range(len(cur_packing_pos))}
    # print(f'adj_list = {adj_list}')
    # def dfs(u, visited, low, disc, parent, times, adj_list, bridges):
    #     visited[u] = True
    #     low[u] = times
    #     disc[u] = times
    #     times += 1
    #     if u not in adj_list:
    #         return
    #     for v in adj_list[u]:
    #         if not visited[v]:
    #             parent[v] = u
    #             dfs(v, visited, low, disc, parent, times, adj_list, bridges)
    #             low[u] = min(low[u], low[v])
    #             if low[v] > disc[u]:
    #                 bridges.append((u, v))
    #         elif v != parent[u]:
    #             low[u] = min(low[u], disc[v])
    # times = 0
    # for i in cur_sub_cluster:
    #     if not visited[i]:
    #         dfs(i, visited, low, disc, parent, times, adj_list, bridges)
    # print(f'bridges = {bridges}')
    #
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    draw_pos = (cur_packing_pos)*10+0.5
    draw_radius = radii*10
    for point in cur_sub_cluster:
        cv2.circle(img, (int(draw_pos[point, 0] * 1000), int(draw_pos[point, 1] * 1000)),
                   int(draw_radius[point] * 1000), (255, 255, 255), 2)
        cv2.putText(img, f'{point}', (int(draw_pos[point, 0] * 1000), int(draw_pos[point, 1] * 1000)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    for i, neighbors in enumerate(sub_cluster_neighbors):
        color = cmap(i)
        for p1, p2 in neighbors:
            if (p1, p2) in cur_neighbors or (p2, p1) in cur_neighbors:
                cv2.line(img, (int(draw_pos[p1, 0] * 1000), int(draw_pos[p1, 1] * 1000)),
                         (int(draw_pos[p2, 0] * 1000), int(draw_pos[p2, 1] * 1000)),
                         (color[2]*255, color[1]*255, color[0]*255), 2)
    # for bridge in bridges:
    #     p1, p2 = bridge
    #     cv2.line(img, (int(draw_pos[p1, 0] * 1000), int(draw_pos[p1, 1] * 1000)),
    #              (int(draw_pos[p2, 0] * 1000), int(draw_pos[p2, 1] * 1000)), (255, 0, 0), 2)
    # highlight blocked connected components
    for i in blocked_connected_components:
        for point in global_connected_components[i]:
            cv2.circle(img, (int(draw_pos[point, 0] * 1000), int(draw_pos[point, 1] * 1000)),
                       int(draw_radius[point] * 1000/2), (0, 0, 255), 2)
    # draw directions
    for i in cur_sub_cluster:
        p1 = draw_pos[i].copy()
        p2 = draw_pos[i].copy()
        direction = directions[i]/np.linalg.norm(directions[i])
        p2[0] += direction[0]*0.5
        p2[1] += direction[1]*0.5
        cv2.line(img, (int(p1[0] * 1000), int(p1[1] * 1000)),
                    (int(p2[0] * 1000), int(p2[1] * 1000)), (255, 0, 0), 2)
    # highlight blocker points
    for blocker_points in blocker_points_list:
        for point in blocker_points:
            cv2.circle(img, (int(draw_pos[point, 0] * 1000), int(draw_pos[point, 1] * 1000)),
                       int(draw_radius[point] * 1000/3), (0, 255, 0), 2)
    cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_sub_cluster_cur.png', img)

    return sub_cluster_connected_components


def get_circle_hulls(packing_pos, radii, sub_clusters):
    # get the convex hull of each sub_cluster
    circle_hulls = []
    packing_pos = np.array(packing_pos)
    print(f'packing_pos = {packing_pos}')
    radii = np.array(radii)*1.05
    for i in range(len(sub_clusters)):
        sub_cluster = np.array(sub_clusters[i])
        print(f'sub_cluster = {sub_cluster}')
        sub_cluster_packing_pos = packing_pos[sub_cluster]
        print(f'sub_cluster_packing_pos = {sub_cluster_packing_pos}')
        sub_cluster_radii = radii[sub_cluster]
        print(f'sub_cluster_radii = {sub_cluster_radii}')
        sub_cluster_hull = []
        circle_pos = multi_interpolate(sub_cluster_packing_pos, sub_cluster_radii)
        circle_hull = computeConvexHull(circle_pos)
        circle_hull = approxConvexHull(circle_hull)
        print(f'circle_hull = {circle_hull}')
        circle_hulls.append(circle_hull)
        print(f'circle_hull_type = {type(circle_hull)}')
    # min_x = np.min(np_circle_hulls[:, :, 0])
    # max_x = np.max(np_circle_hulls[:, :, 0])
    # min_y = np.min(np_circle_hulls[:, :, 1])
    # max_y = np.max(np_circle_hulls[:, :, 1])
    min_x = 1e2
    max_x = -1e2
    min_y = 1e2
    max_y = -1e2
    for circle_hull in circle_hulls:
        for point in circle_hull:
            min_x = min(min_x, point[0])
            max_x = max(max_x, point[0])
            min_y = min(min_y, point[1])
            max_y = max(max_y, point[1])

    scale = max(max_x - min_x, max_y - min_y)
    print(f'scale = {scale}')
    print(f'min_x = {min_x}, max_x = {max_x}, min_y = {min_y}, max_y = {max_y}')
    scaled_circle_hulls = []
    for circle_hull in circle_hulls:
        scaled_circle_hull = []
        for point in circle_hull:
            scaled_point = [(point[0]-min_x)/scale, (point[1]-min_y)/scale]
            scaled_circle_hull.append(scaled_point)
        scaled_circle_hulls.append(np.array(scaled_circle_hull))
    print(f'scaled_circle_hulls = {scaled_circle_hulls}')
    time_stamp = time.time()
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for i in range(len(scaled_circle_hulls)):
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        for j in range(len(scaled_circle_hulls[i])):
            p1 = scaled_circle_hulls[i][j]
            p2 = scaled_circle_hulls[i][(j+1)%len(scaled_circle_hulls[i])]

            cv2.line(img, (int(p1[0]*1000), int(p1[1]*1000)), (int(p2[0]*1000), int(p2[1]*1000)), color, 2)
    cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_circle_hulls.png', img)
    return circle_hulls

def get_circle_hulls2(packing_pos, radii, pre_avg_end_pos, cur_sub_clusters, pre_sub_clusters, pre_packing_pos):
    # convert packing_pos to numpy array with shape (n, 2)
    packing_pos = np.array(packing_pos)
    radii = np.array(radii)
    pre_avg_end_pos = np.array(pre_avg_end_pos)
    # print(f'cur_sub_clusters = {cur_sub_clusters}')
    # print(f'packing_pos = {packing_pos}')
    # print(f'radii = {radii}')
    # print(f'pre_avg_end_pos = {pre_avg_end_pos}')
    # print(f'pre_sub_clusters ')
    # for i in range(len(pre_sub_clusters)):
    #     print(f'{i}: {pre_sub_clusters[i]}')
    # print(f'pre_packing_pos')
    # for i in range(len(pre_packing_pos)):
    #     print(f'{i}: {pre_packing_pos[i]}')
    # print(f'pre_sub_clusters = {pre_sub_clusters}')
    # print(f'pre_packing_pos = {pre_packing_pos}')
    id_to_pre = {}
    id_to_sub = {}
    for i in range(len(pre_sub_clusters)):
        for j in range(len(pre_sub_clusters[i])):
            for k in pre_sub_clusters[i][j]:
                id_to_pre[k] = i
                id_to_sub[k] = j

    pre_sub_cluster_avg_end_pos = []
    for i in range(len(pre_sub_clusters)):
        sub_cluster_avg_pos = []
        for j in range(len(pre_sub_clusters[i])):
            avg_pos = np.zeros(2)
            for id in pre_sub_clusters[i][j]:
                avg_pos += pre_packing_pos[i][id] + pre_avg_end_pos[i]
            avg_pos /= len(pre_sub_clusters[i][j])
            sub_cluster_avg_pos.append(avg_pos)
        pre_sub_cluster_avg_end_pos.append(sub_cluster_avg_pos)

    cur_sub_cluster_avg_end_pos = []
    for i in range(len(cur_sub_clusters)):
        cur_sub_cluster = cur_sub_clusters[i]
        avg_pos = np.zeros(2)
        for j in cur_sub_cluster:
            avg_pos += pre_sub_cluster_avg_end_pos[id_to_pre[j]][id_to_sub[j]]
        avg_pos /= len(cur_sub_cluster)
        cur_sub_cluster_avg_end_pos.append(avg_pos)

    min_x = 1e2
    max_x = -1e2
    min_y = 1e2
    max_y = -1e2
    for i in range(len(cur_sub_clusters)):
        min_x = min(min_x, cur_sub_cluster_avg_end_pos[i][0])
        max_x = max(max_x, cur_sub_cluster_avg_end_pos[i][0])
        min_y = min(min_y, cur_sub_cluster_avg_end_pos[i][1])
        max_y = max(max_y, cur_sub_cluster_avg_end_pos[i][1])
    scale = max(max_x - min_x, max_y - min_y)
    scaled_cur_sub_cluster_avg_end_pos = []
    # rescaled the cur_sub_cluster_avg_end_pos into [0, 1]*[0, 1]
    if len(cur_sub_clusters) == 1:
        scaled_cur_sub_cluster_avg_end_pos.append([0.5, 0.5])
    else:
        for i in range(len(cur_sub_clusters)):
            scaled_pos = [(cur_sub_cluster_avg_end_pos[i][0]-min_x)/scale*0.2, (cur_sub_cluster_avg_end_pos[i][1]-min_y)/scale*0.2]
            scaled_cur_sub_cluster_avg_end_pos.append(scaled_pos)

    # print(f'scale = {scale}')
    # print(f'min_x = {min_x}, max_x = {max_x}, min_y = {min_y}, max_y = {max_y}')
    cur_sub_cluster_center = []
    for i in range(len(cur_sub_clusters)):
        cur_sub_cluster = cur_sub_clusters[i]
        center = np.zeros(2)
        for j in cur_sub_cluster:
            center += packing_pos[j]
        center /= len(cur_sub_cluster)
        cur_sub_cluster_center.append(center)
    new_packing_pos = packing_pos.copy()
    for i in range(len(cur_sub_clusters)):
        cur_sub_cluster = cur_sub_clusters[i]
        for j in cur_sub_cluster:
            new_packing_pos[j] = packing_pos[j] - cur_sub_cluster_center[i] + scaled_cur_sub_cluster_avg_end_pos[i]
    packing_pos = new_packing_pos

    # get the convex hull of each sub_cluster
    circle_hulls = []
    packing_pos = np.array(packing_pos)
    print(f'packing_pos = {packing_pos}')
    # print(f'packing_pos = {packing_pos}')
    radii = np.array(radii) * 1.3
    for i in range(len(cur_sub_clusters)):
        sub_cluster = np.array(cur_sub_clusters[i])
        # print(f'sub_cluster = {sub_cluster}')
        sub_cluster_packing_pos = packing_pos[sub_cluster]
        # print(f'sub_cluster_packing_pos = {sub_cluster_packing_pos}')
        sub_cluster_radii = radii[sub_cluster]
        print(f'sub_cluster_packing_pos = {sub_cluster_packing_pos}')
        print(f'sub_cluster_radii = {sub_cluster_radii}')
        sub_cluster_packing_pos_list = []
        sub_cluster_radii_list = []
        for j in range(len(sub_cluster)):
            sub_cluster_packing_pos_list.append(sub_cluster_packing_pos[j])
            sub_cluster_radii_list.append(sub_cluster_radii[j])
        circle_pos = multi_interpolate(sub_cluster_packing_pos_list, sub_cluster_radii_list)

        # circle_pos = multi_interpolate(sub_cluster_packing_pos.tolist(), sub_cluster_radii.tolist())
        convex_hull = computeConvexHull(circle_pos)
        center_x, center_y, radius = find_circle(convex_hull.exterior.coords[:-1])
        circle_hull = [center_x, center_y, radius]
        circle_hulls.append(circle_hull)
    min_x = 1e2
    max_x = -1e2
    min_y = 1e2
    max_y = -1e2
    # print(f'circle_hulls = {circle_hulls}')
    for [x,y,r] in circle_hulls:
            min_x = min(min_x, x-r)
            max_x = max(max_x, x+r)
            min_y = min(min_y, y-r)
            max_y = max(max_y, y+r)

    scale = max(max_x - min_x, max_y - min_y)
    # print(f'scale = {scale}')
    # print(f'min_x = {min_x}, max_x = {max_x}, min_y = {min_y}, max_y = {max_y}')
    scaled_circle_hulls = []
    for [x,y,r] in circle_hulls:
        scaled_point = [(x-min_x)/scale, (y-min_y)/scale, r/scale]
        scaled_circle_hulls.append(scaled_point)
    # print(f'scaled_circle_hulls = {scaled_circle_hulls}')
    time_stamp = time.time()
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # print(f'scaled_circle_hulls = {scaled_circle_hulls}')
    for [x,y,r] in scaled_circle_hulls:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.circle(img, (int((x)*1000), int((y)*1000)), int(r*1000), color, 2)
    # for i in range(len(packing_pos)):
    #     x = (packing_pos[i][0]-min_x)/scale
    #     y = (packing_pos[i][1]-min_y)/scale
    #     r = radii[i]/scale
    #     cv2.circle(img, (int((x)*1000), int((y)*1000)), int(r*1000), (255, 255, 255), 2)
    #     cv2.putText(img, f'{i}', (int((x)*1000), int((y)*1000)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    # cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_circle_hulls.png', img)
    for i in range(len(cur_sub_clusters)):
        for id in cur_sub_clusters[i]:
            x = (packing_pos[id][0]-min_x)/scale
            y = (packing_pos[id][1]-min_y)/scale
            r = radii[id]/scale
            cv2.circle(img, (int((x)*1000), int((y)*1000)), int(r*1000), (255, 255, 255), 2)
            cv2.putText(img, f'{id}', (int((x)*1000), int((y)*1000)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.imwrite(f'./log/image/PackingProcess/{time_stamp}_circle_hulls.png', img)
    return circle_hulls, packing_pos
