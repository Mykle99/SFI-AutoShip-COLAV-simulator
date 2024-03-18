"""
    map_functions.py

    Summary:
        Contains functionality for plotting the Electronic Navigational Chart (ENC),
        computing distance to land polygons, generating random ship starting positions etc.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""

import copy
import os
import colav_simulator.common.miscellaneous_helper_methods as mhm
import geopandas as gpd
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as scipy_spatial
import seacharts.display.colors as colors
import shapely
import shapely.ops as ops
import math

from typing import Optional, Tuple
from cartopy.feature import ShapelyFeature
from osgeo import osr
from seacharts.enc import ENC
from shapely import affinity, strtree
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon

os.environ["USE_PYGEOS"] = "0"


def create_bbox_from_points(
    enc: ENC, p1: np.ndarray, p2: np.ndarray, buffer: float = 200.0
) -> Tuple[float, float, float, float]:
    """Creates a bounding box from two diagonal corner points.

    Args:
        p1 (np.ndarray): First corner point.
        p2 (np.ndarray): Second corner point.

    Returns:
        Tuple[float, float, float, float]: Bounding box (xmin, ymin, xmax, ymax), with x being easting.
    """
    xmin = min(p1[0], p2[0]) - buffer
    xmax = max(p1[0], p2[0]) + buffer
    ymin = min(p1[1], p2[1]) - buffer
    ymax = max(p1[1], p2[1]) + buffer
    xmin = max(xmin, enc.bbox[1])
    xmax = min(xmax, enc.bbox[3])
    ymin = max(ymin, enc.bbox[0])
    ymax = min(ymax, enc.bbox[2])
    return ymin, xmin, ymax, xmax


def local2latlon(
    x: float | list | np.ndarray, y: float | list | np.ndarray, utm_zone: int
) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
    """Transform coordinates from x (east), y (north) to latitude, longitude.

    Args:
        x (float | list): East coordinate(s) in a local UTM coordinate system.
        y (float | list): North coordinate(s) in a local UTM coordinate system.
        utm_zone (int): UTM zone.

    Raises:
        ValueError: If the input string is not correct.

    Returns:
        Tuple[float | list | np.ndarray, float | list | np.ndarray]: Tuple of latitude and longitude coordinates.
    """
    to_zone = 4326  # Latitude Longitude
    if utm_zone == 32:
        from_zone = 6172  # ETRS89 / UTM zone 32 + NN54 height Møre og Romsdal
    elif utm_zone == 33:
        from_zone = 6173  # ETRS89 / UTM zone 33 + NN54 height
    else:
        raise ValueError('Input "utm_zone" is not correct. Supported zones sofar are 32 and 33.')

    src = osr.SpatialReference()
    src.ImportFromEPSG(from_zone)
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(to_zone)
    transform = osr.CoordinateTransformation(src, tgt)

    if isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray)):
        coordinates = transform.TransformPoints(list(zip(x, y)))
        lat = [coord[0] for coord in coordinates]
        lon = [coord[1] for coord in coordinates]
    else:
        lat, lon, _ = transform.TransformPoint(x, y)

    return lat, lon


def latlon2local(
    lat: float | list | np.ndarray, lon: float | list | np.ndarray, utm_zone: int
) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
    """Transform coordinates from latitude, longitude to UTM32 or UTM33

    Args:
        lat (float | list): Latitude coordinate(s)
        lon (float | list): Longitude coordinate(s)
        utm_zone (str): UTM zone.

    Raises:
        ValueError: If the input string is not correct.

    Returns:
        Tuple[float | list | np.ndarray, float | list | np.ndarray]: Tuple of east and north coordinates.
    """
    from_zone = 4326  # Latitude Longitude
    if utm_zone == 32:
        to_zone = 6172  # ETRS89 / UTM zone 32 + NN54 height Møre og Romsdal
    elif utm_zone == 33:
        to_zone = 6173  # ETRS89 / UTM zone 33 + NN54 height
    else:
        raise ValueError('Input "utm_zone" is not correct. Supported zones sofar are 32 and 33.')

    src = osr.SpatialReference()
    src.ImportFromEPSG(from_zone)
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(to_zone)
    transform = osr.CoordinateTransformation(src, tgt)

    if isinstance(lat, (list, np.ndarray)) and isinstance(lon, (list, np.ndarray)):
        coordinates = transform.TransformPoints(list(zip(lat, lon)))
        x = [coord[0] for coord in coordinates]
        y = [coord[1] for coord in coordinates]
    else:
        x, y, _ = transform.TransformPoint(lat, lon)

    return x, y


def dist_between_latlon_coords(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes the distance between two latitude, longitude coordinates.

    Args:
        lat1 (float): Latitude of first coordinate.
        lon1 (float): Longitude of first coordinate.
        lat2 (float): Latitude of second coordinate.
        lon2 (float): Longitude of second coordinate.

    Returns:
        float: Distance between the two coordinates in meters
    """
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).m


def create_point_list_from_polygons(polygons: list) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a list of x and y coordinates from a list of polygons.

    Args:
        polygons (list): List of shapely polygons.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the x (north) and y (east) coordinates of the polygons.
    """
    px, py, ls = [], [], []
    for i, poly in enumerate(polygons):
        y, x = poly.exterior.coords.xy
        a = np.array(x.tolist())
        b = np.array(y.tolist())
        la, lx = len(a), len(px)
        c = [(i + lx, (i + 1) % la + lx) for i in range(la - 1)]
        px += a.tolist()
        py += b.tolist()
        ls += c

    points = np.array([px, py]).T
    P1, P2 = points[ls][:, 0], points[ls][:, 1]
    return P1, P2


def extract_vertices_from_polygon_list(polygons: list) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a list of x and y coordinates from a list of polygons.

    Args:
        polygons (list): List of shapely polygons.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the x (north) and y (east) coordinates of the polygons.
    """
    px, py = [], []
    for i, poly in enumerate(polygons):
        if isinstance(poly, MultiPolygon):
            for sub_poly in poly:
                y, x = sub_poly.exterior.coords.xy
                px.extend(x[:-1].tolist())
                py.extend(y[:-1].tolist())
        elif isinstance(poly, Polygon):
            y, x = poly.exterior.coords.xy
            px.extend(x[:-1].tolist())
            py.extend(y[:-1].tolist())
        else:
            continue
    return np.array(px), np.array(py)


def extract_safe_sea_area(
    min_depth: int,
    enveloping_polygon: Polygon,
    enc: Optional[ENC] = None,
    as_polygon_list: bool = False,
    buffer: Optional[float] = None,
    show_plots: bool = False,
) -> MultiPolygon | list:
    """Extracts the safe sea area from the ENC as a list of polygons.

    This includes sea polygons that are above the vessel's minimum depth.

    Args:
        - min_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enveloping_polygon (geometry.Polygon): The query polygon.
        - enc (Optional[senc.ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.
        - as_polygon_list (bool, optional): Option for returning the safe sea area as a list of polygons. Defaults to False.
        - buffer (Optional[float], optional): Safety buffer for polygons. Defaults to None.
        - show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        MultiPolygon | list: The safe sea area.
    """
    seabed = enc.seabed[min_depth].geometry
    if buffer is not None:
        seabed = seabed.buffer(-buffer)
    safe_sea = seabed.intersection(enveloping_polygon)

    if enc is not None and show_plots:
        enc.start_display()
        enc.draw_polygon(safe_sea, color="green", alpha=0.25, fill=False)

    if as_polygon_list:
        if isinstance(safe_sea, MultiPolygon):
            return [poly for poly in safe_sea.geoms]
        elif isinstance(safe_sea, Polygon):
            return [safe_sea]
        else:
            return []
    return safe_sea


def create_free_boundary_points_from_enc(enc: ENC, hazards: list) -> Tuple[np.ndarray, np.ndarray]:
    """Creates an array of points on the ENC boundary which is free from grounding hazards.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        hazards (list): List of relevant grounding hazards.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of x and y coordinates of the free boundary points.
    """
    (xmin, ymin, xmax, ymax) = enc.bbox
    n_pts_per_side = 50
    x = np.linspace(xmin, xmax, n_pts_per_side)
    y = np.linspace(ymin, ymax, n_pts_per_side)
    points = []
    for i in range(n_pts_per_side):
        p = Point(x[i], ymin)
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = Point(x[i], ymax)
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = Point(xmin, y[i])
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = Point(xmax, y[i])
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)
    # [enc.draw_circle((p.x, p.y), radius=0.5, color="yellow") for p in points]
    Y = np.array([p.x for p in points])
    X = np.array([p.y for p in points])
    return X, Y


def bbox_to_polygon(bbox: Tuple[float, float, float, float]) -> Polygon:
    """Converts a bounding box to a polygon.

    Args:
        bbox (Tuple[float, float, float, float]): The bounding box (xmin, ymin, xmax, ymax), with x being easting.

    Returns:
        Polygon: The polygon.
    """
    (xmin, ymin, xmax, ymax) = bbox
    return Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])


def point_in_polygon_list(point: Point, polygons: list) -> bool:
    """Checks if a point is in a list of polygons.

    Args:
        point (Point): The point to check.
        polygons (list): List of polygons.

    Returns:
        bool: True if the point is in a hazard, False otherwise.
    """
    for poly in polygons:
        if point.within(poly) or point.touches(poly):
            return True
    return False


def point_in_point_list(point: Point, points: list) -> bool:
    """Checks if a point is in a list of points.

    Args:
        point (Point): The point to check.
        points (list): List of points.

    Returns:
        bool: True if the point is in a hazard, False otherwise.
    """
    for p in points:
        if point.within(p) or point.touches(p):
            return True
    return False


def create_safe_sea_voronoi_diagram(enc: ENC, vessel_min_depth: int = 5) -> Tuple[scipy_spatial.Voronoi, list]:
    """Creates a Voronoi diagram of the safe sea region (i.e. its vertices).

    Args:
        enc (ENC): The Electronic Navigational Chart object.
        vessel_min_depth (float): The safe minimum depth for the vessel to voyage in.

    Returns:
        scipy_spatial.Voronoi: The Voronoi diagram of the safe sea region.
    """
    bbox = enc.bbox
    enc_bbox_poly = bbox_to_polygon(bbox)
    safe_sea = extract_safe_sea_area(vessel_min_depth, enc_bbox_poly, enc, as_polygon_list=True, show_plots=True)
    polygons = []
    for sea_poly in safe_sea:
        if isinstance(sea_poly, MultiPolygon):
            for poly in sea_poly:
                polygons.append(poly)
        elif isinstance(sea_poly, Polygon):
            polygons.append(sea_poly)
        else:
            continue
    px, py = extract_vertices_from_polygon_list(polygons)
    points = np.vstack((py, px)).T
    vor = scipy_spatial.Voronoi(points)
    region_polygons = create_region_polygons_from_voronoi(vor, enc=enc)
    for point in points:
        enc.draw_circle((point[0], point[1]), radius=0.4, color="red")

    # # Keep all voronoi region boundary points that are in the safe sea area
    # safe_points = []
    # for region in vor.regions:
    #     region_vertices = vor.vertices[region]
    #     for vertex in region_vertices:
    #         point = Point(vertex)
    #         if point_in_polygon_list(point, polygons):
    #             safe_points.append((point.x, point.y))
    # settt = set(safe_points)
    # safe_points = list(settt)
    # for point in safe_points:
    #     enc.draw_circle((point[0], point[1]), radius=0.4, color="magenta")
    return vor, region_polygons


def create_safe_sea_triangulation(
    enc: ENC,
    vessel_min_depth: int = 5,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    buffer: Optional[float] = None,
    show_plots: bool = True,
) -> list:
    """Creates a constrained delaunay triangulation of the safe sea region.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        vessel_min_depth (int, optional): The safe minimum depth for the vessel to voyage in. Defaults to 5.
        bbox (Optional[Tuple[float, float, float, float]]): Bounding box of the safe sea region to constrain the cdt within. Defaults to None.
        buffer (Optional[float], optional): Safety buffer for polygons. Defaults to None.
        show_plots (bool, optional): Option for visualization. Defaults to True.

    Returns:
        list: List of triangles.
    """
    if bbox is None:
        bbox = enc.bbox

    safe_sea_poly_list = extract_safe_sea_area(
        vessel_min_depth, bbox_to_polygon(bbox), enc, as_polygon_list=True, buffer=buffer, show_plots=show_plots
    )
    cdt_list = []
    largest_poly_area = 0.0
    for poly in safe_sea_poly_list:
        cdt = constrained_delaunay_triangulation_custom(poly)
        if poly.area > largest_poly_area:
            largest_poly_area = poly.area
            cdt_largest = cdt
        if show_plots:
            # enc.draw_polygon(poly, color="blue", alpha=0.2)
            enc.start_display()
            for triangle in cdt:
                enc.draw_polygon(triangle, color="green", fill=False)
        cdt_list.append(cdt)

    return cdt_largest


def create_region_polygons_from_voronoi(vor: scipy_spatial.Voronoi, enc: Optional[ENC] = None) -> list:
    """Creates a list of polygons from the Voronoi diagram.

    Args:
        vor (scipy_spatial.Voronoi): The Voronoi diagram.
        enc (Optional[ENC], optional): The Electronic Navigational Chart object. Defaults to None.

    Returns:
        list: List of polygons.
    """
    polygons = []
    for region in vor.regions:
        if not region:
            continue
        region_vertices = vor.vertices[region]
        if region_vertices.shape[0] < 3:
            continue
        region_poly = Polygon(region_vertices)
        if region_poly.area < 1.0:
            continue
        polygons.append(region_poly)
        if enc:
            enc.start_display()
            enc.draw_polygon(region_poly, color="yellow", alpha=0.5)
    return polygons


def create_ship_polygon(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    length_scaling: float = 1.0,
    width_scaling: float = 1.0,
) -> Polygon:
    """Creates a ship polygon from the ship's position, heading, length and width.

    Args:
        x (float): The ship's north position
        y (float): The ship's east position
        heading (float): The ship's heading
        length (float): Length of the ship
        width (float): Width of the ship
        length_scaling (float, optional): Length scale factor. Defaults to 1.0.
        width_scaling (float, optional): Length scale factor. Defaults to 1.0.

    Returns:
        np.ndarray: Ship polygon
    """
    eff_length = length * length_scaling
    eff_width = width * width_scaling

    x_min, x_max = x - eff_length / 2.0, x + eff_length / 2.0 - eff_width
    y_min, y_max = y - eff_width / 2.0, y + eff_width / 2.0
    left_aft, right_aft = (y_min, x_min), (y_max, x_min)
    left_bow, right_bow = (y_min, x_max), (y_max, x_max)
    coords = [left_aft, left_bow, (y, x + eff_length / 2.0), right_bow, right_aft]
    poly = Polygon(coords)
    return affinity.rotate(poly, -heading, origin=(y, x), use_radians=True)


def plot_shapely_multipolygon(
    ax: plt.Axes, mp: MultiPolygon, color: str, fill: bool = True, alpha: float = 1.0, zorder: int = 1
) -> plt.Axes:
    """Plots a shapely MultiPolygon object on a matplotlib axes.

    Args:
        ax (plt.Axes): Matplotlib axes handle.
        mp (MultiPolygon): MultiPolygon object to plot.
        color (str, optional): Color of the MultiPolygon.
        fill (bool, optional): Option for filling the MultiPolygon. Defaults to False.
        alpha (float, optional): Transparency of the MultiPolygon. Defaults to 1.0.
        zorder (int, optional): Z-order of the MultiPolygon. Defaults to 1.
    """
    if isinstance(mp, Polygon):
        mp = MultiPolygon([mp])

    for poly in mp.geoms:
        if fill:
            ax.fill(*poly.exterior.xy, color=color, alpha=alpha, zorder=zorder)
        else:
            ax.plot(*poly.exterior.xy, color=color, zorder=zorder)
    return ax


def plot_background(
    ax: plt.Axes,
    enc: ENC,
    show_shore: bool = True,
    show_seabed: bool = True,
    dark_mode: bool = True,
    uniform_seabed_color: bool = False,
) -> None:
    """Creates a static background based on the input seacharts

    Args:
        ax (plt.Axes): Matplotlib axes handle.
        enc (ENC): Electronic Navigational Chart object
        show_shore (bool, optional): Option for showing the shore. Defaults to True.
        show_seabed (bool, optional): Option for showing the seabed. Defaults to True.
        dark_mode (bool, optional): Option for dark mode. Defaults to True.
        uniform_seabed_color (bool, optional): Option for using a uniform color for the seabed. Defaults to False.

    Returns:
        Tuple[]: Tuple of limits in x and y for the background extent
    """
    # For every layer put in list and assign a color
    if enc.land:
        color = "#142c38" if dark_mode else colors.color_picker(enc.land.color)
        plot_shapely_multipolygon(ax, enc.land.geometry, color=color, zorder=enc.land.z_order)

    if show_shore and enc.shore:
        color = "#142c38" if dark_mode else colors.color_picker(enc.shore.color)
        plot_shapely_multipolygon(ax, enc.shore.geometry, color=color, zorder=enc.shore.z_order)

    if show_seabed and enc.seabed:
        bins = len(enc.seabed.keys())
        count = 0
        for _, layer in enc.seabed.items():
            if uniform_seabed_color:
                rank = enc.seabed[0].z_order
                color = colors.color_picker(0, bins)
            else:
                rank = layer.z_order + count
                color = colors.color_picker(count, bins)
            plot_shapely_multipolygon(ax, layer.geometry, color=color, zorder=rank)
            count += 1

    x_min, y_min, x_max, y_max = enc.bbox
    ax.set_xlim((x_min, x_max))  # Easting
    ax.set_ylim((y_min, y_max))  # Northing


def find_minimum_depth(vessel_draft: float, enc: ENC):
    """Find the minimum seabed depth for the given vessel draft (for it to avoid grounding)

    Args:
        vessel_draft (float): The vessel's draft.

    Returns:
        float: The minimum seabed depth required for a safe journey for the vessel.
    """
    lowest_possible_depth = 0
    for depth in enc.seabed:
        if vessel_draft <= float(depth):
            lowest_possible_depth = depth
            break
    return lowest_possible_depth


def extract_relevant_grounding_hazards(vessel_min_depth: int, enc: ENC) -> list:
    """Extracts the relevant grounding hazards from the ENC as a list of (multi) polygons.

    This includes land, shore and seabed polygons that are below the vessel's minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = (
        enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
        if vessel_min_depth > 0
        else MultiPolygon()
    )
    return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]


def extract_relevant_grounding_hazards_as_union(
    vessel_min_depth: int, enc: ENC, buffer: Optional[float] = None, show_plots: bool = False
) -> list:
    """Extracts the relevant grounding hazards from the ENC as a multipolygon.

    This includes land, shore and seabed polygons that are below the vessel's minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.
        buffer (Optional[float], optional): Buffer for polygons. Defaults to None.
        show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = (
        enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
        if vessel_min_depth > 0
        else MultiPolygon()
    )
    # return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]
    relevant_hazards = [enc.land.geometry.union(enc.shore.geometry).union(dangerous_seabed)]
    filtered_relevant_hazards = []
    for hazard in relevant_hazards:
        if isinstance(hazard, MultiPolygon):
            poly = MultiPolygon(Polygon(p.exterior) for p in hazard.geoms if isinstance(p, Polygon))
        elif isinstance(hazard, Polygon):
            poly = MultiPolygon([Polygon(hazard.exterior)])
        else:
            continue

        if buffer is not None:
            poly = poly.buffer(buffer)

        # remove interior
        if isinstance(poly, MultiPolygon):
            poly = MultiPolygon(Polygon(p.exterior) for p in poly.geoms if isinstance(p, Polygon))
        filtered_relevant_hazards.append(poly)

    if show_plots:
        enc.start_display()
        for hazard in filtered_relevant_hazards:
            if isinstance(hazard, MultiPolygon):
                for poly in hazard.geoms:
                    enc.draw_polygon(poly, color="red", fill=False)
            else:
                enc.draw_polygon(hazard, color="red", fill=False)
    return filtered_relevant_hazards


def extract_grounding_hazards_from_relevant_sector_in_enc(
    grounding_hazards : list[MultiPolygon],
    ownship_state : np.ndarray, 
    radius_of_coverage : float,
    angle_of_coverage_behind : float, 
    enc: ENC, 
    show_plots: bool = False) -> list[MultiPolygon]:
    """Filters the grounding hazards by means of defining a relevant 
    grounding sector. Only hazards which are present in this sector
    are saved. The relevant grounding sector is defined by the vessel 
    position, vessel cog, chosen angle_of_coverage and chosen radius_of_coverage.
    The sector uses the vessel position as its center, and must thus be updated in
    due time, if the vessel position changes.

    Args:
    - grounding_hazards (list[MultiPolygon]): Grounding hazards from an ENC parameterized as polygons. 
    - ownship_state (np.ndarray): The ownship state [x, y, psi, u, v, r].
    - angle_of_coverage_behind (float): The angle of which to concider grounding hazards in the
    opposite direction of cog, in degrees.
    - radius_of_coverage (float): The radius of which to concider grounding hazards, the radius of the
    relevant grounding sector. 
    - enc (senc.ENC): The ENC to check for grounding hazards.
    - show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
    list[MultiPolygon]: The grounding hazards inside the relevant grounding sector.
    """
    # Defining a relevant sector.
    ownship_x = ownship_state[0]
    ownship_y = ownship_state[1]
    ownship_cog = ownship_state[2]
    ownship_cog_rad = math.radians(ownship_cog)
    angle_of_coverage_behind_rad = math.radians(90 - angle_of_coverage_behind)

    # Defining a ship point.
    ownship_point = Point(ownship_x, ownship_y)

    # Defining the vertices of a triangle.
    coverage_neg_point = Point(
        ownship_x - radius_of_coverage * math.sin(ownship_cog_rad - angle_of_coverage_behind_rad),
        ownship_y - radius_of_coverage * math.cos(ownship_cog_rad - angle_of_coverage_behind_rad)
    )
    coverage_positive_point = Point(
        ownship_x - radius_of_coverage * math.sin(ownship_cog_rad + angle_of_coverage_behind_rad),
        ownship_y - radius_of_coverage * math.cos(ownship_cog_rad + angle_of_coverage_behind_rad)
    )

    # Using the triangle to define a circle, which is our area of interest (relevant grounding sector).
    center_relevant_sector = Point(
        (coverage_positive_point.x + coverage_neg_point.x) / 2, 
        (coverage_positive_point.y + coverage_neg_point.y) / 2
    )
    radius_relevant_sector = coverage_neg_point.distance(coverage_positive_point)/2
    circle_relevant_sector = Point(
        center_relevant_sector.x + radius_relevant_sector * math.sin(ownship_cog_rad), 
        center_relevant_sector.y + radius_relevant_sector * math.cos(ownship_cog_rad)
    ).buffer(radius_relevant_sector, resolution = 2) # Resolution gives the number of vertices in the "circle".
    
    # Rotate the area of interest to be aligned with the ships COG angle.
    circle_relevant_sector = affinity.rotate(circle_relevant_sector, -ownship_cog, "center")
    
    # Finding the intersection of relevant hazards and relevant area.
    filtered_relevant_hazards_circle_relevant_sector_intersection = circle_relevant_sector.intersection(
        grounding_hazards[0]
    )

    # Finding the center of the relevant region.
    rel_sec_center_x = center_relevant_sector.x + radius_relevant_sector * math.sin(ownship_cog_rad)
    rel_sec_center_y = center_relevant_sector.y + radius_relevant_sector * math.cos(ownship_cog_rad)

    # Finding the coordinates of the endpoint at the "end" of the relevant region.
    rel_sec_behind_x = rel_sec_center_x - 10 * radius_of_coverage * math.sin(ownship_cog_rad)
    rel_sec_behind_y = rel_sec_center_y - 10 * radius_of_coverage * math.cos(ownship_cog_rad)

    # Finding the coordinates of the endpoint at the "front" of the relevant region.
    rel_sec_ahead_x = rel_sec_center_x + 10 * radius_of_coverage * math.sin(ownship_cog_rad)
    rel_sec_ahead_y = rel_sec_center_y + 10 * radius_of_coverage * math.cos(ownship_cog_rad)

    # Defining endpoints and the center point.
    rel_sec_center = Point(rel_sec_center_x, rel_sec_center_y)
    rel_sec_behind_endpoint = Point(rel_sec_behind_x, rel_sec_behind_y)
    rel_sec_ahead_endpoint = Point(rel_sec_ahead_x, rel_sec_ahead_y)

    # Plotting functionality that shows the "circle" of relevant static hazards, ownship, ownship line
    # and the hazardous polygons (before pruning of non desired points).
    if show_plots:
        sector_triangle = Polygon([ownship_point, coverage_neg_point, coverage_positive_point, ownship_point])
        enc.start_display()
        enc.draw_line([rel_sec_behind_endpoint, rel_sec_center, rel_sec_ahead_endpoint], color = "black", width = 3)
        enc.add_vessels((0, int(ownship_x), int(ownship_y), int(ownship_cog), "black"))
        enc.draw_polygon(sector_triangle, color = "green", alpha = 0.5)
        enc.draw_polygon(circle_relevant_sector, color = "orange", alpha = 0.5)
        enc.draw_circle([rel_sec_center.x, rel_sec_center.y], 30, "magenta", thickness = 4.5, fill = True)
        enc.draw_polygon(filtered_relevant_hazards_circle_relevant_sector_intersection, color = "red", alpha = 0.6)
        enc.show_display()

    # If there is only one hazardous polygon the type will be of Polygon.
    # This needs to be changed to MultiPolygon to avoid errors in the
    # subsequent data processing.
    if type(filtered_relevant_hazards_circle_relevant_sector_intersection) != MultiPolygon:
        filtered_relevant_hazards_circle_relevant_sector_intersection = \
            MultiPolygon([filtered_relevant_hazards_circle_relevant_sector_intersection])

    # Removing polygons based on LOS.
    filtered_visible_hazard_polygons = []
    for hazard_polygon in filtered_relevant_hazards_circle_relevant_sector_intersection.geoms:
        # Finding the center of the hazard polygon that is checked.
        hazard_polygon_center_point = hazard_polygon.centroid

        # Defining a line from the ownship to the center of the hazard polygon.
        los_line_from_os_to_hazard_poly = LineString(
            [ownship_point, (hazard_polygon_center_point.x, hazard_polygon_center_point.y)]
        )

        # Checking if any other polygon in the MultiPolygon obstructs the LOS between
        # the ownship and the given hazard polygon.
        hazard_poly_is_obstructed = False
        for other_hazard_polygon in filtered_relevant_hazards_circle_relevant_sector_intersection.geoms:
            if hazard_polygon != other_hazard_polygon:
                if los_line_from_os_to_hazard_poly.intersects(other_hazard_polygon):
                    hazard_poly_is_obstructed = True
                    break
                
        # If the LOS is not obstructed then the hazard_polygon will kept for further processing.
        if not hazard_poly_is_obstructed:
            filtered_visible_hazard_polygons.append(hazard_polygon)

    # Creating a new MultiPolygon from the visible polygons.
    filtered_visible_hazard_polygons = MultiPolygon(filtered_visible_hazard_polygons)
    
    # Bookkeeping lists used to keep track of extreme points in both directions (along ship line + normal to ship line).
    furthest_points_along_line_multi_poly = []
    furthest_points_normal_from_haz_poly_reference_line_multi_poly = []

    along_haz_poly_reference_line_points_id_multi_poly = []
    normal_from_haz_poly_reference_line_points_id_multi_poly = []
    last_hazard_point_id_multi_poly = []

    hazard_poly_id = 0
    hazard_point_id = None
    
    # Finding extreme points for all the polygons.
    for hazard_polygon in filtered_visible_hazard_polygons.geoms:
        # Finding the center of the hazard polygon that is processed.
        hazard_polygon_center_point = hazard_polygon.centroid

        # Finding the y-axis based on the hazard_polygon_center_point.
        haz_poly_reference_line_normal = LineString([
            ownship_point, hazard_polygon_center_point
        ])

        direction_vector = [
            Point(haz_poly_reference_line_normal.coords[1]).x - Point(haz_poly_reference_line_normal.coords[0]).x,
            Point(haz_poly_reference_line_normal.coords[1]).y - Point(haz_poly_reference_line_normal.coords[0]).y
        ]

        # Finding the x-axis which is aligned with the hazard polygon.
        haz_poly_reference_line = LineString([
            Point(ownship_point.x - direction_vector[1] * 10, ownship_point.y + direction_vector[0] * 10),
            Point(ownship_point.x + direction_vector[1] * 10, ownship_point.y - direction_vector[0] * 10)
        ])

        # Finding the y-axis which is aligned with the hazard polygon.
        haz_poly_reference_line_normal = affinity.rotate(haz_poly_reference_line, 90, ownship_point)

        # Redefining for every polygon.
        furthest_points_along_line = [None, None, None, None]
        furthest_points_normal_from_haz_poly_reference_line = [None, None]

        furthest_distances_along_haz_poly_reference_line = [0, 0, 0, 0]
        furthest_distance_normal_from_haz_poly_reference_line = [0, 0]

        along_haz_poly_reference_line_points_id = [None, None, None, None]
        normal_from_haz_poly_reference_line_points_id = [None, None]
        
        # Appending the last hazard_point_id for each (but the last) polygon.
        if hazard_point_id is not None:
            last_hazard_point_id_multi_poly.append(hazard_point_id)
        hazard_point_id = 0
        
        for hazard_point_tuple in hazard_polygon.exterior.coords:
            hazard_point = Point(hazard_point_tuple)

            # Projecting the dangerous point onto the ownship line.
            projected_point_x = haz_poly_reference_line.interpolate(haz_poly_reference_line.project(hazard_point))

            # Projecting the dangerous point onto the normal of the ownship line.
            projected_point_y = haz_poly_reference_line_normal.interpolate(haz_poly_reference_line_normal.project(hazard_point))

            # Calculate the signed distances from the projected points to the origin.
            signed_distance_x = ownship_point.distance(projected_point_x) * \
                (1 if projected_point_x.x >= ownship_point.x else -1)
            signed_distance_y = ownship_point.distance(projected_point_y) * \
                (1 if projected_point_y.y >= ownship_point.y else -1)

            # Finding which quadrant each point belongs to (wrt. the rotated coordinate system). 
            # Origin in the center of the relevant sector.
            if signed_distance_x >= 0 and signed_distance_y >= 0:
                quadrant = 0
            elif signed_distance_x < 0 and signed_distance_y >= 0:
                quadrant = 1
            elif signed_distance_x < 0 and signed_distance_y < 0:
                quadrant = 2
            elif signed_distance_x >= 0 and signed_distance_y < 0:
                quadrant = 3

            # Using the unsigned distance for comparisons.
            distance_along_haz_poly_reference_line = abs(signed_distance_x)

            # Update the furthest distance and point along the line.
            if distance_along_haz_poly_reference_line > furthest_distances_along_haz_poly_reference_line[quadrant]:
                furthest_distances_along_haz_poly_reference_line[quadrant] = distance_along_haz_poly_reference_line
                furthest_points_along_line[quadrant] = hazard_point
                along_haz_poly_reference_line_points_id[quadrant] = hazard_point_id

            # Calculate the distance of the normal of the ownship line to the hazard point.
            distance_normal_from_haz_poly_reference_line = hazard_point.distance(projected_point_x)

            # Update the furthest distance and point normal to the line.
            if (quadrant == 0 or quadrant == 1):
                if (distance_normal_from_haz_poly_reference_line > furthest_distance_normal_from_haz_poly_reference_line[0]):
                    furthest_distance_normal_from_haz_poly_reference_line[0] = distance_normal_from_haz_poly_reference_line
                    furthest_points_normal_from_haz_poly_reference_line[0] = hazard_point
                    normal_from_haz_poly_reference_line_points_id[0] = hazard_point_id
            elif (quadrant == 2 or quadrant == 3):
                if (distance_normal_from_haz_poly_reference_line > furthest_distance_normal_from_haz_poly_reference_line[1]):
                    furthest_distance_normal_from_haz_poly_reference_line[1] = distance_normal_from_haz_poly_reference_line
                    furthest_points_normal_from_haz_poly_reference_line[1] = hazard_point
                    normal_from_haz_poly_reference_line_points_id[1] = hazard_point_id
            
            hazard_point_id += 1

        # Checking if two points were added for the along ship line direction. If three
        # or four points were added, proceed with the two furthest along the ship line
        # (note: only one point from each half-plane).
        num_f_points = 4
        for f_point in furthest_points_along_line:
            if f_point == None:
                num_f_points -= 1
            
        if num_f_points >= 3:
            furthest_points_along_line_temp = [None, None, None, None]
            along_haz_poly_reference_line_points_id_temp = [None, None, None, None]
            if furthest_distances_along_haz_poly_reference_line[0] > furthest_distances_along_haz_poly_reference_line[3]:
                furthest_points_along_line_temp[0] = furthest_points_along_line[0]
                along_haz_poly_reference_line_points_id_temp[0] = along_haz_poly_reference_line_points_id[0]
            else:
                furthest_points_along_line_temp[0] = furthest_points_along_line[3]
                along_haz_poly_reference_line_points_id_temp[0] = along_haz_poly_reference_line_points_id[3]

            if furthest_distances_along_haz_poly_reference_line[1] > furthest_distances_along_haz_poly_reference_line[2]:
                furthest_points_along_line_temp[1] = furthest_points_along_line[1]
                along_haz_poly_reference_line_points_id_temp[1] = along_haz_poly_reference_line_points_id[1]
            else:
                furthest_points_along_line_temp[1] = furthest_points_along_line[2]
                along_haz_poly_reference_line_points_id_temp[1] = along_haz_poly_reference_line_points_id[2]
            furthest_points_along_line = furthest_points_along_line_temp
            along_haz_poly_reference_line_points_id = along_haz_poly_reference_line_points_id_temp

        # Add correct hazard_points (furthest_points_along_line and normal_from_haz_poly_reference_line_points_id) 
        # and belonging ids to bookkeeping lists.
        num_f_p_a_l = 0
        for i in range(len(furthest_points_along_line)):
            if furthest_points_along_line[i] != None:
                furthest_points_along_line_multi_poly.append(furthest_points_along_line[i])
                along_haz_poly_reference_line_points_id_multi_poly.append(along_haz_poly_reference_line_points_id[i])
                i_f_p_a_l = i
                num_f_p_a_l += 1

        # The num. of f_p_a_l points for small polygons, which has an orientation such that they lie in only one quadrant or half-plane,
        # can be 1. Appending the same point twice to avoid errors (as two points are expected). 
        if num_f_p_a_l < 2:
            furthest_points_along_line_multi_poly.append(furthest_points_along_line[i_f_p_a_l])
            along_haz_poly_reference_line_points_id_multi_poly.append(along_haz_poly_reference_line_points_id[i_f_p_a_l])
        
        if furthest_distance_normal_from_haz_poly_reference_line[0] > furthest_distance_normal_from_haz_poly_reference_line[1]:
            furthest_points_normal_from_haz_poly_reference_line_multi_poly.append(furthest_points_normal_from_haz_poly_reference_line[0])
            normal_from_haz_poly_reference_line_points_id_multi_poly.append(normal_from_haz_poly_reference_line_points_id[0])
        else:
            furthest_points_normal_from_haz_poly_reference_line_multi_poly.append(furthest_points_normal_from_haz_poly_reference_line[1])
            normal_from_haz_poly_reference_line_points_id_multi_poly.append(normal_from_haz_poly_reference_line_points_id[1])

        hazard_poly_id += 1

        # Plotting functionality that shows the orientation of the coordinate system that is used
        # to find the points of interest on the hazardous polygon. The plot also shows the points
        # which has been found for the given hazard polygon.
        # if show_plots:
        #     enc.start_display()
        #     enc.draw_polygon(hazard_polygon, color = "red", alpha = 0.6)
        #     enc.draw_line(haz_poly_reference_line.coords, color = "black", width = 2)
        #     enc.draw_line(haz_poly_reference_line_normal.coords, color = "orange", width = 2)
        #     enc.draw_circle([ownship_point.x, ownship_point.y], 7, "cyan", thickness = 3, fill = True)
        #     for i in range(len(furthest_points_along_line)):
        #         if furthest_points_along_line[i] != None:
        #             enc.draw_circle([furthest_points_along_line[i].x, furthest_points_along_line[i].y], 7, "black", thickness = 3, fill = True)
        #     if furthest_distance_normal_from_haz_poly_reference_line[0] > furthest_distance_normal_from_haz_poly_reference_line[1]:
        #         enc.draw_circle(
        #             center = [furthest_points_normal_from_haz_poly_reference_line[0].x, furthest_points_normal_from_haz_poly_reference_line[0].y],
        #             radius = 7, color =  "brown", thickness = 3, fill = True
        #         )
        #     else:
        #         enc.draw_circle(
        #             center = [furthest_points_normal_from_haz_poly_reference_line[1].x, furthest_points_normal_from_haz_poly_reference_line[1].y],
        #             radius = 7, color = "brown", thickness = 3, fill = True
        #         )
        #     enc.show_display()

    # Appending the last hazard_point_id for the last polygon.
    last_hazard_point_id_multi_poly.append(hazard_point_id)

    # Removing unwanted vertices in the Polygons (in the MultiPolygons).
    hazard_poly_id = 0
    hazard_polygons_to_keep = []
    for hazard_polygon in filtered_visible_hazard_polygons.geoms:
        # Checking if it is the normal point or one of the ownship line points
        # which has the lowest vertex id (or either is equal).
        if ((normal_from_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id]
            <= along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2])
            and (normal_from_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id]
            <= along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1])):
            # Checking which along point has the lowest id.
            if (along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2] <
                along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1]):
                ids_between_along_points = [
                    along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2],
                    along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1]
                ]
            else:
                ids_between_along_points = [
                    along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1],
                    along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2]    
            ]
        else:
            # Checking which along point has the lowest id.
            if (along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2] <
                along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1]):
                # Checking if the second along point has a lower id than the normal point.
                if (along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1] < 
                    normal_from_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id]):
                    ids_between_along_points = [
                        along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2],
                        along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1]
                    ]
                else:
                    ids_between_along_points = [
                        0,
                        along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2],
                        along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1],
                        last_hazard_point_id_multi_poly[hazard_poly_id]
                    ]
            else:
                # Checking if the second along point has a lower id than the normal point.
                if (along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2] < 
                    normal_from_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id]):
                    ids_between_along_points = [
                        along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1],
                        along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2],
                    ]
                else:
                    ids_between_along_points = [
                        0,
                        along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1],
                        along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2],
                        last_hazard_point_id_multi_poly[hazard_poly_id]
                    ]

        # Keeping points of interest.
        hazard_point_id = 0
        hazard_points_to_keep = []
        for hazard_point_tuple in hazard_polygon.exterior.coords:
            keep_hazard_point = False
            if hazard_point_id == normal_from_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id]:
                keep_hazard_point = True
            elif hazard_point_id >=  ids_between_along_points[0] and hazard_point_id <= ids_between_along_points[1]:
                keep_hazard_point = True
            elif (len(ids_between_along_points) == 4 and 
                hazard_point_id >=  ids_between_along_points[2] and hazard_point_id <= ids_between_along_points[3]):
                keep_hazard_point = True

            if keep_hazard_point == True:
                hazard_point = Point(hazard_point_tuple)
                hazard_points_to_keep.append(hazard_point)
            hazard_point_id += 1
        
        # There must be atleast four points to make a Polygon.
        # If there are three points (two along points and one normal point) then
        # a new point right beside the normal point will be added.
        # The elif catches (small polygons located in one half-plane) that has had
        # to many vertices removed.
        if len(hazard_points_to_keep) == 3:
            hazard_points_to_keep.append(Point(
                furthest_points_normal_from_haz_poly_reference_line_multi_poly[hazard_poly_id].x + 0.01,
                furthest_points_normal_from_haz_poly_reference_line_multi_poly[hazard_poly_id].y + 0.01
            ))
            hazard_polygons_to_keep.append(Polygon(hazard_points_to_keep))
        elif len(hazard_points_to_keep) < 3:
            hazard_polygons_to_keep.append(Polygon(hazard_polygon))
        else:
            hazard_polygons_to_keep.append(Polygon(hazard_points_to_keep))
        hazard_poly_id += 1

    # Plotting functionality for all the final, processed, hazardous polygons, 
    # their points of interest and the ownship.  
    if show_plots:
        f_p_norm_oship_line_multi_poly = furthest_points_normal_from_haz_poly_reference_line_multi_poly
        enc.start_display()
        enc.draw_polygon(hazard_polygons_to_keep, color = "red", alpha = 0.6)
        for poly_id in range(hazard_poly_id):
            i = 0
            firstBlack = True
            for f_p_along_oship_line in furthest_points_along_line_multi_poly:
                if f_p_along_oship_line == None:
                    continue
                elif i >= 2 * (poly_id) and i < 2 * (poly_id + 1):
                    if firstBlack == True:
                        firstBlack = False
                        enc.draw_circle([f_p_along_oship_line.x, f_p_along_oship_line.y], 10, "black", thickness = 5, fill = True)
                    else:
                        firstBlack = True
                        enc.draw_circle([f_p_along_oship_line.x, f_p_along_oship_line.y], 10, "black", thickness = 5, fill = True)
                    enc.draw_line([f_p_norm_oship_line_multi_poly[poly_id], f_p_along_oship_line], color = "black", width = 1.5)
                i += 1
        for f_p_norm_oship_line in f_p_norm_oship_line_multi_poly:
            enc.draw_circle([f_p_norm_oship_line.x, f_p_norm_oship_line.y], 10, "brown", thickness = 5, fill = True)
        enc.add_vessels((0, int(ownship_x), int(ownship_y), int(ownship_cog), "black"))
        enc.show_display()
    
    # Correcting the return type.
    hazard_polygons_to_keep = [
        MultiPolygon(hazard_polygons_to_keep)
    ]

    return hazard_polygons_to_keep


def extract_grounding_hazards_from_entire_enc(
    vessel_min_depth: float, 
    d_from_land_min : float,
    enc: ENC
    ) -> list[MultiPolygon]:
    """Extracts all relevant grounding hazards from an entire ENC as a list of MultiPolygons.
    Hazards such as land areas (regular land, islands, islets, skerries), shores, shallow waters, 
    as well as areas in proximity to land areas.

    The minimum acceptable water depth for the given vessel is taken into account. 
    The minimum tolerated distance from the vessel to land/shore is also taken into account. 

    Args:
    - vessel_min_depth (float): The minimum depth required for the vessel to avoid grounding.
    - d_from_land_min (float): The minimum acceptable length from ship to land/shore.
    - enc (senc.ENC): The ENC to check for grounding hazards.

    Returns:
    list[MultiPolygon]: All grounding hazards (as outlined above) for the entire ENC.
    """
    # Defining relevant geometries.
    land = enc.land.geometry
    shore = enc.shore.geometry
    seabed_0 = enc.seabed[0].geometry
    seabed_vessel_min_depth = enc.seabed[vessel_min_depth].geometry
    
    # Finding the dangerous seabed for the vessel based on its draught.
    dangerous_seabed = seabed_0.difference(seabed_vessel_min_depth)

    # Finding a buffer around the land and shore geometry.
    land_buffer = land.buffer(d_from_land_min)
    shore_buffer = shore.buffer(d_from_land_min)
    
    # Finding where the buffer intersects with seabed_0 and shore.
    land_seabed_0_intersection = land_buffer.intersection(seabed_0)
    land_shore_intersection = shore_buffer.intersection(seabed_0)

    # Combining the dangerous seabed with the intersection polygons.
    relevant_hazards = [
        dangerous_seabed.union(land_seabed_0_intersection).union(land).union(
        land_shore_intersection).union(shore)#.union(seabed_vessel_min_depth)
    ]

    filtered_relevant_hazards = []
    for hazard in relevant_hazards:
        if isinstance(hazard, Polygon):
            filtered_relevant_hazards.append(MultiPolygon([hazard]))
        else:
            filtered_relevant_hazards.append(
                MultiPolygon(Polygon(p.exterior) for p in hazard.geoms if isinstance(p, Polygon))
            )
        
    return filtered_relevant_hazards


def extract_grounding_hazards_from_relevant_sector_in_enc(
    grounding_hazards : list[MultiPolygon],
    ownship_state : np.ndarray, 
    radius_of_coverage : float,
    angle_of_coverage_behind : float, 
    enc: ENC, 
    show_plots: bool = False) -> list[MultiPolygon]:
    """Filters the grounding hazards by means of defining a relevant 
    grounding sector. Only hazards which are present in this sector
    are saved. The relevant grounding sector is defined by the vessel 
    position, vessel cog, chosen angle_of_coverage and chosen radius_of_coverage.
    The sector uses the vessel position as its center, and must thus be updated in
    due time, if the vessel position changes.

    Args:
    - grounding_hazards (list[MultiPolygon]): Grounding hazards from an ENC parameterized as polygons. 
    - ownship_state (np.ndarray): The ownship state [x, y, psi, u, v, r].
    - angle_of_coverage_behind (float): The angle of which to concider grounding hazards in the
    opposite direction of cog, in degrees.
    - radius_of_coverage (float): The radius of which to concider grounding hazards, the radius of the
    relevant grounding sector. 
    - enc (senc.ENC): The ENC to check for grounding hazards.
    - show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
    list[MultiPolygon]: The grounding hazards inside the relevant grounding sector.
    """
    # Defining a relevant sector.
    ownship_x = ownship_state[0]
    ownship_y = ownship_state[1]
    ownship_cog = ownship_state[2]
    ownship_cog_rad = math.radians(ownship_cog)
    angle_of_coverage_behind_rad = math.radians(90 - angle_of_coverage_behind)

    # Defining a ship point.
    ownship_point = Point(ownship_x, ownship_y)

    # Defining the vertices of a triangle.
    coverage_neg_point = Point(
        ownship_x - radius_of_coverage * math.sin(ownship_cog_rad - angle_of_coverage_behind_rad),
        ownship_y - radius_of_coverage * math.cos(ownship_cog_rad - angle_of_coverage_behind_rad)
    )
    coverage_positive_point = Point(
        ownship_x - radius_of_coverage * math.sin(ownship_cog_rad + angle_of_coverage_behind_rad),
        ownship_y - radius_of_coverage * math.cos(ownship_cog_rad + angle_of_coverage_behind_rad)
    )

    # Using the triangle to define a circle, which is our area of interest (relevant grounding sector).
    center_relevant_sector = Point(
        (coverage_positive_point.x + coverage_neg_point.x) / 2, 
        (coverage_positive_point.y + coverage_neg_point.y) / 2
    )
    radius_relevant_sector = coverage_neg_point.distance(coverage_positive_point)/2
    circle_relevant_sector = Point(
        center_relevant_sector.x + radius_relevant_sector * math.sin(ownship_cog_rad), 
        center_relevant_sector.y + radius_relevant_sector * math.cos(ownship_cog_rad)
    ).buffer(radius_relevant_sector, resolution = 2) # Resolution gives the number of vertices in the "circle".
    
    # Rotate the area of interest to be aligned with the ships COG angle.
    circle_relevant_sector = affinity.rotate(circle_relevant_sector, -ownship_cog, "center")
    
    # Finding the intersection of relevant hazards and relevant area.
    filtered_relevant_hazards_circle_relevant_sector_intersection = circle_relevant_sector.intersection(
        grounding_hazards[0]
    )

    # Finding the center of the relevant region.
    rel_sec_center_x = center_relevant_sector.x + radius_relevant_sector * math.sin(ownship_cog_rad)
    rel_sec_center_y = center_relevant_sector.y + radius_relevant_sector * math.cos(ownship_cog_rad)

    # Finding the coordinates of the endpoint at the "end" of the relevant region.
    rel_sec_behind_x = rel_sec_center_x - 10 * radius_of_coverage * math.sin(ownship_cog_rad)
    rel_sec_behind_y = rel_sec_center_y - 10 * radius_of_coverage * math.cos(ownship_cog_rad)

    # Finding the coordinates of the endpoint at the "front" of the relevant region.
    rel_sec_ahead_x = rel_sec_center_x + 10 * radius_of_coverage * math.sin(ownship_cog_rad)
    rel_sec_ahead_y = rel_sec_center_y + 10 * radius_of_coverage * math.cos(ownship_cog_rad)

    # Defining endpoints and the center point.
    rel_sec_center = Point(rel_sec_center_x, rel_sec_center_y)
    rel_sec_behind_endpoint = Point(rel_sec_behind_x, rel_sec_behind_y)
    rel_sec_ahead_endpoint = Point(rel_sec_ahead_x, rel_sec_ahead_y)

    # Plotting functionality that shows the "circle" of relevant static hazards, ownship, ownship line
    # and the hazardous polygons (before pruning of non desired points).
    if show_plots:
        sector_triangle = Polygon([ownship_point, coverage_neg_point, coverage_positive_point, ownship_point])
        enc.start_display()
        enc.draw_line([rel_sec_behind_endpoint, rel_sec_center, rel_sec_ahead_endpoint], color = "black", width = 3)
        enc.add_vessels((0, int(ownship_x), int(ownship_y), int(ownship_cog), "black"))
        enc.draw_polygon(sector_triangle, color = "green", alpha = 0.5)
        enc.draw_polygon(circle_relevant_sector, color = "orange", alpha = 0.5)
        enc.draw_cifiltered_relevant_hazards_circle_relevant_sector_intersectionrcle([rel_sec_center.x, rel_sec_center.y], 30, "magenta", thickness = 4.5, fill = True)
        enc.draw_polygon(filtered_relevant_hazards_circle_relevant_sector_intersection, color = "red", alpha = 0.6)
        enc.show_display()

    # If there is only one hazardous polygon the type will be of Polygon.
    # This needs to be changed to MultiPolygon to avoid errors in the
    # subsequent data processing.
    if type(filtered_relevant_hazards_circle_relevant_sector_intersection) != MultiPolygon:
        filtered_relevant_hazards_circle_relevant_sector_intersection = \
            MultiPolygon([filtered_relevant_hazards_circle_relevant_sector_intersection])

    # Removing polygons based on LOS.
    filtered_visible_hazard_polygons = []
    for hazard_polygon in filtered_relevant_hazards_circle_relevant_sector_intersection.geoms:
        # Finding the center of the hazard polygon that is checked.
        hazard_polygon_center_point = hazard_polygon.centroid

        # Defining a line from the ownship to the center of the hazard polygon.
        los_line_from_os_to_hazard_poly = LineString(
            [ownship_point, (hazard_polygon_center_point.x, hazard_polygon_center_point.y)]
        )

        # Checking if any other polygon in the MultiPolygon obstructs the LOS between
        # the ownship and the given hazard polygon.
        hazard_poly_is_obstructed = False
        for other_hazard_polygon in filtered_relevant_hazards_circle_relevant_sector_intersection.geoms:
            if hazard_polygon != other_hazard_polygon:
                if los_line_from_os_to_hazard_poly.intersects(other_hazard_polygon):
                    hazard_poly_is_obstructed = True
                    break
                
        # If the LOS is not obstructed then the hazard_polygon will kept for further processing.
        if not hazard_poly_is_obstructed:
            filtered_visible_hazard_polygons.append(hazard_polygon)

    # Creating a new MultiPolygon from the visible polygons.
    filtered_visible_hazard_polygons = MultiPolygon(filtered_visible_hazard_polygons)
    
    # Bookkeeping lists used to keep track of extreme points in both directions (along ship line + normal to ship line).
    furthest_points_along_line_multi_poly = []
    furthest_points_normal_from_haz_poly_reference_line_multi_poly = []

    along_haz_poly_reference_line_points_id_multi_poly = []
    normal_from_haz_poly_reference_line_points_id_multi_poly = []
    last_hazard_point_id_multi_poly = []

    hazard_poly_id = 0
    hazard_point_id = None
    
    # Finding extreme points for all the polygons.
    for hazard_polygon in filtered_visible_hazard_polygons.geoms:
        # Finding the center of the hazard polygon that is processed.
        hazard_polygon_center_point = hazard_polygon.centroid

        # Finding the y-axis based on the hazard_polygon_center_point.
        haz_poly_reference_line_normal = LineString([
            ownship_point, hazard_polygon_center_point
        ])

        direction_vector = [
            Point(haz_poly_reference_line_normal.coords[1]).x - Point(haz_poly_reference_line_normal.coords[0]).x,
            Point(haz_poly_reference_line_normal.coords[1]).y - Point(haz_poly_reference_line_normal.coords[0]).y
        ]

        # Finding the x-axis which is aligned with the hazard polygon.
        haz_poly_reference_line = LineString([
            Point(ownship_point.x - direction_vector[1] * 10, ownship_point.y + direction_vector[0] * 10),
            Point(ownship_point.x + direction_vector[1] * 10, ownship_point.y - direction_vector[0] * 10)
        ])

        # Finding the y-axis which is aligned with the hazard polygon.
        haz_poly_reference_line_normal = affinity.rotate(haz_poly_reference_line, 90, ownship_point)

        # Redefining for every polygon.
        furthest_points_along_line = [None, None, None, None]
        furthest_points_normal_from_haz_poly_reference_line = [None, None]

        furthest_distances_along_haz_poly_reference_line = [0, 0, 0, 0]
        furthest_distance_normal_from_haz_poly_reference_line = [0, 0]

        along_haz_poly_reference_line_points_id = [None, None, None, None]
        normal_from_haz_poly_reference_line_points_id = [None, None]
        
        # Appending the last hazard_point_id for each (but the last) polygon.
        if hazard_point_id is not None:
            last_hazard_point_id_multi_poly.append(hazard_point_id)
        hazard_point_id = 0
        
        for hazard_point_tuple in hazard_polygon.exterior.coords:
            hazard_point = Point(hazard_point_tuple)

            # Projecting the dangerous point onto the ownship line.
            projected_point_x = haz_poly_reference_line.interpolate(haz_poly_reference_line.project(hazard_point))

            # Projecting the dangerous point onto the normal of the ownship line.
            projected_point_y = haz_poly_reference_line_normal.interpolate(haz_poly_reference_line_normal.project(hazard_point))

            # Calculate the signed distances from the projected points to the origin.
            signed_distance_x = ownship_point.distance(projected_point_x) * \
                (1 if projected_point_x.x >= ownship_point.x else -1)
            signed_distance_y = ownship_point.distance(projected_point_y) * \
                (1 if projected_point_y.y >= ownship_point.y else -1)

            # Finding which quadrant each point belongs to (wrt. the rotated coordinate system). 
            # Origin in the center of the relevant sector.
            if signed_distance_x >= 0 and signed_distance_y >= 0:
                quadrant = 0
            elif signed_distance_x < 0 and signed_distance_y >= 0:
                quadrant = 1
            elif signed_distance_x < 0 and signed_distance_y < 0:
                quadrant = 2
            elif signed_distance_x >= 0 and signed_distance_y < 0:
                quadrant = 3

            # Using the unsigned distance for comparisons.
            distance_along_haz_poly_reference_line = abs(signed_distance_x)

            # Update the furthest distance and point along the line.
            if distance_along_haz_poly_reference_line > furthest_distances_along_haz_poly_reference_line[quadrant]:
                furthest_distances_along_haz_poly_reference_line[quadrant] = distance_along_haz_poly_reference_line
                furthest_points_along_line[quadrant] = hazard_point
                along_haz_poly_reference_line_points_id[quadrant] = hazard_point_id

            # Calculate the distance of the normal of the ownship line to the hazard point.
            distance_normal_from_haz_poly_reference_line = hazard_point.distance(projected_point_x)

            # Update the furthest distance and point normal to the line.
            if (quadrant == 0 or quadrant == 1):
                if (distance_normal_from_haz_poly_reference_line > furthest_distance_normal_from_haz_poly_reference_line[0]):
                    furthest_distance_normal_from_haz_poly_reference_line[0] = distance_normal_from_haz_poly_reference_line
                    furthest_points_normal_from_haz_poly_reference_line[0] = hazard_point
                    normal_from_haz_poly_reference_line_points_id[0] = hazard_point_id
            elif (quadrant == 2 or quadrant == 3):
                if (distance_normal_from_haz_poly_reference_line > furthest_distance_normal_from_haz_poly_reference_line[1]):
                    furthest_distance_normal_from_haz_poly_reference_line[1] = distance_normal_from_haz_poly_reference_line
                    furthest_points_normal_from_haz_poly_reference_line[1] = hazard_point
                    normal_from_haz_poly_reference_line_points_id[1] = hazard_point_id
            
            hazard_point_id += 1

        # Checking if two points were added for the along ship line direction. If three
        # or four points were added, proceed with the two furthest along the ship line
        # (note: only one point from each half-plane).
        num_f_points = 4
        for f_point in furthest_points_along_line:
            if f_point == None:
                num_f_points -= 1
            
        if num_f_points >= 3:
            furthest_points_along_line_temp = [None, None, None, None]
            along_haz_poly_reference_line_points_id_temp = [None, None, None, None]
            if furthest_distances_along_haz_poly_reference_line[0] > furthest_distances_along_haz_poly_reference_line[3]:
                furthest_points_along_line_temp[0] = furthest_points_along_line[0]
                along_haz_poly_reference_line_points_id_temp[0] = along_haz_poly_reference_line_points_id[0]
            else:
                furthest_points_along_line_temp[0] = furthest_points_along_line[3]
                along_haz_poly_reference_line_points_id_temp[0] = along_haz_poly_reference_line_points_id[3]

            if furthest_distances_along_haz_poly_reference_line[1] > furthest_distances_along_haz_poly_reference_line[2]:
                furthest_points_along_line_temp[1] = furthest_points_along_line[1]
                along_haz_poly_reference_line_points_id_temp[1] = along_haz_poly_reference_line_points_id[1]
            else:
                furthest_points_along_line_temp[1] = furthest_points_along_line[2]
                along_haz_poly_reference_line_points_id_temp[1] = along_haz_poly_reference_line_points_id[2]
            furthest_points_along_line = furthest_points_along_line_temp
            along_haz_poly_reference_line_points_id = along_haz_poly_reference_line_points_id_temp

        # Add correct hazard_points (furthest_points_along_line and normal_from_haz_poly_reference_line_points_id) 
        # and belonging ids to bookkeeping lists.
        num_f_p_a_l = 0
        for i in range(len(furthest_points_along_line)):
            if furthest_points_along_line[i] != None:
                furthest_points_along_line_multi_poly.append(furthest_points_along_line[i])
                along_haz_poly_reference_line_points_id_multi_poly.append(along_haz_poly_reference_line_points_id[i])
                i_f_p_a_l = i
                num_f_p_a_l += 1

        # The num. of f_p_a_l points for small polygons, which has an orientation such that they lie in only one quadrant or half-plane,
        # can be 1. Appending the same point twice to avoid errors (as two points are expected). 
        if num_f_p_a_l < 2:
            furthest_points_along_line_multi_poly.append(furthest_points_along_line[i_f_p_a_l])
            along_haz_poly_reference_line_points_id_multi_poly.append(along_haz_poly_reference_line_points_id[i_f_p_a_l])
        
        if furthest_distance_normal_from_haz_poly_reference_line[0] > furthest_distance_normal_from_haz_poly_reference_line[1]:
            furthest_points_normal_from_haz_poly_reference_line_multi_poly.append(furthest_points_normal_from_haz_poly_reference_line[0])
            normal_from_haz_poly_reference_line_points_id_multi_poly.append(normal_from_haz_poly_reference_line_points_id[0])
        else:
            furthest_points_normal_from_haz_poly_reference_line_multi_poly.append(furthest_points_normal_from_haz_poly_reference_line[1])
            normal_from_haz_poly_reference_line_points_id_multi_poly.append(normal_from_haz_poly_reference_line_points_id[1])

        hazard_poly_id += 1

        # Plotting functionality that shows the orientation of the coordinate system that is used
        # to find the points of interest on the hazardous polygon. The plot also shows the points
        # which has been found for the given hazard polygon.
        # if show_plots:
        #     enc.start_display()
        #     enc.draw_polygon(hazard_polygon, color = "red", alpha = 0.6)
        #     enc.draw_line(haz_poly_reference_line.coords, color = "black", width = 2)
        #     enc.draw_line(haz_poly_reference_line_normal.coords, color = "orange", width = 2)
        #     enc.draw_circle([ownship_point.x, ownship_point.y], 7, "cyan", thickness = 3, fill = True)
        #     for i in range(len(furthest_points_along_line)):
        #         if furthest_points_along_line[i] != None:
        #             enc.draw_circle([furthest_points_along_line[i].x, furthest_points_along_line[i].y], 7, "black", thickness = 3, fill = True)
        #     if furthest_distance_normal_from_haz_poly_reference_line[0] > furthest_distance_normal_from_haz_poly_reference_line[1]:
        #         enc.draw_circle(
        #             center = [furthest_points_normal_from_haz_poly_reference_line[0].x, furthest_points_normal_from_haz_poly_reference_line[0].y],
        #             radius = 7, color =  "brown", thickness = 3, fill = True
        #         )
        #     else:
        #         enc.draw_circle(
        #             center = [furthest_points_normal_from_haz_poly_reference_line[1].x, furthest_points_normal_from_haz_poly_reference_line[1].y],
        #             radius = 7, color = "brown", thickness = 3, fill = True
        #         )
        #     enc.show_display()

    # Appending the last hazard_point_id for the last polygon.
    last_hazard_point_id_multi_poly.append(hazard_point_id)

    # Removing unwanted vertices in the Polygons (in the MultiPolygons).
    hazard_poly_id = 0
    hazard_polygons_to_keep = []
    for hazard_polygon in filtered_visible_hazard_polygons.geoms:
        # Checking if it is the normal point or one of the ownship line points
        # which has the lowest vertex id (or either is equal).
        if ((normal_from_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id]
            <= along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2])
            and (normal_from_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id]
            <= along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1])):
            # Checking which along point has the lowest id.
            if (along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2] <
                along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1]):
                ids_between_along_points = [
                    along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2],
                    along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1]
                ]
            else:
                ids_between_along_points = [
                    along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1],
                    along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2]    
            ]
        else:
            # Checking which along point has the lowest id.
            if (along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2] <
                along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1]):
                # Checking if the second along point has a lower id than the normal point.
                if (along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1] < 
                    normal_from_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id]):
                    ids_between_along_points = [
                        along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2],
                        along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1]
                    ]
                else:
                    ids_between_along_points = [
                        0,
                        along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2],
                        along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1],
                        last_hazard_point_id_multi_poly[hazard_poly_id]
                    ]
            else:
                # Checking if the second along point has a lower id than the normal point.
                if (along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2] < 
                    normal_from_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id]):
                    ids_between_along_points = [
                        along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1],
                        along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2],
                    ]
                else:
                    ids_between_along_points = [
                        0,
                        along_haz_poly_reference_line_points_id_multi_poly[(1 + hazard_poly_id) * 2 - 1],
                        along_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id * 2],
                        last_hazard_point_id_multi_poly[hazard_poly_id]
                    ]

        # Keeping points of interest.
        hazard_point_id = 0
        hazard_points_to_keep = []
        for hazard_point_tuple in hazard_polygon.exterior.coords:
            keep_hazard_point = False
            if hazard_point_id == normal_from_haz_poly_reference_line_points_id_multi_poly[hazard_poly_id]:
                keep_hazard_point = True
            elif hazard_point_id >=  ids_between_along_points[0] and hazard_point_id <= ids_between_along_points[1]:
                keep_hazard_point = True
            elif (len(ids_between_along_points) == 4 and 
                hazard_point_id >=  ids_between_along_points[2] and hazard_point_id <= ids_between_along_points[3]):
                keep_hazard_point = True

            if keep_hazard_point == True:
                hazard_point = Point(hazard_point_tuple)
                hazard_points_to_keep.append(hazard_point)
            hazard_point_id += 1
        
        # There must be atleast four points to make a Polygon.
        # If there are three points (two along points and one normal point) then
        # a new point right beside the normal point will be added.
        # The elif catches (small polygons located in one half-plane) that has had
        # to many vertices removed.
        if len(hazard_points_to_keep) == 3:
            hazard_points_to_keep.append(Point(
                furthest_points_normal_from_haz_poly_reference_line_multi_poly[hazard_poly_id].x + 0.01,
                furthest_points_normal_from_haz_poly_reference_line_multi_poly[hazard_poly_id].y + 0.01
            ))
            hazard_polygons_to_keep.append(Polygon(hazard_points_to_keep))
        elif len(hazard_points_to_keep) < 3:
            hazard_polygons_to_keep.append(Polygon(hazard_polygon))
        else:
            hazard_polygons_to_keep.append(Polygon(hazard_points_to_keep))
        hazard_poly_id += 1

    # Plotting functionality for all the final, processed, hazardous polygons, 
    # their points of interest and the ownship.  
    if show_plots:
        f_p_norm_oship_line_multi_poly = furthest_points_normal_from_haz_poly_reference_line_multi_poly
        enc.start_display()
        if hazard_polygons_to_keep != []:
            enc.draw_polygon(hazard_polygons_to_keep, color = "red", alpha = 0.6)
        for poly_id in range(hazard_poly_id):
            i = 0
            firstBlack = True
            for f_p_along_oship_line in furthest_points_along_line_multi_poly:
                if f_p_along_oship_line == None:
                    continue
                elif i >= 2 * (poly_id) and i < 2 * (poly_id + 1):
                    if firstBlack == True:
                        firstBlack = False
                        enc.draw_circle([f_p_along_oship_line.x, f_p_along_oship_line.y], 10, "black", thickness = 5, fill = True)
                    else:
                        firstBlack = True
                        enc.draw_circle([f_p_along_oship_line.x, f_p_along_oship_line.y], 10, "black", thickness = 5, fill = True)
                    enc.draw_line([f_p_norm_oship_line_multi_poly[poly_id], f_p_along_oship_line], color = "black", width = 1.5)
                i += 1
        for f_p_norm_oship_line in f_p_norm_oship_line_multi_poly:
            enc.draw_circle([f_p_norm_oship_line.x, f_p_norm_oship_line.y], 10, "brown", thickness = 5, fill = True)
        enc.add_vessels((0, int(ownship_x), int(ownship_y), int(ownship_cog), "black"))
        enc.show_display()
    
    # Correcting the return type.
    hazard_polygons_to_keep = [
        MultiPolygon(hazard_polygons_to_keep)
    ]

    return hazard_polygons_to_keep


def fill_rtree_with_geometries(geometries: list) -> Tuple[strtree.STRtree, list]:
    """Fills an rtree with the given multipolygon geometries. Used for fast spatial queries.

    Args:
        - geometries (list): The geometries to fill the rtree with.

    Returns:
        Tuple[strtree.STRtree, list]: The rtree containing the geometries, and the Polygon objects used to build it.
    """
    poly_list = []
    for poly in geometries:
        assert isinstance(poly, MultiPolygon), "Only MultiPolygon members are supported"
        for sub_poly in poly.geoms:
            poly_list.append(sub_poly)
    return strtree.STRtree(poly_list), poly_list


def generate_random_goal_position(
    rng: np.random.Generator,
    enc: ENC,
    xs_start: np.ndarray,
    safe_sea_cdt: list,
    safe_sea_cdt_weights: list,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    min_distance_from_start: float = 300.0,
    max_distance_from_start: float = 10000.0,
    sector_width: float = 60.0 * np.pi / 180.0,
    min_distance_to_land: float = 50.0,
    show_plots: bool = False,
) -> Tuple[float, float]:
    """Generates a random goal position for the ship, given its starting state (position, speed and heading).

    Args:
        rng (np.random.Generator): Numpy random generator.
        enc (ENC): Electronic Navigational Chart object.
        xs_start (np.ndarray): Starting CSOG state of the ship [x, y, U, chi]^T.
        safe_sea_cdt (list): List of triangles defining the safe sea region, used to sample more efficiently.
        safe_sea_cdt_weights (list): List of weights for the safe sea region triangles, used to sample more efficiently.
        min_distance_from_start (float, optional): Minimum distance from the starting position. Defaults to 300.0.
        max_distance_from_start (float, optional): Maximum distance from the starting position. Defaults to 10000.0.
        sector_width (float, optional): Width of the sector to sample from. Defaults to 60.0 * np.pi / 180.0.
        min_distance_to_land (float, optional): Minimum distance to land. Defaults to 50.0.
        show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        Tuple[float, float]: Goal position (northing, easting) for the ship.
    """
    if bbox is None:
        bbox = enc.bbox
    bbox_poly = bbox_to_polygon(bbox)

    if max_distance_from_start <= min_distance_from_start:
        print(
            "WARNING: Max_distance_from_start must be larger than min_distance_from_start in goal position sampling. Setting to default values.."
        )
        max_distance_from_start = min_distance_from_start + 500.0

    northing = xs_start[0] + max_distance_from_start * np.cos(xs_start[3])
    easting = xs_start[1] + max_distance_from_start * np.sin(xs_start[3])
    sector_radius = max(max_distance_from_start, min_distance_from_start)
    n_points = 100
    angle_range = np.linspace(-sector_width / 2.0 + xs_start[3], sector_width / 2.0 + xs_start[3], n_points)
    arc = [
        (xs_start[1] + sector_radius * np.sin(angle), xs_start[0] + sector_radius * np.cos(angle))
        for angle in angle_range
    ]
    arc_linestring = LineString(arc)
    sector_poly = Polygon(list(arc_linestring.coords) + [(xs_start[1], xs_start[0])])
    sector_poly = sector_poly.intersection(bbox_poly)
    if show_plots:
        enc.start_display()
        enc.draw_polygon(sector_poly, color="green", fill=True, alpha=0.2)
    max_iter = 3000
    for it in range(max_iter):
        p = mhm.sample_from_triangulation(rng, safe_sea_cdt, safe_sea_cdt_weights)
        easting, northing = p[0], p[1]

        dist2start = np.linalg.norm(np.array([northing, easting]) - np.array([xs_start[0], xs_start[1]]))
        inside_sector = sector_poly.contains(Point(easting, northing))
        dist2land = enc.land.geometry.distance(Point(easting, northing))
        if (
            (min_distance_from_start <= dist2start <= max_distance_from_start)
            and inside_sector
            and (dist2land >= min_distance_to_land)
        ):
            break

        if it == max_iter - 1:
            print("WARNING: No goal position that satisfies the constraints found. Returning a random position...")

    return northing, easting


def generate_random_position_from_draft(
    rng: np.random.Generator,
    enc: ENC,
    draft: float,
    safe_sea_cdt: Optional[list] = None,
    safe_sea_cdt_weights: Optional[list] = None,
    min_land_clearance: float = 50.0,
) -> Tuple[float, float]:
    """
    Randomly defining easting and northing coordinates of a ship
    inside the safe sea region by considering a ship draft, with an optional land clearance distance.

    Args:
        - rng (np.random.Generator): Numpy random generator.
        - enc (ENC): Electronic Navigational Chart object
        - draft (float): Ship's draft in meters.
        - safe_sea_cdt (Optional[list]): List of triangles defining the safe sea region, used to sample more efficiently. Defaults to None.
        - safe_sea_cdt_weights (Optional[list]): List of weights for the safe sea region triangles, used to sample more efficiently. Defaults to None.

    Returns:
        - Tuple[float, float]: Tuple of starting x and y coordinates for the ship.
    """
    depth = find_minimum_depth(draft, enc)
    safe_sea = enc.seabed[depth]
    bbox = enc.bbox

    max_iter = 1000
    northing = enc.bbox[1] + 0.5 * (enc.bbox[3] - enc.bbox[1])
    easting = enc.bbox[0] + 0.5 * (enc.bbox[2] - enc.bbox[0])
    for i in range(max_iter):
        if safe_sea_cdt is not None:
            p = mhm.sample_from_triangulation(rng, safe_sea_cdt, safe_sea_cdt_weights)
            easting, northing = p[0], p[1]
        else:
            easting, northing = rng.uniform(bbox[0], bbox[2]), rng.uniform(bbox[1], bbox[3])

        inside_bbox = mhm.inside_bbox(np.array([northing, easting]), (bbox[1], bbox[0], bbox[3], bbox[2]))
        d2land = enc.land.geometry.distance(Point(easting, northing))
        if safe_sea.geometry.contains(Point(easting, northing)) and inside_bbox and d2land >= min_land_clearance:
            break

    return northing, easting


def find_closest_collision_free_point_on_segment(
    enc: ENC, p1: np.ndarray, p2: np.ndarray, draft: float = 5.0, hazards: Optional[list] = None, min_dist: float = 30.0
) -> np.ndarray:
    """Finds the closest collision free point on a line segment between two points.

    Args:
        enc (ENC): Electronic Navigational Chart object
        p1 (np.ndarray): First position [x1, y1]^T. x = north, y = east.
        p2 (np.ndarray): Second position.
        draft (float, optional): Vessel draft. Defaults to 5.0.
        hazards (Optional[list], optional): List of Multipolygon/Polygon objects that are relevant. Used if not none. Defaults to None.
        min_dist (float, optional): Minimum distance to the hazard. Defaults to 5.0.

    Returns:
        np.ndarray: The closest collision free point on the line segment.
    """
    assert p1.shape == (2,) and p2.shape == (2,), "p1 and p2 must be 2D vectors"
    segment = LineString([(p1[1], p1[0]), (p2[1], p2[0])])
    if hazards is None:
        hazards = extract_relevant_grounding_hazards_as_union(find_minimum_depth(draft, enc), enc, buffer=min_dist)

    for hazard in hazards:
        if hazard.is_empty:
            continue

        hazard = hazard.buffer(min_dist)

        # enc.draw_polygon(hazard, color="orange", fill=False)

        if segment.intersects(hazard):
            intersection = segment.intersection(hazard)
            nearest_point = ops.nearest_points(Point(p1[1], p1[0]), intersection)[1]
            # enc.draw_circle((nearest_point.x, nearest_point.y), radius=1.0, color="yellow", fill=False)
            return np.array([nearest_point.y, nearest_point.x])
    return p2


def compute_distance_vectors_to_grounding(
    vessel_trajectory: np.ndarray,
    min_vessel_depth: int,
    enc: ENC,
    disable_bbox_check: bool = False,
    show_plots: bool = False,
) -> np.ndarray:
    """Computes the distance vectors to grounding at each step of the given vessel trajectory or point
    if n_samples = 1.

    Args:
        - vessel_trajectory (np.ndarray): The vessels trajectory, 2 x n_samples.
        - min_vessel_depth (int): The minimum depth required for the vessel to avoid grounding.
        - vessel_trajectory (np.ndarray): The vessel's trajectory, 2 x n_samples.
        - minimum_vessel_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enc (ENC): The ENC to check for grounding.
        - disable_bbox_check (bool, optional): Option for disabling the inside bounding box check for a position. Defaults to False.
        - show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        - np.ndarray: The distance to grounding at each step of the vessel trajectory.
    """
    n_samples = vessel_trajectory.shape[1]
    x_min, y_min, x_max, y_max = enc.bbox
    bbox_poly = bbox_to_polygon((float(x_min), float(y_min), float(x_max), float(y_max)))
    if show_plots:
        enc.start_display()
    relevant_hazards = extract_relevant_grounding_hazards_as_union(min_vessel_depth, enc)
    distance_vectors = np.ndarray((2, vessel_trajectory.shape[1]))
    for idx in range(n_samples):
        point = Point(vessel_trajectory[0, idx], vessel_trajectory[1, idx])
        for hazard in relevant_hazards:
            if not bbox_poly.contains(point) and not disable_bbox_check:
                distance_vectors[:, idx] = np.array([0.0, 0.0])
                break
            nearest_poly_points = []
            for hazard in relevant_hazards:
                nearest_point = ops.nearest_points(point, hazard)[1]
                nearest_poly_points.append(nearest_point)

            min_dist = 1e12
            min_dist_vec = np.array([1e6, 1e6])
            for _, near_point in enumerate(nearest_poly_points):
                points = [
                    (np.asarray(point.coords.xy[0])[0], np.asarray(point.coords.xy[1])[0]),
                    (np.asarray(near_point.coords.xy[0])[0], np.asarray(near_point.coords.xy[1])[0]),
                ]

                if show_plots:
                    enc.draw_line(points, color="black", linewidth=0.5, marker_type="o")

                dist_vec = np.array([points[1][0] - points[0][0], points[1][1] - points[0][1]])
                if np.linalg.norm(dist_vec) <= min_dist:
                    min_dist_vec = dist_vec
                    min_dist = np.linalg.norm(min_dist_vec)
            distance_vectors[:, idx] = min_dist_vec
    if show_plots:
        enc.show_display()
    return distance_vectors


def calc_nominal_and_actual_OS_traj_dist(
        os_trajectory: np.ndarray,
        os_initial_pos: np.ndarray,
        os_waypoints: np.ndarray    
    ) -> Tuple[float, float]:
    """Calculates the nominal and actual OS trajectory distances (euclidean).
    
    Args:
        os_trajectory (np.ndarray): Ownship trajectory in NE.
        os_initial_pos (np.ndarray): Initial position of the OS in NE.
        os_waypoints (np.ndarray): Waypoints for the OS in NE.

    Returns:
        Tuple[float, float]: Nominal and actual trajectory distances for the OS.
    """
    actual_traj_dist = np.sum(np.linalg.norm(np.diff(os_trajectory, axis = 1), axis = 0))

    # The distance between the initial state and the initial waypoint
    nominal_traj_dist = np.linalg.norm(os_initial_pos - os_waypoints[:, 0].reshape(-1, 1))

    # The distances between consecutive waypoints
    nominal_traj_dist += np.sum(np.linalg.norm(np.diff(os_waypoints, axis = 1), axis = 0))
    np.set_printoptions(threshold=np.inf)
    return nominal_traj_dist, actual_traj_dist


def get_distance_vectors_to_obstacles(
    trajectory: np.ndarray,
    do_list: list,
    enc: ENC,
    T: float,
    dt: float,
    min_vessel_depth: int = 5,
    disable_bbox_check: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the distance vectors from the trajectory to the obstacles (dynamic and static).

    Args:
        trajectory (np.ndarray): Trajectory/position data (minimum 2 x n_samples).
        do_list (list): List of dynamic obstacles on the form (ID, state, cov, length, width)
        enc (senc.ENC): ENC object.
        T (float): Prediction horizon.
        dt (float): Time step.
        min_vessel_depth (int, optional): Minimum vessel depth. Defaults to 5.
        disable_bbox_check (bool, optional): Option for disabling the inside bounding box check for a position. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of distance vectors to dynamic obstacles and list of distance vectors to static obstacles.
    """
    distance_vectors_so = compute_distance_vectors_to_grounding(trajectory, min_vessel_depth, enc, disable_bbox_check)
    distance_vectors_do = compute_distance_vectors_to_dynamic_obstacles(trajectory, do_list, T, dt)
    return distance_vectors_do, distance_vectors_so


def compute_minimum_distance_to_collision_and_grounding(
    trajectory: np.ndarray,
    do_list: list,
    enc: ENC,
    T: float,
    dt: float,
    min_vessel_depth: int = 5,
    disable_bbox_check: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Check if the trajectory collides with any of the obstacles (dynamic and static) over the prediction horizon.

    Args:
        trajectory (np.ndarray): Trajectory/position data (minimum 2 x n_samples) with EN coordinates
        do_list (list): List of dynamic obstacles on the form (ID, state, cov, length, width) with EN coordinates
        enc (ENC): ENC object.
        T (float): Prediction horizon.
        dt (float): Time step.
        min_vessel_depth (int, optional): Minimum allowable vessel depth. Defaults to 5.
        disable_bbox_check (bool, optional): Option for disabling the inside bounding box check for a position. Defaults to False.

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: The minimum distances to collision and grounding, respectively. Also returns the corresponding distance vectors
    """
    distance_vectors_do, distance_vectors_so = get_distance_vectors_to_obstacles(
        trajectory, do_list, enc, T, dt, min_vessel_depth, disable_bbox_check
    )
    min_dist_so = 1e12
    if distance_vectors_so.size > 0:
        min_dist_so = np.min(np.linalg.norm(distance_vectors_so, axis=0))
    min_dist_do = 1e12
    if distance_vectors_do.size > 0:
        min_dist_do = np.min(np.linalg.norm(distance_vectors_do, axis=0))
    return min_dist_do, min_dist_so, distance_vectors_do, distance_vectors_so


def compute_distance_vectors_to_dynamic_obstacles(
    trajectory: np.ndarray, do_list: list, T: float, dt: float
) -> np.ndarray:
    """Computes the (shortest) distance vectors to dynamic obstacles, assuming EN coordinates.

    Args:
        trajectory (np.ndarray): Trajectory/position data (minimum 2 x n_samples) with EN coordinates
        do_list (list): List of dynamic obstacles on the form (ID, state, cov, length, width) with EN coordinates
        T (float): Prediction horizon.
        dt (float): Time step.

    Returns:
        np.ndarray: (Shortest) Distance vectors to dynamic obstacles.
    """
    if len(do_list) == 0:
        return np.empty(0)
    n_samples = trajectory.shape[1]
    assert n_samples > 1, "Trajectory must have at least two samples"
    assert n_samples == int(T / dt), "Must have n_samples = int(T / dt)"
    distance_vectors = np.ndarray((2, n_samples))
    for k in range(n_samples):
        t = k * dt
        p_k = trajectory[:, k]
        min_do_dist_vec = np.array([1e6, 1e6])
        min_do_dist = 1e12
        for ID, do_state, do_cov, do_length, do_width in do_list:
            p_do_k = do_state[:2] + np.array([do_state[2], do_state[3]]) * t
            dist_vec = p_do_k - p_k
            if np.linalg.norm(dist_vec) < min_do_dist:
                min_do_dist = np.linalg.norm(dist_vec)
                min_do_dist_vec = dist_vec
        distance_vectors[:, k] = min_do_dist_vec
    return distance_vectors


def compute_distance_vectors_to_dynamic_obstacles_from_trajs(
    os_trajectory: np.ndarray, 
    do_trajectories: list,
    show_plots: bool = False,
    enc: ENC = None 
) -> np.ndarray:
    """Computes the (shortest) distance vectors to dynamic obstacles, assuming EN coordinates.
    Admits trajectories of equal length.

    Args:
        os_trajectory (np.ndarray): Ownship trajectory.
        do_trajectories (list): List of dynamic obstacle trajectories.

    Returns:
        np.ndarray: Distance vectors to the closest dynamic obstacle for all steps.
    """
    if len(do_trajectories) == 0:
        return np.empty(0)
    n_samples = os_trajectory.shape[1]
    assert n_samples > 1, "Trajectory must have at least two samples"
    distance_vectors = np.ndarray((2, n_samples))
    if show_plots:
        enc.start_display()
    for k in range(n_samples):
        p_os_k = os_trajectory[:, k]
        min_do_dist_vec_k = np.array([1e6, 1e6])
        min_do_dist_k = 1e12

        for do_trajectory in do_trajectories:
            p_do_k = do_trajectory[:, k]
            dist_vec_k = p_do_k - p_os_k
            if np.linalg.norm(dist_vec_k) < min_do_dist_k:
                min_do_dist_k = np.linalg.norm(dist_vec_k)
                min_do_dist_vec_k = dist_vec_k

        distance_vectors[:, k] = min_do_dist_vec_k
        if show_plots: # and k%40 == 0:
            points = [
                    (os_trajectory[:, k]),
                    (os_trajectory[:, k] + min_do_dist_vec_k)
                ]
            enc.draw_line(points, color = "black", linewidth = 0.5, marker_type = "o")
            #enc.draw_line(points, color = "blue", linewidth = 0.5, marker_type = "o")
            #enc.draw_circle((p_do_k), color = "red", fill = True, radius = 5)
            #enc.draw_circle((p_os_k), color = "green", fill = True, radius = 5)
    if show_plots:
        enc.show_display()
    return distance_vectors


def compute_distance_vector_to_bbox(
    x: float, y: float, bbox: Tuple[float, float, float, float], enc: Optional[ENC] = None
) -> np.ndarray:
    """Computes the distance vector to the closest point on the bounding box.

    Args:
        x (float): Easting coordinate.
        y (float): Northing coordinate.
        bbox (Tuple[float, float, float, float]): Bounding box (xmin, ymin, xmax, ymax).
        - enc (senc.ENC): The ENC to check for grounding.

    Returns:
        np.ndarray: Distance vector to the closest point on the bounding box.
    """
    south_line = LineString([(bbox[0], bbox[1]), (bbox[2], bbox[1])])
    east_line = LineString([(bbox[2], bbox[1]), (bbox[2], bbox[3])])
    north_line = LineString([(bbox[2], bbox[3]), (bbox[0], bbox[3])])
    west_line = LineString([(bbox[0], bbox[3]), (bbox[0], bbox[1])])
    lines = [north_line, east_line, south_line, west_line]
    min_dist = 1e12
    distance_vector = np.array([1e6, 1e6])
    # if enc is not None:
    #     enc.start_display()

    for line in lines:
        d2line = line.distance(Point(x, y))
        line_point = ops.nearest_points(Point(x, y), line)[1]
        if d2line < min_dist:
            min_dist = d2line
            distance_vector = np.array([line_point.x - x, line_point.y - y])
        # if enc is not None:
        #     enc.draw_line([(x, y), (line_point.x, line_point.y)], color="red")
        #     enc.draw_circle((line_point.x, line_point.y), radius=0.5, color="red")
        #     enc.draw_circle((x, y), radius=0.5, color="blue")

    return distance_vector


def min_distance_to_hazards(hazards: list, x: float, y: float) -> float:
    """Compute the minimum distance to hazards from a given point.

    Args:
        hazards (list): List of Multipolygon/Polygon objects that are relevant
        x (float): Ship's easting coordinate
        y (float): Ship's northing coordinate

    Returns:
        float: Minimum distance to hazards in meters.
    """
    min_dist = 1e12
    if isinstance(hazards, Polygon):
        hazards = [hazards]
    for hazard in hazards:
        if hazard.is_empty:
            continue

        dist = hazard.distance(Point(x, y))
        if dist < min_dist:
            min_dist = dist
    return min_dist


def min_distance_and_point_to_hazards(hazards: list, x: float, y: float) -> list[float, Point]:
    """Compute the minimum distance to hazards from a given point.

    Args:
        hazards (list): List of Multipolygon/Polygon objects that are relevant
        x (float): Ship's easting coordinate
        y (float): Ship's northing coordinate

    Returns:
        float: Minimum distance to hazards in meters.
        Point: The point on the hazards which is the closest to Point(x, y).
    """
    min_dist = 1e12
    if isinstance(hazards, Polygon):
        hazards = [hazards]
    closest_point_on_hazard = None
    ship_point = Point(x, y)
    for hazard in hazards:
        if hazard.is_empty:
            continue

        dist = ship_point.distance(hazard)
        if dist < min_dist:
            min_dist = dist
            closest_point_on_hazard = ops.nearest_points(ship_point, hazard)[1]
    return [min_dist, closest_point_on_hazard]


def check_if_segment_crosses_grounding_hazards(
    enc: ENC, p1: np.ndarray, p2: np.ndarray, draft: float = 5.0, hazards: Optional[list] = None
) -> bool:
    """Checks if a line segment between two positions/points crosses nearby grounding hazards (land, shore).

    Args:
        enc (ENC): Electronic Navigational Chart object
        p1 (np.ndarray): First position [x1, y1]^T. x = north, y = east.
        p2 (np.ndarray): Second position.
        draft (float): Ship's draft in meters.¨
        hazards (Optional[list]): List of Multipolygon/Polygon objects that are relevant. Used if not none. Defaults to None.

    Returns:
        bool: True if path segment crosses land, False otherwise.

    """
    # Create linestring with east as the x value and north as the y value
    p2_reverse = (p2[1], p2[0])
    p1_reverse = (p1[1], p1[0])
    wp_line = LineString([p1_reverse, p2_reverse])

    min_depth = find_minimum_depth(draft, enc)

    if hazards is None:
        hazards = extract_relevant_grounding_hazards_as_union(min_depth, enc)

    for hazard in hazards:
        if hazard.is_empty:
            continue

        intersects_hazards = wp_line.intersects(hazard)
        if intersects_hazards:
            return True

    return False


def generate_ship_sector_polygons(
    pos_x: float, pos_y: float, chi: float, safety_radius: float
) -> Tuple[Polygon, Polygon, Polygon]:
    """Generates sector polygons for the ship portside, front and starboardside.

    Args:
        pos_x (float): X position of ship
        pos_y (float): Y position of ship
        chi (float): Course over ground of ship
        safety_radius (int): radius for collision polygons on port, front and starboard of ship

    Returns:
        Tuple[Polygon, Polygon, Polygon]: Tuple of polygons for port, front and starboard collision zones
    """

    ship_center = Point([pos_x, pos_y])

    # zone parameters
    angle = np.pi / 4  # zone intersection angle
    offset = np.pi / 2  # zone offset angle
    num_points = 100

    # Close coast zone port
    angle_range_port = np.linspace(-chi + 2 * offset - angle, -chi + 2 * offset + angle, num_points)
    arc_port = [
        (ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle))
        for angle in angle_range_port
    ]
    arc_line_port = LineString(arc_port)
    zone_port = Polygon(list(arc_line_port.coords) + [ship_center])

    # Close coast zone front
    angle_range_front = np.linspace(-chi + offset - angle, -chi + offset + angle, num_points)
    arc_front = [
        (ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle))
        for angle in angle_range_front
    ]
    arc_line_front = LineString(arc_front)
    zone_front = Polygon(list(arc_line_front.coords) + [ship_center])

    # Close coast zone starboard
    angle_range_starboard = np.linspace(-chi - angle, -chi + angle, num_points)
    arc_starboard = [
        (ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle))
        for angle in angle_range_starboard
    ]
    arc_line_starboard = LineString(arc_starboard)
    zone_starboard = Polygon(list(arc_line_starboard.coords) + [ship_center])

    return zone_port, zone_front, zone_starboard


def distances_to_coast(
    poly_port: Polygon, poly_front: Polygon, poly_starboard: Polygon, poly_ship: Polygon, poly_land: MultiPolygon
) -> Tuple[float, float, float]:
    """Calculates distance to coast based on intersection between collision polygons and land polygon.

    Args:
        poly_port (Polygon): Polygon of collision zone on portside of ship
        poly_front (Polygon): Polygon of collision zone in front of ship
        poly_starboard (Polygon): Polygon of collision zone on starboardside of ship
        poly_ship (Polygon): Polygon of ship
        poly_land (MultiPolygon): Polygon of land

    Returns:
        Tuple[float, float, float]: Tuple of distances to land on portside, front and starboardside
    """
    port_intersec_land = poly_port.intersection(poly_land)
    front_intersec_land = poly_front.intersection(poly_land)
    starboard_intersec_land = poly_starboard.intersection(poly_land)

    dist_port = poly_ship.distance(port_intersec_land)
    dist_front = poly_ship.distance(front_intersec_land)
    dist_starboard = poly_ship.distance(starboard_intersec_land)

    return dist_port, dist_front, dist_starboard


def generate_enveloping_polygon(trajectory: np.ndarray, buffer: float) -> Polygon:
    """Creates an enveloping polygon around the trajectory of the vessel, buffered by the given amount.

    Args:
        - trajectory (np.ndarray): Trajectory with min shape 2 x n_samples
        - buffer (float): Buffer size

    Returns:
        Polygon: The query polygon
    """
    point_list = []
    for k in range(trajectory.shape[1]):
        point_list.append((trajectory[1, k], trajectory[0, k]))
    trajectory_linestring = LineString(point_list).buffer(buffer)
    return trajectory_linestring


def extract_hazards_within_bounding_box(
    hazards: list, bbox: Tuple[float, float, float, float], enc: Optional[ENC] = None, show_plots: bool = False
) -> list:
    """Extracts the hazards that are inside the given bounding box.

    Args:
        hazards (list): List of Multipolygon hazards to consider.
        bbox (Tuple[float, float, float, float]): Bounding box to consider in the form (x_min, y_min, x_max, y_max), x = easting, y = northing.
        enc (Optional[ENC], optional): Electronic Navigational Chart object. Defaults to None.
        show_plots (bool, optional): Whether to show plots or not. Defaults to False.

    Returns:
        list: List of hazards inside the bounding box.
    """
    bbox_poly = bbox_to_polygon(bbox)
    intersections = []
    for hazard in hazards:
        if bbox_poly.intersects(hazard):
            overlap = bbox_poly.intersection(hazard)
            if isinstance(overlap, Polygon):
                overlap = MultiPolygon([overlap])
            elif isinstance(overlap, Point):
                overlap = MultiPolygon([overlap.buffer(0.1)])
            elif isinstance(overlap, LineString):
                overlap = MultiPolygon([overlap.buffer(0.1)])
            intersections.append(overlap)

    if enc and show_plots:
        enc.start_display()
        for intersection in intersections:
            enc.draw_polygon(intersection, color="full_horizon", fill=True, alpha=0.5)
    return intersections


def extract_polygons_near_trajectory(
    trajectory: np.ndarray,
    geometry_tree: strtree.STRtree,
    buffer: float,
    enc: ENC = None,
    show_plots: bool = False,
) -> Tuple[list, Polygon]:
    """Extracts the polygons that are relevant for the trajectory of the vessel, inside a corridor of the given buffer size.

    Args:
        - trajectory (np.ndarray): Trajectory to consider.
        - geometry_tree (strtree.STRtree): The rtree containing the relevant grounding hazard polygons.
        - buffer (float): Buffer size
        - enc (Optional[ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.
        - show_plots (bool, optional): Whether to show plots or not. Defaults to False.

    Returns:
        Tuple[list, Polygon]: List of tuples of relevant polygons inside query/envelope polygon and the corresponding original polygon they belong to. Also returns the query polygon.
    """
    enveloping_polygon = generate_enveloping_polygon(trajectory, buffer)
    bbox_poly = bbox_to_polygon(enc.bbox)
    enveloping_polygon = enveloping_polygon.intersection(bbox_poly)
    polygons_near_trajectory_indices = geometry_tree.query(enveloping_polygon)
    polygons_near_trajectory = [geometry_tree.geometries[idx] for idx in polygons_near_trajectory_indices]
    poly_list = []
    for poly in polygons_near_trajectory:
        relevant_poly_list = []
        intersection_poly = enveloping_polygon.intersection(poly)
        if intersection_poly.area == 0.0 and intersection_poly.length == 0.0:
            continue

        if isinstance(intersection_poly, MultiPolygon):
            for sub_poly in intersection_poly.geoms:
                relevant_poly_list.append(sub_poly)
        else:
            relevant_poly_list.append(intersection_poly)
        poly_list.append((relevant_poly_list, poly))

    if enc is not None and show_plots:
        enc.start_display()
        enc.draw_polygon(enveloping_polygon, color="yellow", alpha=0.2)
        # for poly_sublist, _ in poly_list:
        #     for poly in poly_sublist:
        #         enc.draw_polygon(poly, color="red", fill=False)

    return poly_list, enveloping_polygon


def extract_boundary_polygons_inside_envelope(
    poly_tuple_list: list, enveloping_polygon: Polygon, enc: Optional[ENC] = None, show_plots: bool = True
) -> list:
    """Extracts the boundary trianguled polygons that are relevant for the trajectory of the vessel, inside the given envelope polygon.

    Args:
        - poly_tuple_list (list): List of tuples with relevant polygons inside query/envelope polygon and the corresponding original polygon they belong to.
        - enveloping_polygon (Polygon): The query polygon.
        - enc (Optional[senc.ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.
        - show_plots (bool, optional): Whether to show plots or not. Defaults to False.

    Returns:
        list: List of boundary polygons.
    """
    boundary_polygons = []
    for relevant_poly_list, original_polygon in poly_tuple_list:
        for relevant_polygon in relevant_poly_list:
            triangle_boundaries = extract_triangle_boundaries_from_polygon(
                relevant_polygon, enveloping_polygon, original_polygon
            )
            if not triangle_boundaries:
                continue

            if enc is not None and show_plots:
                # enc.draw_polygon(poly, color="pink", alpha=0.3)
                for tri in triangle_boundaries:
                    enc.draw_polygon(tri, color="red", fill=False)

            boundary_polygons.extend(triangle_boundaries)
    return boundary_polygons


def extract_triangle_boundaries_from_polygon(
    polygon: Polygon, planning_area_envelope: Polygon, original_polygon: Polygon
) -> list:
    """Extracts the triangles that comprise the boundary of the polygon.

    Triangles are filtered out if they have two vertices on the envelope boundary and is inside of the original polygon.

    Args:
        - polygon (Polygon): The polygon in consideration inside the envelope polygon.
        - planning_area_envelope (Polygon): A polygon representing the relevant area the vessel is planning to navigate in.
        - original_polygon (Polygon): The original polygon that the relevant polygon belongs to.

    Returns:
        list: List of shapely polygons representing the boundary triangles for the polygon.
    """
    cdt = constrained_delaunay_triangulation_custom(polygon)
    # return cdt
    original_polygon_boundary = LineString(original_polygon.exterior.coords).buffer(0.0001)
    boundary_triangles = []
    if len(cdt) == 1:
        return cdt

    # Check if triangle has two vertices on the envelope boundary and is inside of the original polygon
    for tri in cdt:
        v_count = 0
        idx_prev = 0
        for idx, v in enumerate(tri.exterior.coords):
            if v_count == 2 and idx_prev == idx - 1 and tri not in boundary_triangles:
                boundary_triangles.append(tri)
                break
            v_point = Point(v)
            if original_polygon_boundary.contains(v_point):
                v_count += 1
                idx_prev = idx

    return boundary_triangles


def constrained_delaunay_triangulation_custom(polygon: Polygon) -> list:
    """Converts a polygon to a list of triangles. Basically constrained delaunay triangulation.

    Args:
        - polygon (Polygon): The polygon to triangulate.

    Returns:
        list: List of triangles as shapely polygons.
    """
    assert polygon.is_empty is False, "Polygon is empty"
    res_intersection_gdf = gpd.GeoDataFrame(geometry=[polygon])
    # Create ID to identify overlapping polygons
    res_intersection_gdf["TRI_ID"] = res_intersection_gdf.index
    # List to keep triangulated geometries
    triangles = []
    # List to keep the original IDs
    triangle_ids = []
    # Triangulate single or multi-polygons
    for i, _ in res_intersection_gdf.iterrows():
        tri_ = ops.triangulate(res_intersection_gdf.geometry.values[i])
        triangles.append(tri_)
        for _ in range(0, len(tri_)):
            triangle_ids.append(res_intersection_gdf.TRI_ID.values[i])
    # Check if it is a single or multi-polygon
    len_list = len(triangles)
    triangles = np.array(triangles).flatten().tolist()
    # unlist geometries for multi-polygons
    if len_list > 1:
        triangles = [item for sublist in triangles for item in sublist]
    # Create triangulated polygons
    filtered_triangles = gpd.GeoDataFrame(triangles)
    filtered_triangles = filtered_triangles.set_geometry(triangles)
    del filtered_triangles[0]
    # Assign original IDs to each triangle
    filtered_triangles["TRI_ID"] = triangle_ids
    # Create new ID for each triangle
    filtered_triangles["LINK_ID"] = filtered_triangles.index
    # Create centroids from all triangles
    filtered_triangles["centroid"] = filtered_triangles.centroid
    filtered_triangles_centroid = filtered_triangles.set_geometry("centroid")
    del filtered_triangles_centroid["geometry"]
    del filtered_triangles["centroid"]
    # Find triangle centroids inside original polygon
    filtered_triangles_join = gpd.sjoin(
        filtered_triangles_centroid[["centroid", "TRI_ID", "LINK_ID"]],
        res_intersection_gdf[["geometry", "TRI_ID"]],
        how="inner",
        predicate="within",
    )
    # Remove overlapping from other triangles (Necessary for multi-polygons overlapping or close to each other)
    filtered_triangles_join = filtered_triangles_join[
        filtered_triangles_join["TRI_ID_left"] == filtered_triangles_join["TRI_ID_right"]
    ]
    # Remove overload triangles from same filtered_triangless
    filtered_triangles = filtered_triangles[filtered_triangles["LINK_ID"].isin(filtered_triangles_join["LINK_ID"])]
    filtered_triangles = filtered_triangles.geometry.values
    # double check
    cdt_triangles = []
    area_eps = 1e-4
    for tri in triangles:
        intersection_poly = tri.intersection(polygon)
        if isinstance(intersection_poly, Point) or isinstance(intersection_poly, LineString):
            continue
        if intersection_poly.area < area_eps:
            continue

        if isinstance(intersection_poly, MultiPolygon) or isinstance(intersection_poly, GeometryCollection):
            for sub_poly in intersection_poly.geoms:
                if sub_poly.area < area_eps or isinstance(sub_poly, Point) or isinstance(sub_poly, LineString):
                    continue
                cdt_triangles.append(sub_poly)
        else:
            cdt_triangles.append(intersection_poly)
    return cdt_triangles


def plot_trajectory(
    trajectory: np.ndarray,
    enc: ENC,
    color: str,
    edge_style: Optional[str] = None,
    buffer: Optional[float] = 0.5,
    linewidth: Optional[float] = 1.0,
    alpha: Optional[float] = 1.0,
) -> None:
    """Plots the trajectory on the ENC.

    Args:
        trajectory (np.ndarray): Input trajectory, minimum 2 x n_samples.
        enc (ENC): Electronic Navigational Chart object
        color (str): Color of the trajectory
        marker_type (Optional[str], optional): Marker type for the trajectory. Defaults to None.@
        marker_size (Optional[float], optional): Marker size for the trajectory. Defaults to None.
        edge_style (Optional[str], optional): Edge style for the trajectory. Defaults to None.
        buffer (Optional[float], optional): Buffer of the trajectory. Defaults to 0.5.
        linewidth (Optional[float], optional): linewidth of the trajectory. Defaults to 0.5.
    """
    enc.start_display()
    trajectory_line = []
    for k in range(trajectory.shape[1]):
        trajectory_line.append((trajectory[1, k], trajectory[0, k]))
    enc.draw_line(
        trajectory_line,
        color=color,
        buffer=buffer,
        linewidth=linewidth,
        edge_style=edge_style,
        alpha=alpha,
    )


def plot_disturbance(
    magnitude: float,
    direction: float,
    name: str,
    enc: ENC,
    color: str,
    linewidth: Optional[float] = 2.5,
    location: Optional[str] = "topright",
    text_location_offset: Optional[Tuple[float, float]] = (0.0, 0.0),
) -> plt.axes:
    """Plots a disturbance vector on the ENC as a vector arrow inside a circle.
    The name of the disturbance is plotted below the circle, with an offset given by text_location_offset.

    Args:
        magnitude (float): Magnitude of the disturbance / length of the disturbance vector
        direction (float): Direction of the disturbance (defined in a north-east coordinate system)
        name (str): Name of the disturbance
        enc (ENC): Electronic Navigational Chart object
        color (str): Color of the disturbance vector
        linewidth (Optional[float]): Arrow thickness. Defaults to 1.0.
        location (Optional[str]): Location of the disturbance vector in ["topleft", "topright", "bottomleft", "bottomright"]. Defaults to "topright".
        text_location_offset (Optional[Tuple[float, float]]): Offset of the text location. Defaults to (0.0, 0.0).
    """
    enc.start_display()
    xmin, ymin, xmax, ymax = enc.bbox  # x is east, y is north
    if location == "topright":
        origin = (xmax - 0.1 * (xmax - xmin), ymax - 0.1 * (ymax - ymin))
    elif location == "topleft":
        origin = (xmin + 0.1 * (xmax - xmin), ymax - 0.1 * (ymax - ymin))
    elif location == "bottomright":
        origin = (xmax - 0.1 * (xmax - xmin), ymin + 0.1 * (ymax - ymin))
    elif location == "bottomleft":
        origin = (xmin + 0.1 * (xmax - xmin), ymin + 0.1 * (ymax - ymin))

    arrow_start = origin
    arrow_end = (origin[0] + magnitude * np.sin(direction), origin[1] + magnitude * np.cos(direction))
    text_location = (
        origin[0] + text_location_offset[0] - 0.8 * magnitude,
        origin[1] - 1.2 * magnitude + text_location_offset[1],
    )

    circle_handle = enc.draw_circle(origin, radius=magnitude, color="white", fill=True, alpha=0.2)
    arrow_handle = enc.draw_arrow(arrow_start, arrow_end, color=color, width=linewidth, fill=True)
    text_handle = enc.draw_text(name, text_location, color=color, size=10)
    return [circle_handle, arrow_handle, text_handle]


def plot_waypoints(
    waypoints: np.ndarray,
    enc: ENC,
    color: str,
    point_buffer: Optional[float] = 10,
    disk_buffer: Optional[float] = 80,
    hole_buffer: Optional[float] = 10,
    linewidth: Optional[float] = None,
    alpha: Optional[float] = 0.6,
    show_annuluses: Optional[bool] = True,
    draft: Optional[float] = 5.0,
):
    lines = [
        LineString([(wp1[1], wp1[0]), (wp2[1], wp2[0])]).buffer(point_buffer)
        for wp1, wp2 in zip(waypoints.T, waypoints[:, 1:].T)
    ]
    if show_annuluses:
        points = [Point((wp[1], wp[0])) for wp in waypoints.T]
        disks = [p.buffer(disk_buffer) for p in points]
        holes = [p.buffer(hole_buffer) for p in points]
        path = shapely.unary_union(lines + disks)
        for i, hole in enumerate(holes):
            path = path.difference(hole)
    else:
        lines.pop(1)
        path = shapely.unary_union(lines)

    # hazards = extract_relevant_grounding_hazards_as_union(find_minimum_depth(draft, enc), enc)[0]
    # if path.intersects(hazards):
    #     overlap = path.intersection(hazards)
    #     enc.draw_polygon(overlap, "red", thickness=linewidth, alpha=alpha)
    #     path = path.difference(hazards)
    enc.draw_polygon(path, color, thickness=linewidth, alpha=alpha)


def plot_dynamic_obstacles(
    dynamic_obstacles: list, color: str, enc: ENC, T: float, dt: float, map_origin: Optional[np.ndarray] = None
) -> None:
    """Plots the dynamic obstacles as ellipses and ship polygons.

    Args:
        dynamic_obstacles (list): List of tuples containing (ID, state, cov, length, width)
        color (string): Color of the ellipses
        enc (ENC): Electronic Navigational Chart object
        T (float): Horizon to predict straight line trajectories for the dynamic obstacles
        dt (float): Time step for the straight line trajectories
        map_origin (np.ndarray, optional): Origin of the map in the form [x, y]^T
    """
    N = int(T / dt)
    enc.start_display()
    dynamic_obstacles_copy = copy.deepcopy(dynamic_obstacles)
    for ID, state, cov, length, width in dynamic_obstacles_copy:
        if map_origin is not None:
            state[:2] += map_origin
        ellipse_x, ellipse_y = mhm.create_probability_ellipse(cov, 0.67)
        ell_geometry = Polygon(zip(ellipse_y + state[1], ellipse_x + state[0]))
        # enc.draw_polygon(ell_geometry, color=color, alpha=0.4)

        for k in range(0, N, 5):
            do_poly = create_ship_polygon(
                state[0] + k * dt * state[2],
                state[1] + k * dt * state[3],
                np.arctan2(state[3], state[2]),
                length,
                width,
                length_scaling=1.0,
                width_scaling=1.0,
            )
            enc.draw_polygon(do_poly, color=color)
        do_poly = create_ship_polygon(
            state[0], state[1], np.arctan2(state[3], state[2]), length, width, length_scaling=1.0, width_scaling=1.0
        )
        enc.draw_polygon(do_poly, color=color)


def plot_rrt_tree(node_list: list, enc: ENC) -> None:
    """Plots an RRT tree given by the list of nodes containing (state, parent_id, id, trajectory, inputs, cost)

    Args:
        node_list (list): List of nodes containing (state, parent_id, id, trajectory, inputs, cost)
        enc (ENC): Electronic Navigational Chart object
    """
    enc.start_display()
    for node in node_list:
        # enc.draw_circle(
        #     (node["state"][1], node["state"][0]), 2.5, color="green", fill=False, thickness=0.8, edge_style=None
        # )
        for sub_node in node_list:
            if node["id"] == sub_node["id"] or sub_node["parent_id"] != node["id"]:
                continue
            points = [(tt[1], tt[0]) for tt in sub_node["trajectory"]]
            if len(points) > 1:
                enc.draw_line(points, color="white", buffer=0.5, linewidth=0.5)


def standardize_polygon_intersections(intersection: Point | LineString | MultiLineString) -> Point:
    """Converts a shapely intersection to a point.
    If intersection contains multiple points, the closest one is returned.

    Args:
        - intersection (Point | Linestring | Multilinestring): The intersection to convert

    Returns:
        Point: Shapely point object containing the closest point of intersection

    """
    if isinstance(intersection, LineString):
        return Point(intersection.coords[0])
    elif isinstance(intersection, Point):
        return intersection
    elif isinstance(intersection, MultiLineString):
        return Point(intersection.geoms[0].coords[0])


def multi_polygon_to_list_of_ndarray(filtered_relevant_hazards: list[MultiPolygon]) -> list[np.ndarray]:
    """Converts a list of MultiPolygons to a list of numpy ndarrays.

    Args:
        - filtered_relevant_hazards list[MultiPolygon]: The data which is to be converted from "shapely" to "numpy".

    Returns:
        list[np.ndarray]: The converted data. Now compatible with the C++ binding of the PSBMPC.
    """
    polygon_list = []
    for multi_polygon in filtered_relevant_hazards:
        for polygon in multi_polygon.geoms:
            temp_polygon_np = np.array([])
            for point in polygon.exterior.coords:
                temp_polygon_np = np.append(temp_polygon_np, [point[0], point[1]])
            polygon_list.append(temp_polygon_np)
    return polygon_list


def multi_polygon_to_list_of_ndarray_flip_x_y(filtered_relevant_hazards: list[MultiPolygon]) -> list[np.ndarray]:
    """Converts a list of MultiPolygons to a list of numpy ndarrays. The function also flips the x and y coordinates.

    Args:
        - filtered_relevant_hazards list[MultiPolygon]: The data which is to be converted from "shapely" to "numpy".

    Returns:
        list[np.ndarray]: The converted data. Now compatible with the C++ binding of the PSBMPC.
    """
    polygon_list = []
    if isinstance(filtered_relevant_hazards[0], Polygon):
        for polygon in filtered_relevant_hazards:
            temp_polygon_np = np.array([])
            for point in polygon.exterior.coords:
                temp_polygon_np = np.append(temp_polygon_np, [point[1], point[0]])
            polygon_list.append(temp_polygon_np)
    else:
        for multi_polygon in filtered_relevant_hazards:
            for polygon in multi_polygon.geoms:
                temp_polygon_np = np.array([])
                for point in polygon.exterior.coords:
                    temp_polygon_np = np.append(temp_polygon_np, [point[1], point[0]])
                polygon_list.append(temp_polygon_np)
    return polygon_list


def extract_grounding_hazards_method_from_GPU_paper(
    filtered_relevant_hazards : list[MultiPolygon],    
    radius_of_coverage : float,
    ownship_state : np.ndarray
    ) -> list[MultiPolygon]:
    """Extracts the relevant grounding hazards from a list[MultiPolygon] and returns them as a multipolygon.
    This includes land, shore and seabed polygons that are below the vessel's minimum depth.

    The implemented method is the same as the one used in: "Ship Collision Avoidance and Anti 
    Grounding Using Parallelized Cost Evaluation in Probabilistic Scenario-Based Model Predictive Control", 
    https://ieeexplore.ieee.org/document/9924235.

    Args:
    - filtered_relevant_hazards (list[MultiPolygon]): Grounding hazards from an ENC parameterized as polygons.
    - radius_of_coverage (float): The radius of which to concider grounding hazards, the radius of the relevant grounding sector.  
    - ownship_state (np.ndarray): The ownship state [x, y, psi, u, v, r].

    Returns:
        list[MultiPolygon]: The grounding hazards inside the relevant grounding sector.
    """
    # Defining a relevant sector.
    ownship_x = ownship_state[0]
    ownship_y = ownship_state[1]

    # Defining a ship point.
    ownship_point = Point(ownship_x, ownship_y)

    circle_relevant_sector = Point(
        ownship_point.x, 
        ownship_point.y
    ).buffer(radius_of_coverage, resolution = 100)

    # Finding the intersection of relevant hazards and relevant area.
    filtered_relevant_hazards_circle_relevant_sector_intersection = circle_relevant_sector.intersection(
        filtered_relevant_hazards[0]
    )

    if type(filtered_relevant_hazards_circle_relevant_sector_intersection) != MultiPolygon:
        filtered_relevant_hazards_circle_relevant_sector_intersection = \
            MultiPolygon([filtered_relevant_hazards_circle_relevant_sector_intersection])

    return filtered_relevant_hazards_circle_relevant_sector_intersection
