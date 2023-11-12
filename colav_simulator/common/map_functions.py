"""
    map_functions.py

    Summary:
        Contains functionality for plotting the Electronic Navigational Chart (ENC),
        computing distance to land polygons, generating random ship starting positions etc.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""
import colav_simulator.common.miscellaneous_helper_methods as mhm
import geopandas as gpd
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as scipy_spatial
import seacharts.display.colors as colors
import shapely.ops as ops
import math

from typing import Optional, Tuple
from cartopy.feature import ShapelyFeature
from osgeo import osr
from seacharts.enc import ENC
from shapely import affinity, strtree
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, MultiLineString


def local2latlon(x: float | list | np.ndarray, y: float | list | np.ndarray, utm_zone: int) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
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


def latlon2local(lat: float | list | np.ndarray, lon: float | list | np.ndarray, utm_zone: int) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
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
    min_depth: int, enveloping_polygon: Polygon, enc: Optional[ENC] = None, as_polygon_list: bool = False, show_plots: bool = False
) -> MultiPolygon | list:
    """Extracts the safe sea area from the ENC as a list of polygons.

    This includes sea polygons that are above the vessel`s minimum depth.

    Args:
        - min_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enveloping_polygon (geometry.Polygon): The query polygon.
        - enc (Optional[senc.ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.
        - as_polygon_list (bool, optional): Option for returning the safe sea area as a list of polygons. Defaults to False.
        - show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        MultiPolygon | list: The safe sea area.
    """
    safe_sea = enc.seabed[min_depth].geometry.intersection(enveloping_polygon)
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
        bbox (Tuple[float, float, float, float]): The bounding box.

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


def create_safe_sea_triangulation(enc: ENC, vessel_min_depth: int = 5, show_plots: bool = True) -> list:
    """Creates a constrained delaunay triangulation of the safe sea region.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        vessel_min_depth (int, optional): The safe minimum depth for the vessel to voyage in. Defaults to 5.

    Returns:
        list: List of triangles.
    """
    safe_sea_poly_list = extract_safe_sea_area(vessel_min_depth, bbox_to_polygon(enc.bbox), enc, as_polygon_list=True, show_plots=True)
    cdt_list = []
    largest_poly_area = 0.0
    for poly in safe_sea_poly_list:
        cdt = constrained_delaunay_triangulation_custom(poly)
        if poly.area > largest_poly_area:
            largest_poly_area = poly.area
            cdt_largest = cdt
        if show_plots:
            enc.draw_polygon(poly, color="orange", alpha=0.5)
            enc.start_display()
            for triangle in cdt:
                enc.draw_polygon(triangle, color="black", fill=False)
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


def create_ship_polygon(x: float, y: float, heading: float, length: float, width: float, length_scaling: float = 1.0, width_scaling: float = 1.0) -> Polygon:
    """Creates a ship polygon from the ship`s position, heading, length and width.

    Args:
        x (float): The ship`s north position
        y (float): The ship`s east position
        heading (float): The ship`s heading
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


def plot_background(ax: plt.Axes, enc: ENC, show_shore: bool = True, show_seabed: bool = True, dark_mode: bool = True) -> None:
    """Creates a static background based on the input seacharts

    Args:
        ax (plt.Axes): Matplotlib axes handle.
        enc (ENC): Electronic Navigational Chart object
        show = Option for visualization

    Returns:
        Tuple[]: Tuple of limits in x and y for the background extent
    """
    # For every layer put in list and assign a color

    if enc.land:
        color = "#142c38" if dark_mode else colors.color_picker(enc.land.color)
        ax.add_feature(ShapelyFeature([enc.land.geometry], color=color, zorder=enc.land.z_order, crs=enc.crs))

    if show_shore and enc.shore:
        color = "#142c38" if dark_mode else colors.color_picker(enc.shore.color)
        ax.add_feature(ShapelyFeature([enc.shore.geometry], color=color, zorder=enc.shore.z_order, crs=enc.crs))

    if show_seabed and enc.seabed:
        bins = len(enc.seabed.keys())
        count = 0
        for _, layer in enc.seabed.items():
            rank = layer.z_order + count
            color = colors.color_picker(count, bins)
            ax.add_feature(ShapelyFeature([layer.geometry], color=color, zorder=rank, crs=enc.crs))
            count += 1

    x_min, y_min, x_max, y_max = enc.bbox
    ax.set_extent((x_min, x_max, y_min, y_max), crs=enc.crs)


def find_minimum_depth(vessel_draft: float, enc: ENC):
    """Find the minimum seabed depth for the given vessel draft (for it to avoid grounding)

    Args:
        vessel_draft (float): The vessel`s draft.

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

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
    return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]


def extract_relevant_grounding_hazards_as_union(vessel_min_depth: int, enc: ENC, buffer: Optional[float] = None, show_plots: bool = False) -> list:
    """Extracts the relevant grounding hazards from the ENC as a multipolygon.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.
        buffer (Optional[float], optional): Buffer for polygons. Defaults to None.
        show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        geometry.MultiPolygon: The relevant grounding hazards.
    """
    dangerous_seabed = enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
    relevant_hazards = [enc.land.geometry.union(enc.shore.geometry).union(dangerous_seabed)]
    filtered_relevant_hazards = []
    for hazard in relevant_hazards:
        if isinstance(hazard, Polygon):
            poly = MultiPolygon([hazard])
        else:
            poly = MultiPolygon(Polygon(p.exterior) for p in hazard.geoms if isinstance(p, Polygon))
        if buffer is not None:
            poly = poly.buffer(buffer)
        filtered_relevant_hazards.append(poly)

    if show_plots:
        enc.start_display()
        for hazard in filtered_relevant_hazards:
            enc.draw_polygon(hazard, color="red", alpha=0.5)
    return filtered_relevant_hazards


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


def generate_random_start_position_from_draft(
    rng: np.random.Generator, enc: ENC, draft: float, min_land_clearance: float = 100.0, safe_sea_cdt: Optional[list] = None
) -> Tuple[float, float]:
    """
    Randomly defining starting easting and northing coordinates of a ship
    inside the safe sea region by considering a ship draft, with an optional land clearance distance.

    Args:
        - rng (np.random.Generator): Numpy random generator.
        - enc (ENC): Electronic Navigational Chart object
        - draft (float): Ship's draft in meters.
        - min_land_clearance (float): Minimum distance to land in meters.
        - safe_sea_cdt (Optional[list]): List of triangles defining the safe sea region, used to sample more efficiently. Defaults to None.

    Returns:
        - Tuple[float, float]: Tuple of starting x and y coordinates for the ship.
    """
    depth = find_minimum_depth(draft, enc)
    safe_sea = enc.seabed[depth]
    bbox = enc.bbox

    is_safe = False
    iter_count = 0
    while not is_safe:
        if safe_sea_cdt is not None:
            random_triangle = rng.choice(safe_sea_cdt)
            assert isinstance(random_triangle, Polygon) and len(random_triangle.exterior.coords) >= 4, "The safe sea region must be a polygon and triangle."
            x, y = random_triangle.exterior.coords.xy
            p1 = np.array([x[0], y[0]])
            p2 = np.array([x[1], y[1]])
            p3 = np.array([x[2], y[2]])
            random_point = mhm.sample_from_triangle_region(p1, p2, p3, rng)
            easting, northing = random_point[0], random_point[1]
        else:
            easting, northing = rng.uniform(bbox[0], bbox[2]), rng.uniform(bbox[1], bbox[3])

        is_ok_clearance = min_distance_to_land(enc, easting, northing) >= min_land_clearance
        if safe_sea.geometry.contains(Point(easting, northing)) and is_ok_clearance:
            break

        iter_count += 1
        if iter_count > 1000:
            raise Exception("Could not find a valid start position. Check the map data of your ENC object.")

    return northing, easting


def compute_distance_vectors_to_grounding(vessel_trajectory: np.ndarray, minimum_vessel_depth: int, enc: ENC, show_plots: bool = False) -> np.ndarray:
    """Computes the distance vectors to grounding at each step of the given vessel trajectory.

    Args:
        - vessel_trajectory (np.ndarray): The vessel`s trajectory, 2 x n_samples.
        - minimum_vessel_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enc (ENC): The ENC to check for grounding.
        - show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        - np.ndarray: The distance to grounding at each step of the vessel trajectory.
    """
    if show_plots:
        enc.start_display()
    relevant_hazards = extract_relevant_grounding_hazards_as_union(minimum_vessel_depth, enc)
    vessel_traj_linestring = mhm.ndarray_to_linestring(vessel_trajectory)

    distance_vectors = np.ndarray((2, vessel_trajectory.shape[1]))
    for idx, point in enumerate(vessel_traj_linestring.coords):
        for hazard in relevant_hazards:
            dist = hazard.distance(Point(point))

            point = Point(vessel_traj_linestring.coords[idx])
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
                    enc.draw_line(points, color="black", width=0.5, marker_type="o")

                dist_vec = np.array([points[1][0] - points[0][0], points[1][1] - points[0][1]])
                if np.linalg.norm(dist_vec) <= min_dist:
                    min_dist_vec = dist_vec
                    min_dist = np.linalg.norm(min_dist_vec)
            distance_vectors[:, idx] = min_dist_vec
    return distance_vectors


def compute_closest_grounding_dist(vessel_trajectory: np.ndarray, minimum_vessel_depth: int, enc: ENC, show_enc: bool = False) -> Tuple[float, np.ndarray, int]:
    """Computes the closest distance to grounding for the given vessel trajectory.

    Args:
        - vessel_trajectory (np.ndarray): The vessel`s trajectory, 2 x n_samples.
        - minimum_vessel_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enc (senc.ENC): The ENC to check for grounding.

    Returns:
        - Tuple[float, int]: The closest distance to grounding, corresponding distance vector and the index of the trajectory point.
    """
    relevant_hazards = extract_relevant_grounding_hazards(minimum_vessel_depth, enc)
    vessel_traj_linestring = mhm.ndarray_to_linestring(vessel_trajectory)
    if enc and show_enc:
        enc.start_display()
        for hazard in relevant_hazards:
            enc.draw_polygon(hazard, color="red")
    # intersection_points = find_intersections_line_polygon(vessel_traj_linestring, relevant_hazards, enc)

    # Will find the closest grounding point.
    min_dist = 1e12
    for idx, point in enumerate(vessel_traj_linestring.coords):
        for hazard in relevant_hazards:
            dist = hazard.distance(Point(point))
            if dist < min_dist:
                min_dist = dist
                min_idx = idx

    closest_point = Point(vessel_traj_linestring.coords[min_idx])
    nearest_poly_points = []
    for hazard in relevant_hazards:
        nearest_point = ops.nearest_points(closest_point, hazard)[1]
        nearest_poly_points.append(nearest_point)

    epsilon = 0.01
    for i, point in enumerate(nearest_poly_points):
        points = [
            (np.asarray(closest_point.coords.xy[0])[0], np.asarray(closest_point.coords.xy[1])[0]),
            (np.asarray(point.coords.xy[0])[0], np.asarray(point.coords.xy[1])[0]),
        ]

        if enc and show_enc:
            enc.draw_line(points, color="cyan", marker_type="o")

        min_dist_vec = np.array([points[1][0] - points[0][0], points[1][1] - points[0][1]])
        if np.linalg.norm(min_dist_vec) <= min_dist + epsilon and np.linalg.norm(min_dist_vec) >= min_dist - epsilon:
            break

    if enc and show_enc:
        enc.close_display()
    return min_dist, min_dist_vec, min_idx


def min_distance_to_land(enc: ENC, y: float, x: float) -> float:
    """Compute the minimum distance to land from a given point.

    Args:
        enc (ENC): Electronic Navigational Chart object
        y (float): Ship's easting coordinate
        x (float): Ship's northing coordinate

    Returns:
        float: Minimum distance to land in meters.
    """
    position = Point(y, x)
    distance = enc.land.geometry.distance(position)
    return distance


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


def min_distance_and_point_to_hazards(hazards: list, x: float, y: float) -> [float, Point]:
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


def check_if_segment_crosses_grounding_hazards(enc: ENC, p2: np.ndarray, p1: np.ndarray, draft: float = 5.0) -> bool:
    """Checks if a line segment between two positions/points crosses nearby grounding hazards (land, shore).

    Args:
        enc (ENC): Electronic Navigational Chart object
        p2 (np.ndarray): Second position.
        p1 (np.ndarray): First position.
        draft (float): Ship's draft in meters.

    Returns:
        bool: True if path segment crosses land, False otherwise.

    """
    # Create linestring with east as the x value and north as the y value
    p2_reverse = (p2[1], p2[0])
    p1_reverse = (p1[1], p1[0])
    wp_line = LineString([p1_reverse, p2_reverse])

    entire_seabed = enc.seabed[0].geometry
    min_depth = find_minimum_depth(draft, enc)

    seabed_down_to_draft = entire_seabed.difference(enc.seabed[min_depth].geometry)

    intersects_relevant_seabed = wp_line.intersects(seabed_down_to_draft)

    intersects_land_or_shore = wp_line.intersects(enc.shore.geometry)

    crosses_grounding_hazards = intersects_land_or_shore or intersects_relevant_seabed

    return crosses_grounding_hazards


def generate_ship_sector_polygons(pos_x: float, pos_y: float, chi: float, safety_radius: float) -> Tuple[Polygon, Polygon, Polygon]:
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
    arc_port = [(ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle)) for angle in angle_range_port]
    arc_line_port = LineString(arc_port)
    zone_port = Polygon(list(arc_line_port.coords) + [ship_center])

    # Close coast zone front
    angle_range_front = np.linspace(-chi + offset - angle, -chi + offset + angle, num_points)
    arc_front = [(ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle)) for angle in angle_range_front]
    arc_line_front = LineString(arc_front)
    zone_front = Polygon(list(arc_line_front.coords) + [ship_center])

    # Close coast zone starboard
    angle_range_starboard = np.linspace(-chi - angle, -chi + angle, num_points)
    arc_starboard = [(ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle)) for angle in angle_range_starboard]
    arc_line_starboard = LineString(arc_starboard)
    zone_starboard = Polygon(list(arc_line_starboard.coords) + [ship_center])

    return zone_port, zone_front, zone_starboard


def distances_to_coast(poly_port: Polygon, poly_front: Polygon, poly_starboard: Polygon, poly_ship: Polygon, poly_land: MultiPolygon) -> Tuple[float, float, float]:
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
        - trajectory (np.ndarray): Trajectory with columns [x, y, psi, u, v, r]
        - buffer (float): Buffer size

    Returns:
        Polygon: The query polygon
    """
    point_list = []
    for k in range(trajectory.shape[1]):
        point_list.append((trajectory[1, k], trajectory[0, k]))
    trajectory_linestring = LineString(point_list).buffer(buffer)
    return trajectory_linestring


def extract_polygons_near_trajectory(
    trajectory: np.ndarray, geometry_tree: strtree.STRtree, buffer: float, enc: Optional[ENC] = None, show_plots: bool = False
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
    polygons_near_trajectory = geometry_tree.query(enveloping_polygon)
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


def extract_boundary_polygons_inside_envelope(poly_tuple_list: list, enveloping_polygon: Polygon, enc: Optional[ENC] = None, show_plots: bool = True) -> list:
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
            triangle_boundaries = extract_triangle_boundaries_from_polygon(relevant_polygon, enveloping_polygon, original_polygon)
            if not triangle_boundaries:
                continue

            if enc is not None and show_plots:
                # enc.draw_polygon(poly, color="pink", alpha=0.3)
                for tri in triangle_boundaries:
                    enc.draw_polygon(tri, color="red", fill=False)

            boundary_polygons.extend(triangle_boundaries)
    return boundary_polygons


def extract_triangle_boundaries_from_polygon(polygon: Polygon, planning_area_envelope: Polygon, original_polygon: Polygon) -> list:
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


# def constrained_delaunay_triangulation(polygon: Polygon) -> list:
#     """Uses the triangle library to compute a constrained delaunay triangulation.

#     Args:
#         polygon (Polygon): The polygon to triangulate.

#     Returns:
#         list: List of triangles as shapely polygons.
#     """
#     x, y = polygon.exterior.coords.xy
#     vertices = np.array([list(a) for a in zip(x, y)])
#     cdt = tr.triangulate({"vertices": vertices})
#     triangle_indices = cdt["triangles"]
#     triangles = [Polygon([cdt["vertices"][i] for i in tri]) for tri in triangle_indices]

#     cdt_triangles = []
#     for tri in triangles:
#         intersection_poly = tri.intersection(polygon)

#         if isinstance(intersection_poly, Point) or isinstance(intersection_poly, LineString):
#             continue

#         if intersection_poly.area == 0.0:
#             continue

#         # cdt_triangles.append(tri)
#         if isinstance(intersection_poly, MultiPolygon) or isinstance(intersection_poly, GeometryCollection):
#             for sub_poly in intersection_poly.geoms:
#                 if sub_poly.area == 0.0 or isinstance(sub_poly, Point) or isinstance(sub_poly, LineString):
#                     continue
#                 cdt_triangles.append(sub_poly)
#         else:
#             cdt_triangles.append(intersection_poly)
#     return cdt_triangles


def constrained_delaunay_triangulation_custom(polygon: Polygon) -> list:
    """Converts a polygon to a list of triangles. Basically constrained delaunay triangulation.

    Args:
        - polygon (Polygon): The polygon to triangulate.

    Returns:
        list: List of triangles as shapely polygons.
    """
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
        filtered_triangles_centroid[["centroid", "TRI_ID", "LINK_ID"]], res_intersection_gdf[["geometry", "TRI_ID"]], how="inner", predicate="within"
    )
    # Remove overlapping from other triangles (Necessary for multi-polygons overlapping or close to each other)
    filtered_triangles_join = filtered_triangles_join[filtered_triangles_join["TRI_ID_left"] == filtered_triangles_join["TRI_ID_right"]]
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


def plot_trajectory(trajectory: np.ndarray, enc: ENC, color: str, marker_type: Optional[str] = None, edge_style: Optional[str] = None) -> None:
    """Plots the trajectory on the ENC.

    Args:
        trajectory (np.ndarray): Input trajectory, minimum 2 x n_samples.
        enc (ENC): Electronic Navigational Chart object
        color (str): Color of the trajectory
    """
    enc.start_display()
    trajectory_line = []
    for k in range(trajectory.shape[1]):
        trajectory_line.append((trajectory[1, k], trajectory[0, k]))
    enc.draw_line(trajectory_line, color=color, width=0.5, thickness=0.5, marker_type=marker_type, edge_style=edge_style)


def plot_dynamic_obstacles(dynamic_obstacles: list, enc: ENC, T: float, dt: float) -> None:
    """Plots the dynamic obstacles as ellipses and ship polygons.

    Args:
        dynamic_obstacles (list): List of tuples containing (ID, state, cov, length, width)
        enc (ENC): Electronic Navigational Chart object
        T (float): Horizon to predict straight line trajectories for the dynamic obstacles
        dt (float): Time step for the straight line trajectories
    """
    N = int(T / dt)
    enc.start_display()
    for (ID, state, cov, length, width) in dynamic_obstacles:
        ellipse_x, ellipse_y = mhm.create_probability_ellipse(cov, 0.99)
        ell_geometry = Polygon(zip(ellipse_y + state[1], ellipse_x + state[0]))
        enc.draw_polygon(ell_geometry, color="orange", alpha=0.3)

        for k in range(0, N, 10):
            do_poly = create_ship_polygon(
                state[0] + k * dt * state[2], state[1] + k * dt * state[3], np.arctan2(state[3], state[2]), length, width, length_scaling=1.0, width_scaling=1.0
            )
            enc.draw_polygon(do_poly, color="red")
        do_poly = create_ship_polygon(state[0], state[1], np.arctan2(state[3], state[2]), length, width, length_scaling=1.0, width_scaling=1.0)
        enc.draw_polygon(do_poly, color="red")


def plot_rrt_tree(node_list: list, enc: ENC) -> None:
    """Plots an RRT tree given by the list of nodes containing (state, parent_id, id, trajectory, inputs, cost)

    Args:
        node_list (list): List of nodes containing (state, parent_id, id, trajectory, inputs, cost)
        enc (ENC): Electronic Navigational Chart object
    """
    enc.start_display()
    for node in node_list:
        enc.draw_circle((node["state"][1], node["state"][0]), 2.5, color="green", fill=False, thickness=0.8, edge_style=None)
        for sub_node in node_list:
            if node["id"] == sub_node["id"] or sub_node["parent_id"] != node["id"]:
                continue
            points = [(tt[1], tt[0]) for tt in sub_node["trajectory"]]
            if len(points) > 1:
                enc.draw_line(points, color="white", width=0.5, thickness=0.5, marker_type=None)


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
