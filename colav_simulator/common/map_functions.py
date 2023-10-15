"""
    map_functions.py

    Summary:
        Contains functionality for plotting the Electronic Navigational Chart (ENC),
        computing distance to land polygons, generating random ship starting positions etc.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""
import random
import colav_simulator.common.miscellaneous_helper_methods as mhm
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import seacharts.display.colors as colors
import shapely.ops as ops
import math

from typing import Tuple
from cartopy.feature import ShapelyFeature
from osgeo import osr
from seacharts.enc import ENC
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon


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


def plot_background(ax: plt.Axes, enc: ENC, show_shore: bool = True, show_seabed: bool = True) -> None:
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
        color = colors.color_picker(enc.land.color)
        ax.add_feature(ShapelyFeature([enc.land.geometry], color=color, zorder=enc.land.z_order, crs=enc.crs))

    if show_shore and enc.shore:
        color = colors.color_picker(enc.shore.color)
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
    """Extracts the relevant grounding hazards from the ENC as a list of polygons.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
    return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]


def extract_relevant_grounding_hazards_as_union(vessel_min_depth: int, enc: ENC, show_plots: bool = False) -> list:
    """Extracts the union of the relevant grounding hazards from the ENC as a list of polygons.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        - vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enc (senc.ENC): The ENC to check for grounding.
        - show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
    relevant_hazards = [enc.land.geometry.union(enc.shore.geometry).union(dangerous_seabed)]
    filtered_relevant_hazards = []
    for hazard in relevant_hazards:
        filtered_relevant_hazards.append(MultiPolygon(Polygon(p.exterior) for p in hazard.geoms if isinstance(p, Polygon)))

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


def generate_random_start_position_from_draft(enc: ENC, draft: float, min_land_clearance: float = 100.0) -> Tuple[float, float]:
    """
    Randomly defining starting easting and northing coordinates of a ship
    inside the safe sea region by considering a ship draft, with an optional land clearance distance.

    Args:
        - enc (ENC): Electronic Navigational Chart object
        - draft (float): Ship's draft in meters.
        - min_land_clearance (float): Minimum distance to land in meters.

    Returns:
        - Tuple[float, float]: Tuple of starting x and y coordinates for the ship.
    """
    depth = find_minimum_depth(draft, enc)
    safe_sea = enc.seabed[depth]
    bbox = enc.bbox

    is_safe = False
    iter_count = 0
    while not is_safe:
        easting = random.uniform(bbox[0], bbox[2])
        northing = random.uniform(bbox[1], bbox[3])
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
    for hazard in hazards:
        dist = hazard.distance(Point(x, y))
        if dist < min_dist:
            min_dist = dist
    return min_dist


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
