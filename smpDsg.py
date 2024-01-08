import numpy as np
import geopandas as gpd

from shapely import geometry

def get_fishnet(geom_p, lng):
    """
    tiles the bounds of the geometry into squares

    parameters:

    geom_p = (poly) geometry of the polygon to be sampled
    lng = (float) length of each square's side (map units)

    returns: geodataframe of tiles (polygons)

    """
    xmin, ymin, xmax, ymax = geom_p.total_bounds
    x, y = (xmin, ymin)
    geom_lst = []
    while y <= ymax:
        while x <= xmax:
            geom = geometry.Polygon(
                [
                    (x, y),
                    (x, y + lng),
                    (x + lng, y + lng),
                    (x + lng, y),
                    (x, y),
                ]
            )
            geom_lst.append(geom)
            x += lng

        x = xmin
        y += lng

    return gpd.GeoDataFrame(geom_lst, columns=["geometry"], crs=geom_p.crs)


def _samp_c(x, size):
    cnt = x.shape[0]
    if cnt >= size:
        return x.sample(size)


def get_tiled_pnts(geom_p, min_dist=60, area=20000, n=2):
    """
    produces a systematic sample with a random start that assures all points are at least a specified minium distance and that at least a specified number of observations fall within a specifed area

    parameters:

    geom_p = (poly) geometry of the polygon to be sampled
    min_dist = (float) minimum distance between sample locations
    area = (float) tile size in map units
    n = total number of observations

    returns: geodataframe of point locations

    """
    xmin, ymin, xmax, ymax = geom_p.total_bounds
    xdif = xmax - xmin
    lng = np.sqrt(area)
    rx, ry = np.random.random(2) * lng / 2
    xs = np.arange(xmin + rx, xmax, min_dist)
    ys = np.arange(ymin + ry, ymax, min_dist)
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.flatten()
    yv = yv.flatten()
    xind = (xv - xmin) // lng
    yind = (yv - ymin) // lng
    clms = xdif // lng + 1
    cind = yind * clms + xind
    pnts = gpd.points_from_xy(x=xv, y=yv)
    pnts_check = pnts.intersects(geom_p.unary_union)
    pnts = pnts[pnts_check]
    cind = cind[pnts_check]
    dic = {"cind": cind, "geometry": pnts}
    gdf = gpd.GeoDataFrame(dic, crs=geom_p.crs)
    return gdf.groupby("cind", group_keys=False).apply(lambda x: _samp_c(x, n))


def get_random_sample(geom_p, n=1000):
    """
    produces a random sample given a geometry

    parameters:
    geom_p = (polygon) project polygon
    n = number of observations

    returns: geodataframe of point locations

    """
    xmin, ymin, xmax, ymax = geom_p.total_bounds
    xdif = xmax - xmin
    ydif = ymax - ymin
    pnts_lst = []
    while len(pnts_lst) < n:
        x = (np.random.random() * xdif) + xmin
        y = (np.random.random() * ydif) + ymin
        pnt = geometry.Point([x, y])
        if pnt.intersects(geom_p).values:
            pnts_lst.append(pnt)

    dic = {"geometry": pnts_lst}
    gdf = gpd.GeoDataFrame(dic, crs=geom_p.crs)

    return gdf


def get_systematic_sample(geom_p, smpspcx, ydisty):
    """
    produces a systematic random sample

    parameters:
    geom_p = (polygon) project polygon
    xdist = distance in easting (lon) between observations
    ydist = distance in northing (lat) between observations

    returns: geodataframe of point locations
    """
    xmin, ymin, xmax, ymax = geom_p.total_bounds
    rx = np.random.random() * xdist / 2
    ry = np.random.random() * ydist / 2
    xs = np.arange(xmin + rx, xmax, xdist)
    ys = np.arange(ymin + ry, ymax, ydist)
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.flatten()
    yv = yv.flatten()
    pnts = gpd.points_from_xy(x=xv, y=yv)
    pnts_check = pnts.intersects(geom_p.unary_union)
    pnts = pnts[pnts_check]
    dic = {"geometry": pnts}
    gdf = gpd.GeoDataFrame(dic, crs=geom_p.crs)
    return gdf