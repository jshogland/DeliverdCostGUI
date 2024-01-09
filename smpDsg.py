import numpy as np
import geopandas as gpd
import py3dep
import pystac_client
import stackstac
import planetary_computer
from raster_tools import Raster, general, zonal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from shapely import geometry

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


def get_systematic_sample(geom_p, xdist, ydist):
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

def get_stac_data(geo,url="https://planetarycomputer.microsoft.com/api/stac/v1",name="sentinel-2-l2a",qry={'eo:cloud_cover':{'lt':1}},res=None,crs=5070,dt='2022-06',limit=1000):
    '''
    gets data from planetary computer

    geo = (polygon) geometry bounding box (WGS84)
    url = (string) base url to planetary computer
    name = (string) catelog resource
    qry =  (dictoinary) of property values
    res = (tuple of numbers) output resolution (x,y)
    crs = (int) output crs
    dt = (strin) data time intervale 2022/2023
    limit = (int) max number of items to return

    returns (stac items, dataframe of tiles, and xarray data array)
    '''
    catalog = pystac_client.Client.open(url, modifier=planetary_computer.sign_inplace)
    srch = catalog.search(collections=name, intersects=geo, query=qry, datetime=dt, limit=limit,)
    ic = srch.item_collection()
    df = gpd.GeoDataFrame.from_features(ic.to_dict(), crs="epsg:4326")
    xra = stackstac.stack(ic,resolution=res,epsg=crs)
    return ic, df, xra

def get_3dep_data(sgeo,res=30,out_crs=None):
    '''
    downloads 3dep data from a specified service and resolution and returns a raster object
    
    sgeo: object, polygon bounding box used to extract data (WGS 84 - EPSG:4326)
    res: int, spatial resolution
    out_crs: object, optional crs used to project geopandas dataframe to a different crs
    
    return: raster object
    '''
    out_rs=py3dep.get_dem(sgeo,res,4326).expand_dims({'band':1})
    if(not out_crs is None):
        out_rs=(out_rs.rio.reproject(out_crs))
        
    return Raster(out_rs.chunk())

def _samp_c(x,size):
    cnt=x.shape[0]
    if(cnt>=size):
        return x.sample(size)

def pnts_select_kmeans(bsample,pred,k=10,n=150):
    '''
    produces a random sample spread across k-means classes given a larger random sample
    
    parameters:
    
    bsample = (point) geodataframe of random locations (big sample)
    predictors = (str list) field names
    k = (int) number of k-means class to split predictor space up into
    n = total number of samples
    
    returns: geodataframe of point locations
    
    '''
    nt=int(n/k)
    if(nt<1):nt=1
    X=StandardScaler().fit_transform(bsample[pred])
    km=KMeans(n_clusters=k).fit(X)
    bsample['cind']=(km.labels_)
    test=np.unique(bsample.cind,return_counts=True)
    nt=int(k/np.sum(test[1]>=nt) * nt)
    return bsample.groupby('cind', group_keys=False).apply(lambda x: _samp_c(x,nt)) 

def get_stratifed_sample(geom_p,sources,month='2022-06',n=150,k=10):
    b_smp=get_random_sample(geom_p=geom_p,n=n*k)
    rslst=[]
    if('elevation' in sources):
        rslst.append(get_3dep_data(geom_p,res=30,out_crs=5070))

    if('sentinel2' in sources):
        rslst.append(get_stac_data(geom_p,dt=month))

    if('landsat' in sources):
        rslst.append(get_stac_data(geom_p,name='landsat-c2-I2',dt=month))

    if('lidar' in sources):
        rslst.append(get_stac_data(geom_p,dt=month))

    rs = general.band_concat(rslst)
    tbl=zonal.extract_points_eager(b_smp,rs,axis=1)
    b_smp=b_smp.join(tbl)
    pred=tbl.columns
    spnts=pnts_select_kmeans(b_smp,pred,k=k,n=n)
    return spnts



