import numpy as np, geopandas as gpd, pandas as pd, py3dep, pystac_client, stackstac, planetary_computer
from raster_tools import Raster, general, zonal, surface
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from shapely import geometry
import pickle

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

def get_spread_pnts(geom_p,min_dist=60,area=20000,n=2):
    '''
    produces a systematic sample with a random start that assures all points are at least a specified minium distance and that at least a specfied number of observations fall within a specifed area

    parameters:

    geom_p = (poly) geometry of the polygon to be sampled
    min_dist = (float) minimum distance between sample locations
    area = (float) tile size in map units
    n = total number of observations

    returns: geodataframe of point locations

    '''
    xmin,ymin,xmax,ymax=geom_p.total_bounds
    xdif=xmax-xmin
    lng = np.sqrt(area)
    rx,ry=np.random.random(2)*lng/2
    xs=np.arange(xmin+rx,xmax,min_dist)
    ys=np.arange(ymin+ry,ymax,min_dist)
    xv,yv=np.meshgrid(xs,ys)
    xv=xv.flatten()
    yv=yv.flatten()
    xind=((xv-xmin)//lng)
    yind=((yv-ymin)//lng)
    clms=xdif//lng + 1
    cind=yind*clms+xind
    pnts=(gpd.points_from_xy(x=xv,y=yv))
    pnts_check=pnts.intersects(geom_p.unary_union)
    pnts=pnts[pnts_check]
    cind=cind[pnts_check]
    dic = {'cind':cind,'geometry':pnts}
    gdf=gpd.GeoDataFrame(dic,crs=geom_p.crs)
    return gdf.groupby('cind', group_keys=False).apply(lambda x: _samp_c(x,n))

def mosaic_stac(xr):
    return stackstac.mosaic(xr)

def get_stac_data(geo,url="https://planetarycomputer.microsoft.com/api/stac/v1",name="sentinel-2-l2a",res=30,crs=5070,**kwarg):
    '''
    gets tiled data from planetary computer as a dask backed xarray that intersects the geometry of the point, line, or polygon

    geo = (polygon) geometry bounding box (WGS84)
    url = (string) base url to planetary computer https://planetarycomputer.microsoft.com/api/stac/v1
    name = (string) catelog resource
    qry =  (dictoinary) of property values {'eo:cloud_cover':{'lt':1}}
    res = (tuple of numbers) output resolution (x,y)
    crs = (int) output crs
    datetime = (string) data time intervale e.g., one month: 2023-06, range: 2023-06-02/2023-06-17
    limit = (int) max number of items to return

    returns (xarray data array and stac item catalog)
    '''
    catalog = pystac_client.Client.open(url, modifier=planetary_computer.sign_inplace)
    srch = catalog.search(collections=name, intersects=geo, **kwarg)
    ic = srch.item_collection()
    if(len(ic.items)>0):
        xra = stackstac.stack(ic,resolution=res,epsg=crs)
        xra = mosaic_stac(xra)
    else:
        xra=None
    
    return xra,ic

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
        t1= x.sample(size)
    else:
        t1=x
    return t1

def pnts_select_kmeans(bsample,pred,k=10,n=150,proportional=True):
    '''
    produces a random sample spread across k-means classes with the same proportional size sample given a larger random sample.
    
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
    bsample['km_class']=(km.labels_)
    test=np.unique(bsample['km_class'],return_counts=True)
    if(not proportional):
        nt=int(k/np.sum(test[1]>=nt) * nt)
        cl=bsample['km_class'].values
        wsb_sample=bsample.groupby('km_class', group_keys=False).apply(lambda x: _samp_c(x,nt[x['km_class'].values[0]])) 
        
    else:
        csum=test[1].sum()
        pc=test[1]/csum
        nt=dict(zip(test[0],(pc*n).astype('int32')))
        cl=bsample['km_class'].values
        wsb_sample=bsample.groupby('km_class', group_keys=False).apply(lambda x: _samp_c(x,nt[x['km_class'].values[0]])) 
    
    return wsb_sample

def _samp_sb(x,size):
        if(x.shape[0]>size):
            t1=x.sort_values(by='km_dist').iloc[:size,:]
        else:
            t1=x
        return t1

def get_spread_balanced_sample_subarea(b_sample,pred,subarea=None,n=150,k=10,equal_n_classes=True,center=False):
    '''
    returns a well spread and balanced sample from a larger random sample using k-means clustering and centering samples on cluster classes
    b_sample=(geodataframe) larger random sample to choose sample locations from
    pred=(list[str]) list of column names that are used as predictors
    subarea=(geodataframe) the geodataframe fo the subregion to select samples from the larger region (accessible areas)
    n=(int)total number of observations to select
    k=n(int)umber of k-mean classes
    equal_n_classes=(bool)

    returns a geodataframe of chosen observations spread in multi-dimensional space and centered on k-mean clusters
    '''
    nt=int(n/k)
    if(nt<1):nt=1
    X=StandardScaler().fit_transform(b_sample[pred])
    km = KMeans(n_clusters=k).fit(X)
    b_sample['km_class']=km.labels_
    if(subarea is None):
        sl=[True]*b_sample.shape[0]
    else:
        sl=b_sample.intersects(subarea.to_crs(b_sample.crs).unary_union)

    ac_smp=b_sample[sl]
    X2=X[sl,:]
    tr=km.transform(X2)
    test=np.unique(ac_smp['km_class'],return_counts=True)
    pdic=dict(zip(test[0],test[1]/test[1].sum()))
    cl=ac_smp['km_class'].values
    pr=ac_smp['km_class'].replace(pdic)
    ac_smp['km_dist']=tr[np.arange(0,ac_smp.shape[0]),cl]
    ac_smp['km_wt']=pr
    if(equal_n_classes):
        nt=int(k/np.sum(test[1]>=nt) * nt)
        if(center):
            wsb_sample=ac_smp.groupby('km_class', group_keys=False).apply(lambda x: _samp_sb(x,nt)) 
        else:
            wsb_sample=ac_smp.groupby('km_class', group_keys=False).apply(lambda x: _samp_c(x,nt))
        
    else:
        csum=test[1].sum()
        pc=test[1]/csum
        nt=dict(zip(test[0],(pc*n).astype('int32')))
        if(center):
            wsb_sample=ac_smp.groupby('km_class', group_keys=False).apply(lambda x: _samp_sb(x,nt[x['km_class'].values[0]])) 
        else:
            wsb_sample=ac_smp.groupby('km_class', group_keys=False).apply(lambda x: _samp_c(x,nt[x['km_class'].values[0]])) 
    
    return wsb_sample
        
class eknn:
    '''
    An ensemble KNN model
    functions: create_eknn, predict
    attributes: scalar function, id column name (ids), list of predictor variables (pred), the underlying training data (_traindf), list of knn models (ens_lst) 

    '''
    def __init__(self,ens_lst=[],scaler=None,ids='',predictors=[],clbls=None):
        self.s_scaler=scaler
        self.ids=ids
        self.pred=predictors
        self._traindf=None
        self.class_lbls_=clbls

    def create_eknn(self,df,predictors,ids,nmodels=50,nobs=None):
        '''
        builds the eknn model and sets attributes

        df=(data frame) the training dataframe
        predictors=([str]) list of string column names
        ids=(str) column name of the unique ids 
        nmodels(int)=number of knn models to create
        nobs(int)=number of observations to select (with replacement) from the dataframe to train each knn model 

        returns a list of knn models (attribute ens_lst)
        '''
        self.ids=ids
        self.pred=predictors
        self._traindf=df
        self.ens_lst=[]
        self.s_scaler=StandardScaler()
        self.s_scaler.fit(df[self.pred])
        X=self.s_scaler.transform(df[self.pred])
        y=df[self.ids].values
        if(nobs is None):nobs=df.shape[0]
        sids=np.arange(0,nobs)
        for i in range(nmodels):
            ids2=np.random.choice(sids,nobs,replace=True)
            X2=X[ids2,:]
            y2=y[ids2]
            neigh = KNeighborsClassifier(n_neighbors=1)
            neigh.fit(X2,y2)
            self.ens_lst.append(neigh)
        
        return self.ens_lst
    
    
    def predict(self,df,response,categorical=False):
        '''
        Predicts model estimates for a new dataset. 
        For continuous data a mean and standard error is estimated for each observation. 
        For categorical data the most likely class, the probability of that class, and the standard error of that probability is estimated for each observation

        df=(dataframe) of values to predict
        response=(str) the column name of the response variable to estimate in the training dataset (used to reclassify observation ids to values)
        categorical=(bool) determines if the estimates are continuous or categorical

        returns array of estimates
        '''
        #need to figure out remap_dictionary from dataframe column that is wanted to be predicted
        self.class_lbls_= None
        remap_dict=dict(zip(self._traindf[self.ids],self._traindf[response]))
        def _c_func(x,cls):
            prob=np.zeros_like(cls,dtype='float')
            vls,cnt=np.unique(x,return_counts=True)
            indx=np.in1d(cls, vls).nonzero()[0]
            p=cnt/x.shape[0]
            prob[indx]=p
            se=np.sqrt(prob*(1-prob)/x.shape[0])
            return np.concatenate([prob,se],axis=0)
        
        if((len(self.ens_lst)<1) or (self.s_scaler is None)):
            return None
        else:
            X=self.s_scaler.transform(df[self.pred])
            pred_lst=[]
            for mdl in self.ens_lst:
                lbls=mdl.predict(X)
                tdf=pd.DataFrame(lbls,columns=['km_class'])
                vls=(tdf.replace(remap_dict).values).flatten()
                pred_lst.append(vls)
            
            mdf=pd.DataFrame(pred_lst,columns=df.index)
            if(categorical):
                self.class_lbls_=np.unique(self._traindf[response])
                outvls=mdf.apply(_c_func,args=[self.class_lbls_],axis=0).T.values

            else:
                m=mdf.mean().values
                s=mdf.std().values
                a=np.array([m,s])
                outvls=a.T
                

            return outvls
        
def save_mdl(mdl,outpath):
    pickle.dump(mdl,open(outpath,'wb'))
    return

def load_mdl(mdlpath):
    return pickle.load(open(mdlpath, 'rb'))
        
