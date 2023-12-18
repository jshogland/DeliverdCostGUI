#import raster_tools and modules 
from raster_tools import Raster, surface, distance, open_vectors, creation, Vector
import os, time
import geopandas as gpd
import numpy as np
from dask.diagnostics import ProgressBar
from shapely.geometry import box 
import osmnx as ox
import pandas
import numpy as np
import py3dep

import warnings

#turn warnings off
warnings.filterwarnings("ignore")

#specify paths to data layers default demo
study_area_path = None
lyr_sawmill_path = None
lyr_roads_path = None
lyr_barriers_path = None


#transportation speed
h_speed={'residential':25,'unclassified':15,'tertiary':35,'secondary':45,'primary':55,'trunk':55,'motorway':65}
#mtfcc_dic={'S1400':40,'S1200':56,'S1100':88}

#extraction rates of travel
sk_r=2.44
cb_r=3.35

#component rates
sk_d=165
cb_d=400

fb_d=15
hf_d=27
pr_d=56
lt_d=98
ht_d=2470
pf_d=2470

#payloads
sk_p=1.25
cb_p=1.04

lt_p=12.25

#optional
cb_o=False
pbar=None
log=None

def get_osm_data(sgeo,osm_dic={'highway':['motorway','trunk','primary','secondary','tertiary','unclassified','residential']},out_crs=None):
    '''
    downloads openstreetmaps data for a specified dictionary of layers and returns a geopandas dataframe
    
    sgeo: object, polygon bounding box used to extract data (WGS 84 - EPSG:4326)
    osm_dic: dictionary, dictionary of data types and resources
    out_crs: object, optional crs used to project geopandas dataframe to a differnt crs
    
    return: geopandas dataframe
    '''
    out_gdf=ox.features_from_polygon(sgeo,osm_dic)
    if(not out_crs is None):
        out_gdf=out_gdf.to_crs(out_crs)
    return out_gdf


def get_3dep_data(sgeo,res=30,out_crs=None):
    '''
    downloads 3dep data from a specified service and resolution and returns a raster object
    
    sgeo: object, polygon bounding box used to extract data (WGS 84 - EPSG:4326)
    res: int, spatial resolution
    out_crs: object, optional crs used to project geopandas dataframe to a differnt crs
    
    return: raster object
    '''
    out_rs=py3dep.get_dem(sgeo,res,4326).expand_dims({'band':1})
    if(not out_crs is None):
        out_rs=(out_rs.rio.reproject(out_crs))
        
    return Raster(out_rs.chunk())

def _remove_file(path):
    if os.path.exists(path): os.remove(path)
    return
     
def _run():
    #get the vectors and rasters objects
    warnings.simplefilter("ignore")
    print("Reading the data")
    if not pbar is None: pbar.value=pbar.value+1

    saw=open_vectors(lyr_sawmill_path).data.compute()
    s_area=open_vectors(study_area_path).data.compute()
    
    #create bounding box poly
    ext=saw.union(s_area.unary_union)
    ext_wgs84=ext.to_crs('EPSG:4326').buffer(0.15)
    ply=box(*ext_wgs84.total_bounds)
    
    #get osm data
    osm_rds={'highway':['motorway','trunk','primary','secondary','tertiary','unclassified','residential']}
    osm_strms={'waterway':['river','stream','cannel','ditch']}
    osm_waterbody={'water':['lake','reservoir','pond']}

    print("Getting Road Data...")
    if not pbar is None: pbar.value=pbar.value+1
    if(lyr_roads_path is None):
        rds=get_osm_data(ply,osm_rds,out_crs=s_area.crs).reset_index()
    else:
        rds=open_vectors(lyr_roads_path).data.compute()
    if(lyr_barriers_path is None):
        print("Getting Stream Data...")
        strms=get_osm_data(ply,osm_strms,out_crs=s_area.crs).reset_index()
        print("Getting Water Body Data...")
        wtrbd=get_osm_data(ply,osm_waterbody,out_crs=s_area.crs).reset_index()
    else:
        pass

    #project all data into 5070
    rds=rds.to_crs(5070)
    strms=strms.to_crs(5070)
    wtrbd=wtrbd.to_crs(5070)
    saw=saw.to_crs(5070)
    s_area=s_area.to_crs(5070)
        
    print("Getting Elevation Data...")
    if not pbar is None: pbar.value=pbar.value+1
    elv=get_3dep_data(ply,30,out_crs=s_area.crs)
    
    #set speed for road segments
    print("Subsetting and attributing the data")
    if not pbar is None: pbar.value=pbar.value+1
    rds['speed']=rds['highway'].map(h_speed)
    tms=rds.maxspeed.str.slice(0,2)
    tms=tms.where(tms.str.isnumeric(),24).astype(float)
    rds['speed'].where(rds['maxspeed'].isna(),tms)
    rds['conv']=2*(((1/(rds['speed']*1609.344))*lt_d)/lt_p) #1609.344 converts miles per hour to meters per hour
    
    #snap biomass and sawmill facilities to road vertices
    print("Snapping sawmills to roads")
    if not pbar is None: pbar.value=pbar.value+1
    tmp_rds=rds
    tmp_rds_seg=tmp_rds.sindex.nearest(saw.geometry,return_all=False)[1]
    lns=tmp_rds.iloc[tmp_rds_seg].geometry.values
    saw['cline']=lns
    saw['npt']=saw.apply(lambda row: row['cline'].interpolate(row['cline'].project(row['geometry'])),axis=1)#, result_type = 'expand')
    saw=saw.set_geometry('npt').set_crs(saw.crs)
              
    #create barriers to off road skidding
    if(lyr_barriers_path is None):
        strm_b=strms[strms['intermittent'].isna()].buffer(30)
        wb_b=wtrbd.buffer(30)
        
        barv=gpd.GeoDataFrame(geometry=pandas.concat([strm_b,wb_b]),crs=rds.crs)
    else:
        barv=open_vectors(lyr_barriers_path).compute()
        
    bar2=Vector(barv).to_raster(elv,all_touched=True).set_null_value(None) < 1
    
    # create slope and road distance surfaces
    print("Creating base layers for thresholding")
    if not pbar is None: pbar.value=pbar.value+1
    slp = surface.slope(elv,degrees=False).eval() #compute so that slope only needs to be calculated once
    c_rs = creation.constant_raster(elv).set_null_value(0) #constant value of 1 to multiply by distance
    rds_rs = (Vector(rds).to_raster(elv,'conv').set_null_value(0)).eval() #source surface with all non-road cells (value of zero) set to null
    
    # convert transportation rates and payload into on road cost surface that can be multiplied by the surface distance along a roadway to estimate hauling costs
    print("Calculating on road hauling costs")
    if not pbar is None: pbar.value=pbar.value+1
    saw_rs=(Vector(saw).to_raster(elv).set_null_value(0)).eval()
    on_d_saw = distance.cda_cost_distance(rds_rs,saw_rs,elv)
    
    # convert transportation surfaces to source surfaces measured in cents / CCF
    src_saw = (on_d_saw * 100).astype(int)
    
    # create extraction surface distance surfaces that can be multiplied by rates to estimate dollars per unit
    print("Calculating extraction costs")
    if not pbar is None: pbar.value=pbar.value+1

    #barriers to motion
    b_dst_cs2=bar2.set_null_value(0) # skidding and cable
    
    #calc distance
    saw_d,saw_t,saw_a=distance.cost_distance_analysis(b_dst_cs2,src_saw,elv)

    # Transportation, extraction, Felling, processing costs, and Additional Treatments
    print("Calculating additional felling, processing, and treatment costs")
    if not pbar is None: pbar.value=pbar.value+1
    f1=slp<=0.35
    fell=(f1*fb_d).where(f1,hf_d)
    prc=creation.constant_raster(elv,pr_d).astype(float)
    oc=fell+prc

    #Additional treatment costs (per/acre)
    ht_cost=creation.constant_raster(elv,(ht_d*0.222395)).astype(float) #0.222395 acres per cell
    pf_cost=creation.constant_raster(elv,(pf_d*0.222395)).astype(float) #0.222395 acres per cell

    # Convert extraction rates to a multiplier that can be used to calculate dollars per ccf given distance
    print("Combining costs...")  
    if not pbar is None: pbar.value=pbar.value+1
    s_c= (2 * (((1/(sk_r*1000))*sk_d)/sk_p))
    c_c= (2 * (((1/(cb_r*1000))*cb_d)/cb_p))

    # Calculate potential saw costs $/CCF
    sk_saw_cost=(saw_d * s_c) + (saw_a/100) + oc
    cb_saw_cost=(saw_d * c_c) + (saw_a/100) + oc

    # Operations
    rd_dist=distance.cda_cost_distance(c_rs,(rds_rs>0).astype(int),elv)

    sk=f1 & (rd_dist<460)
    cb=(~f1 & (rd_dist<305))*2
    opr=sk+cb

    #saving rasters
    outdic={}
    print('Saving default rasters...')
    if not pbar is None: pbar.value=pbar.value+1
    o1=opr==1
    o2=opr==2
    sc1=sk_saw_cost*o1
    sc2=cb_saw_cost*o2
    saw_cost=sc1+sc2
    saw_cost=saw_cost.where(saw_cost>=0,np.nan)
    _remove_file('d_cost.tif')
    saw_cost.save('d_cost.tif')
    outdic['Delivered Cost']='d_cost.tif'
    add_tr_fr_cost=ht_cost+pf_cost
    _remove_file('a_cost.tif')
    add_tr_fr_cost.save('a_cost.tif')
    outdic['Additional Treatment Cost']='a_cost.tif'
       
    if (cb_o==True):
        print('Saving optional rasters saw, bio, additional cost surfaces, operation surface')
        if not pbar is None: pbar.value=pbar.value+1
        
        _remove_file('skidder_cost.tif')
        sk_saw_cost.save('skidder_cost.tif')
        outdic['Skidder Cost']='skidder_cost.tif'

        _remove_file('cable_cost.tif')
        cb_saw_cost.save('cable_cost.tif')
        outdic['Cable Cost']='cable_cost.tif'                

        _remove_file('hand_treatment_costs.tif')
        ht_cost.save('hand_treatment_costs.tif')
        outdic['Hand Treatment Cost']='hand_treatment_costs.tif'

        _remove_file('prescribed_fire_costs.tif')
        pf_cost.save('prescribed_fire_costs.tif')
        outdic['Prescribed Fire Cost']='prescribed_fire_costs.tif'

        _remove_file('potential_harv_system.tif')
        opr.save('potential_harv_system.tif')
        outdic['Potential Harvesting System'] = 'potential_harv_system.tif'
    
    if not pbar is None: pbar.value=pbar.max
    
    return outdic

def run():
    '''
    runs the delivered cost routine and returns the path to [sawlog_cost,additional_treatment_cost, and optional
    *skidder_saw_cost,cable_saw_cost,hand_treatment_costs,prescribed_fire_costs,potential_harv_system]
    '''
    st=time.time()
    with ProgressBar():
        outdic=_run()
    
    et=time.time()
    print('Total processing time = ' + str(et-st))
    return outdic