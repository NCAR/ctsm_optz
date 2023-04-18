import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
import matplotlib
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import pyproj
import cmocean
import seaborn as ssn
import os
import sys

# load data
job_file = sys.argv[1]
job_list = pd.read_csv(job_file, sep=' ', header=None)
metric_output_dir = sys.argv[2]
swe_dir = sys.argv[3]
hru_shp = sys.argv[4]
sim_domain_file = sys.argv[5]
job_list.columns = ['repcase','nlnum','job_id']
max_distance = 0.125*0.125/4 * 2
thres = 0.1 # unit:mm

# process no leap date
start_date = '2004-09-01'
end_date = '2009-08-31'
time_series_noleap = []
for date in pd.date_range(start_date, end_date):
    if date.month != 2 or date.day != 29:
        time_series_noleap.append(date)

# start load observed data
df_temp_sum=pd.read_csv('/glade/work/yifanc/NNA/data/SNOTEL/snotel_temp_summary.1980-2019.F.csv',index_col=[0], parse_dates=['datetime'])
df_precip_sum=pd.read_csv('/glade/work/yifanc/NNA/data/SNOTEL/snotel_precip_summary.1980-2019.inch.csv',index_col=[0], parse_dates=['datetime'])
df_sum=pd.read_csv('/glade/work/yifanc/NNA/data/SNOTEL/snotel_SWE_summary.1980-2019.inch.csv',index_col=[0], parse_dates=['datetime'])

sntl_site_gdf=gpd.read_file('/glade/work/yifanc/NNA/data/SNOTEL/snotel_location_info.shp')
sntl_site_gdf = sntl_site_gdf.set_index('index')

# simulated domain file
domain_ds = xr.open_dataset(sim_domain_file)

## hru shape file
#usgs_subbasin = gpd.read_file(hru_shp)
#usgs_subbasin['subbasin'] = 1
#usgs_subbasin_dissolve = usgs_subbasin.dissolve(by='subbasin')
#usgs_subbasin_dissolve = usgs_subbasin_dissolve.set_crs(epsg=4326)
#sntl_site_gdf = sntl_site_gdf.set_crs(epsg=4326)

# site within our study domain
#site_within = []
#for site in sntl_site_gdf.index.values:
#    if sntl_site_gdf.loc[site].geometry.within(usgs_subbasin_dissolve.loc[1].geometry):
#        site_within.append(site)

# find the sites within the domain
landmask = domain_ds.mask.values
lat_domain = domain_ds.yc.values
lon_domain = domain_ds.xc.values
lat_corner = domain_ds.yv.values
lon_corner = domain_ds.xv.values
site_within = []
for site in sntl_site_gdf.index.values:
    lat = sntl_site_gdf['latitude'].loc[site]
    lon = sntl_site_gdf['longitude'].loc[site]%360
    distance_to_location =  (lat_domain-lat)**2 + (lon_domain-lon)**2
    [lat_ind,lon_ind] = np.unravel_index( np.argmin(distance_to_location), landmask.shape)
    
    lat_corner_sel = lat_corner[:,lat_ind,lon_ind]
    lon_corner_sel = lon_corner[:,lat_ind,lon_ind]
    if np.min(abs(lat - lat_corner_sel)) <= lat_corner_sel.max() - lat_corner_sel.min():
        if np.min(abs(lon - lon_corner_sel)) <= lon_corner_sel.max() - lon_corner_sel.min():
            print(site)
            site_within.append(site)

# check whether selected sites have data
site_within_data = []
for site in site_within:
    if site in df_sum.columns.values:
        site_within_data.append(site)

# check the selected site's data is not nan
site_w_data = np.isnan(df_sum[site_within_data].loc[slice(start_date,end_date)]).sum()
site_w_data = site_w_data[site_w_data == 0].index.values

def cal_snow_rate(df, year_second, thres):
    '''
    this is only for calcluate snow melt rate at daily timestep
    '''
    swe_in_second_year = df.loc[slice("%s-01"%(year_second),"%s-08"%(year_second))]
    date_w_max_snow = int(swe_in_second_year.argmax()) 
    date_w_no_snow = np.where(swe_in_second_year<thres)[0] 
    if len(date_w_no_snow) == 0:
        date_w_no_snow = [len(swe_in_second_year) - 1] # august-31st and the id starts from 0 so we need to subtract 1
    duration_melt = date_w_no_snow[0] - date_w_max_snow
    snow_melt_rate =(swe_in_second_year[date_w_max_snow] - swe_in_second_year[date_w_no_snow[0]])/duration_melt
    return snow_melt_rate

for job in job_list.iterrows():
    if job[1]['nlnum'] == -1:
        h2osno_file = swe_dir + '%s.clm2.h0.****-09-01-00000.nc'%(job[1]['repcase'])
        print("nlnum = %s"%(job[1]['nlnum']))
    else:
        h2osno_file = swe_dir + '%s.clm2_%s.h0.****-09-01-00000.nc'%(job[1]['repcase'], "%04i"%(job[1]['nlnum']))
    print(h2osno_file)
    ds_rasm_h2osno = xr.open_mfdataset(h2osno_file)
    ds_rasm_h2osno.H2OSNO.load()
    ds_rasm_chosen = ds_rasm_h2osno
    ######################################
    site_w_data_avail_2000s = []
    site_w_metric = []
    mape_max_snow_sum = []
    snow_melt_rate_sum = []
    snow_duration_diff_sum = []

    for site_sel in site_w_data: # ['SNOTEL:1189_AK_SNTL']:#

        # selected grid cell
        lat_sel = sntl_site_gdf.loc[site_sel]['latitude']
        lon_sel = sntl_site_gdf.loc[site_sel]['longitude']%360
        distance = (domain_ds.yc.values-lat_sel)**2 + (domain_ds.xc.values-lon_sel)**2
        # find the nearest grid cell
        lat_ind, lon_ind = np.unravel_index(np.argmin(distance),shape=distance.shape)
        if (distance[lat_ind,lon_ind] < max_distance):

            lat_grid = ds_rasm_h2osno.lat.values[lat_ind]
            lon_grid = ds_rasm_h2osno.lon.values[lon_ind]
            obs_data = df_sum[site_sel].values*25.4
            sim_data = ds_rasm_chosen.H2OSNO.sel(lat=lat_grid,lon=lon_grid, time=slice(start_date,end_date))
            
            # preprocess data
            sim_data_df = pd.DataFrame(sim_data[0:len(time_series_noleap)],index=time_series_noleap,columns=['sim'])
            obs_data_df = pd.DataFrame(obs_data, index=df_sum.index.tz_localize(None),columns=['obs'])
            df_concat = pd.concat([obs_data_df,sim_data_df],axis=1).dropna()
            if len(df_concat)>0:
                site_w_data_avail_2000s.append(site_sel)
                # start_process_data
                mape_max_snow_site = []
                snow_melt_site = []
                snow_duration_diff_site = []
                for year in range(2004,2009):
                    df_sel = df_concat.loc[slice("%s-09-01"%(year),"%s-08-31"%(year+1))]
                    if len(df_sel) > 300:
                        # calculate the maximum absolute percentage error
                        max_snow = np.max(df_sel)
                        mape_max_snow = (max_snow['sim'] - max_snow['obs'])/max_snow['obs']
                        mape_max_snow_site.append(mape_max_snow)

                        # snow melt rate
                        melt_rate_obs = cal_snow_rate(df_sel.loc[:,'obs'],year+1,thres)
                        melt_rate_sim = cal_snow_rate(df_sel.loc[:,'sim'],year+1,thres)
                        melt_diff = (melt_rate_sim - melt_rate_obs)/melt_rate_obs
                        snow_melt_site.append(melt_diff)

                        # difference of snow duration period
                        df_obs = df_sel[['obs']]
                        df_sim = df_sel[['sim']]
                        snow_duration_obs = (df_obs[df_obs['obs']>=thres].index[-1] - df_obs[df_obs['obs']>=thres].index[0])/np.timedelta64(1,'D')
                        snow_duration_sim = (df_sim[df_sim['sim']>=thres].index[-1] - df_sim[df_sim['sim']>=thres].index[0])/np.timedelta64(1,'D')
                        snow_duration_diff = ((snow_duration_sim - snow_duration_obs))/snow_duration_obs
                        snow_duration_diff_site.append(snow_duration_diff)

                if len(mape_max_snow_site)>0:
                    site_w_metric.append(site_sel)
                    mape_max_snow_sum.append(np.median(mape_max_snow_site))
                    snow_duration_diff_sum.append(np.median(snow_duration_diff_site))
                    snow_melt_rate_sum.append(np.median(snow_melt_site))
                    print(site_sel,snow_melt_site)

        else:
            print("%s,%s is not in our study domain. Their lat/lon ind is %s, %s"%(lat_sel,lon_sel,lat_ind,lon_ind))
    df_result = pd.DataFrame(np.transpose([mape_max_snow_sum, snow_melt_rate_sum,snow_duration_diff_sum]),index=site_w_metric,
                             columns=['mape_max_swe','snow_melt_rate','median_bias_swe_duration_Day'])
    metric = np.sqrt((df_result**2).sum(axis=1)).median()
    print(metric)
    df_result.to_csv(metric_output_dir+"metric.swe.%s.csv"%(job[1]['job_id']))
