import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
import matplotlib
import hydroeval as he
import geopandas as gpd
import os
import sys

# load data
job_file = sys.argv[1]
job_list = pd.read_csv(job_file, sep=' ', header=None)
metric_output_dir = sys.argv[2]
flow_dir = sys.argv[3]
mapping_file = sys.argv[4]
print("1", job_file)
print("2", metric_output_dir)
print("3", flow_dir)
print("4", mapping_file)

job_list.columns = ['repcase','nlnum','job_id']
cfs_2_cms = 0.028316847

# process no leap date
start_date = '2004-09-01'
end_date = '2009-08-31'
time_series_noleap = []
for date in pd.date_range(start_date, end_date):
    if date.month != 2 or date.day != 29:
        time_series_noleap.append(date)

# process observed data
df_id = pd.read_csv(mapping_file,index_col=[0])
input_dir = '/glade/work/yifanc/NNA/data/USGS/flow/'
df_flow_obs = pd.DataFrame([])
for site in df_id.index.values:
    flow_file = input_dir + '%s.flow.csv'%(site)
    if os.path.isfile(flow_file):
        df_flow = pd.read_csv(flow_file, index_col=[0], parse_dates=[0])
        df_flow = df_flow[df_flow['qualifiers']=='A'][["%s"%site]]
        df_flow_obs = pd.concat([df_flow_obs, df_flow],axis=1)

# start calculating objective functions
for job in job_list.iterrows():
    # start read sim file
    if job[1]['nlnum'] == -1:
        flow_file = flow_dir + '%s.h.****-**-01-00000.nc'%(job[1]['repcase'])
    else:
        flow_file = flow_dir + '%s.clm2_%s.h.****-**-01-00000.nc'%(job[1]['repcase'], "%04i"%(job[1]['nlnum']))
    ds_rasm_flow = xr.open_mfdataset(flow_file)
    ds_rasm_flow.load()
    df_metric_sum = pd.DataFrame()
    for site_sel in df_id.index.values: # ['SNOTEL:1189_AK_SNTL']:#
        seg_sel = df_id.loc[site_sel]['segID']
        sim_flow = ds_rasm_flow.IRFroutedRunoff.sel(seg=seg_sel, time=slice(start_date,end_date))/cfs_2_cms
        sim_df = pd.DataFrame(sim_flow, index=time_series_noleap, columns=['sim'])
        obs_df = df_flow_obs[["%s"%site_sel]]
        obs_df.columns = ['obs']
        df_concat = pd.concat([sim_df,obs_df],axis=1)
        df_concat = df_concat.dropna()
        print(df_concat)
        if len(df_concat)>0:
            kge=he.kge(df_concat['sim'].values, df_concat['obs'].values)[0][0]
#             df_concat.groupby(df_concat.index.month).mean().plot()
            df_metric = pd.DataFrame(np.concatenate([df_concat.mean().values,[kge]]).T,index=['sim_flow_cfs','obs_flow_cfs','kge'], columns=[site_sel])
            df_metric_sum = pd.concat([df_metric_sum,df_metric],axis=1)

    metric_file = metric_output_dir + "metric.flow.%s.csv"%(job[1]['job_id'])
    if 'all' not in job_file: # only output one kge value
        median_kge = np.median(df_metric_sum.loc['kge'].values)
        file1 = open(metric_file,"w")#write mode 
        file1.write("kge_daily,%.8f"%(median_kge)) 
        file1.close()
    else:
        df_metric_sum.to_csv(metric_file)
