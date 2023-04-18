from __future__ import division, print_function, absolute_import
import sys
sys.path.append('/glade/work/yifanc/code/optmz/MO-ASMO/src/')
import numpy as np
import sampling
import gp
import NSGA2
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from os import path
import pickle
import smt
from sklearn import preprocessing
import random
import re
from sklearn.metrics import mean_squared_error
import matplotlib
import seaborn as sn
matplotlib.use('Agg')

metric_list = '/glade/work/yifanc/NNA/optmz/surrogate_model/metrics/'
namelist_file = '/glade/work/yifanc/NNA/optmz/baseline/basin/hs_ds_pe_4km_tanana/CaseDocs/lnd_in'
#basepftfile = '/glade/p/cesmdata/cseg/inputdata/lnd/clm2/paramdata/clm50_params.c210507.nc'
basepftfile = '/glade/p/cesmdata/cseg/inputdata/lnd/clm2/paramdata/clm50_params.c210607.nc'

# define the iteration time
init_iteration = sys.argv[1]
init_iteration = int(init_iteration)
location="pe_basin"
iteration=init_iteration + 1

# define the number of parameters in each updated run
N_resample = 20
num_init_sampling = 200
evaluate_metric=['kge_daily','swe_metric']
nOutput = len(evaluate_metric)
sheetname = 'T14'
# define hyper parameters
pop = 100
gen = 100
crossover_rate = 0.9
mu = 20
mum = 20

# define hyperparameter
alpha = 1e-3
leng_lb = 1e-3
leng_ub = 1e3
nu = 2.5

#baseline
baseline_dir = '/glade/work/yifanc/NNA/optmz/surrogate_model/metrics/baseline/'
basin_list = ['talkeetna','salcha','beaver']
num_basin = len(basin_list)


metric_baseline = pd.DataFrame()
swe_metric_b_sum = pd.DataFrame([])
flow_metric_b_sum = pd.DataFrame([])
for basin in basin_list:
    flow_metric_b = pd.read_csv(baseline_dir + 'metric.flow.hs_ds_pe_OM_4km_baseline_%s.csv'%(basin), header=None, index_col=[0])
    flow_metric_b.columns=[basin]
    flow_metric_b_sum = pd.concat([flow_metric_b_sum,flow_metric_b.T])
    swe_metric_b = pd.read_csv(baseline_dir + 'metric.swe.hs_ds_pe_OM_4km_baseline_%s.csv'%(basin), index_col=[0])
    swe_metric_cal = np.sqrt(pow(swe_metric_b,2).sum(axis=1))
    swe_metric_b_sum = pd.concat([swe_metric_b_sum,swe_metric_cal])
metric_baseline_mean = pd.DataFrame([flow_metric_b_sum.mean()[0],swe_metric_b_sum.mean()[0]],index=[evaluate_metric], columns=['baseline'])
metric_baseline_mean_norm = metric_baseline_mean.copy()
metric_baseline_mean_norm.loc['kge_daily','baseline'] = 1 - metric_baseline_mean_norm.loc['kge_daily','baseline']

# read the job list
param_df = pd.DataFrame([])
job_id_list = np.array([])

for iter_ in range(0, init_iteration+1):
    job_id_file = '/glade/work/yifanc/NNA/optmz/workflow/pe_basin_%s.main_run.txt'%(iter_)
    param_pertubed_file = '/glade/work/yifanc/NNA/optmz/workflow/pe_basin_%s.param_list.txt'%(iter_)
    # read the file
    param_df_temp = pd.read_csv(param_pertubed_file,index_col=[0])
    job_id_df = pd.read_csv(job_id_file,header=None)
    job_id_list_temp = job_id_df[0].values    
    if iter_ == 0:
        num_init_jobs = len(job_id_list_temp)
    # concat parameter file and job id list
    param_df = pd.concat([param_df,param_df_temp])
    job_id_list = np.concatenate([job_id_list, job_id_list_temp])
num_param = param_df.shape[1]

# calculate y
metric_mean = pd.DataFrame([])
init_job_list = []
optimized_job_list = []
soil_param_list = ['om_frac_sf','slopebeta']
for i,job_id in enumerate(job_id_list):
    swe_metric_sum = pd.DataFrame([])
    flow_metric_sum = pd.DataFrame([])
    num_flow_metric_avail = 0
    for basin in basin_list:
        flow_file = metric_list+'metric.flow.%s_%s.csv'%(job_id,basin)
        if path.isfile(flow_file):        
            flow_metric = pd.read_csv(flow_file, header=None, index_col=[0])
            flow_metric.columns=[basin]
            flow_metric_sum = pd.concat([flow_metric_sum,flow_metric.T])
            swe_metric = pd.read_csv(metric_list+'metric.swe.%s_%s.csv'%(job_id,basin), index_col=[0])
            swe_metric_cal = np.sqrt(pow(swe_metric,2).sum(axis=1))
            swe_metric_sum = pd.concat([swe_metric_sum,swe_metric_cal])
            num_flow_metric_avail += 1
        else:
            print(basin, job_id, np.round(param_df.loc[job_id,soil_param_list].values.astype(np.float),decimals=3))
            flow_metric = pd.DataFrame([np.NaN], columns=['kge_daily'], index=[basin])
            flow_metric_sum = pd.concat([flow_metric_sum,flow_metric])
    if num_flow_metric_avail == num_basin: 
        if (i<num_init_jobs):
            init_job_list.append(job_id)
        else:
            optimized_job_list.append(job_id)

    metric_exp = pd.DataFrame([[flow_metric_sum.mean(skipna=False)[0]],[swe_metric_sum.mean(skipna=False)[0]]], index=['kge_daily','swe_metric'],columns=[job_id])
    metric_mean = pd.concat([metric_mean,metric_exp],axis=1)
# filter non-existing metrics
for metric in evaluate_metric:
    num_nan = np.sum(np.isnan(metric_mean.loc[metric]))
    if num_nan == metric_mean.shape[1]:
        metric_mean = metric_mean.drop(metric)
        evaluate_metric.remove(metric)

nOutput = len(evaluate_metric)

# normalize the metrics
norm_metric_mean = metric_mean.copy()
norm_metric_mean.loc['kge_daily',:] = 1 - norm_metric_mean.loc['kge_daily',:]
norm_metric_mean = norm_metric_mean.dropna(axis=1)
print("shape of the norm_metric_mean is: ", norm_metric_mean.shape)
print("shape of the metric_mean is: ", metric_mean.shape)
num_init_sampling_remining = len(init_job_list)

# plot the CTSM-ASMO performance
start_id = num_init_sampling_remining + (init_iteration - 1)*N_resample
figure_dir = "/glade/work/yifanc/NNA/optmz/figure/"
if init_iteration > 0:
    if nOutput == 2:
        plt.figure(figsize=[5,5],dpi=150)
        plt.scatter(norm_metric_mean.iloc[0,start_id:],norm_metric_mean.iloc[1,start_id:],label='optimized situation')
        plt.scatter(1-metric_baseline_mean.loc['kge_daily','baseline'],metric_baseline_mean.loc['swe_metric','baseline'],label='baseline situation')
        # plt.xlim([0.4,0.8])
        # plt.ylim([0.4,0.8])
        plt.xlabel(evaluate_metric[0])
        plt.ylabel(evaluate_metric[1])
        plt.legend()
        plt.savefig(figure_dir + 'simulated_pareto_surface.iter_%s.png'%(init_iteration),dpi=200,bbox_inches='tight')
    if nOutput == 1:
        plt.figure(figsize=[5,5],dpi=150)
        kge_daily_b = 1-metric_baseline_mean.loc['kge_daily','baseline']
        sn.distplot(norm_metric_mean.iloc[0,start_id:],label='optimized')
        fig = plt.gca()
        ylim = fig.get_ylim()
        plt.plot([kge_daily_b,kge_daily_b],ylim, label='default')
        plt.ylim(ylim)
        plt.xlabel('flow objective (smaller better)')
        plt.legend()
        plt.savefig(figure_dir + 'simulated_pareto_surface.iter_%s.png'%(init_iteration),dpi=200,bbox_inches='tight')

# get the upper and lower bound of the parameter files
f = open(namelist_file, "r")
namelist_f = pd.DataFrame([],columns=['value'])
for x in f:
    string_list = x.split(sep='=')
    st_list = []
    for st in string_list:
        st_list.append(st.strip())
    if len(st_list) == 2:
        df = pd.DataFrame([st_list[1]],columns=['value'],index=[st_list[0]])
        namelist_f = pd.concat([namelist_f,df],axis=0)
        
# load parameter name and range
pf_excel = '/glade/work/yifanc/NNA/optmz/paramfile/param_selection/sensitive_parameters_in_arctic_hydrology.xlsx'
pf = pd.ExcelFile(pf_excel)
pf = pf.parse(sheetname)
#pf = pf[pf['whether_asmo']=='y']
pf = pf[(pf['whether_asmo']=='y')&(pf['only_in_fates']=='no')]
nInput = len(pf)
file_location = pf['location'].values
N_var_list = pf[pf['location']=='N']['Parameters'].values
P_var_list = pf[pf['location']=='P']['Parameters'].values
# define variable name for specific variables
var_min='min_asmo'
var_max='max_asmo'
var_pft_min = 'pft_mins'
var_pft_max = 'pft_maxs'
var_param = 'Parameters'
pf[var_min] = pf[var_min].astype(str)
pf[var_max] = pf[var_max].astype(str)
pft_type_sel = np.arange(79)#np.array([2,11,12]).astype(np.int32)
# read in parameter files
param_ds = xr.open_dataset(basepftfile)

# calculate the parameter bound
vmin_list = []
vmax_list = []
for row in pf.iterrows():
    param_name = row[1][var_param]
    if row[1]['location'] == 'N':
        default_value = namelist_f.loc[param_name]
        if type(default_value) is str:
            default_value = float(default_value.split('d')[0])
        if 'percent' in row[1][var_min]:
            pct_min = float(row[1][var_min].split('percent')[0])
            pct_max = float(row[1][var_max].split('percent')[0])
            vmin_list.append(default_value*(1-pct_min/100))
            vmax_list.append(default_value*(1+pct_max/100))
        else:
            vmin_list.append(float(row[1][var_min]))
            vmax_list.append(float(row[1][var_max]))
    else:
        if row[1]['location'] == 'P':
            default_value = param_ds[param_name].values
            if 'percent' in row[1][var_min]:
                pct_min = float(row[1][var_min].split('percent')[0])
                pct_max = float(row[1][var_max].split('percent')[0])
                vmin_list.append(default_value*(1-pct_min/100))
                vmax_list.append(default_value*(1+pct_max/100))
            else:
                if row[1][var_min] == 'pft':
                    pft_mins = row[1][var_pft_min]
                    pft_maxs = row[1][var_pft_max]
                    pft_mins = np.array(pft_mins.split(',')).astype(np.float64)
                    pft_maxs = np.array(pft_maxs.split(',')).astype(np.float64)
                    vmin_list.append(pft_mins[pft_type_sel])
                    vmax_list.append(pft_maxs[pft_type_sel])
#                     print(pft_mins[pft_type_sel], pft_maxs[pft_type_sel])
                else:
                    if type(default_value) == np.ndarray:
                        vmin_list.append(np.full(default_value.shape,float(row[1][var_min])))
                        vmax_list.append(np.full(default_value.shape,float(row[1][var_max])))
                    else:
                        vmin_list.append(float(row[1][var_min]))
                        vmax_list.append(float(row[1][var_max]))
pf['vmin_value'] = vmin_list
pf['vmax_value'] = vmax_list
xlb = pf['vmin_value'].values
xub = pf['vmax_value'].values

# make bound into single value
xlb_single_value = []
xub_single_value = []
for i_param,param in enumerate(pf[var_param].values):
    lb = xlb[i_param]
    ub = xub[i_param]
    if type(lb) is not np.float64:
        xlb_single_value.append(np.mean(lb))
        xub_single_value.append(np.mean(ub))
    else:
        xlb_single_value.append(lb)
        xub_single_value.append(ub)
xub_single_value = np.array(xub_single_value).astype(np.float64)
xlb_single_value = np.array(xlb_single_value).astype(np.float64)

# prepare training data
param_train_df = param_df.copy()
index_list = param_df.index.values
match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *\-?\+?\ *[0-9]+)?')
for var in param_df.columns.values:
    if type(param_df.loc[index_list[0],var]) == str:
        v_list = []
        for index in index_list:
            my_list = [float(x) for x in re.findall(match_number, param_df.loc[index,var])]
#            my_list = np.array(re.findall(r'\d+\.\d+|\d+\.', param_df.loc[index,var])).astype(np.float32)
#            if len(my_list) == 0:
#                my_list = np.array(re.findall(r'\d+\.', param_df.loc[index,var])).astype(np.float32)
            v_list.append(np.mean(my_list))
        param_train_df[var] = v_list
# calculate normalization scalar
param_init_df = param_train_df.iloc[0:num_init_sampling_remining]
d = preprocessing.normalize(param_init_df,axis=0,return_norm=True)
normalization_scalar = d[1]
# normalize the training data
scaled_df = param_train_df/normalization_scalar
# normalize the bounds 
xlb_single_value_scaled = xlb_single_value/normalization_scalar
xub_single_value_scaled = xub_single_value/normalization_scalar

## =============== ASMO ============== ##
# start training the surrogate models
available_id_list = norm_metric_mean.columns.values
x = scaled_df.loc[available_id_list].values #param_df.values #.shape[0]
y = norm_metric_mean.loc[evaluate_metric].T.values #norm_metric_mean.T.values
print(x.shape, y.shape, nInput, nOutput, x.shape[0],xlb_single_value_scaled, xub_single_value_scaled)

# check the performance of the simulated metrics with the predicted metrics
# we only need to do this when init_iteration > 0
sm_init_file = '/glade/work/yifanc/NNA/optmz/surrogate_model/sm/%s.iter_%s.sm'%(location, "%i"%(init_iteration-1))
if init_iteration>0:
    sm_init = pickle.load(open(sm_init_file,"rb"))
    predict_metric_old = sm_init.predict(x[start_id:,]).T
    # plotting
    for i_metric,metric in enumerate(evaluate_metric):
        simulated_metric = norm_metric_mean.iloc[i_metric,start_id:].values
        predicted_metric = predict_metric_old[i_metric]
        rmse = mean_squared_error(simulated_metric, predicted_metric,squared=False)
        plt.figure(figsize=[5,5],dpi=150)
        plt.scatter(simulated_metric, predicted_metric)
        plt.xlim([0.2,1])
        plt.ylim([0.2,1])
        plt.plot([0,1],[0,1])
        plt.xlabel('Simulated %s'%(metric))
        plt.ylabel('Predicted %s'%(metric))
        plt.text(0.7,0.22,'RMSE=%s'%("%.2f"%(rmse)))
        plt.savefig('/glade/work/yifanc/NNA/optmz/figure/compare_pareto_surface_predicted_simulated.iter_%s.%s.png'%(init_iteration-1,metric),dpi=200,bbox_inches='tight')

# start train the surrogate model
sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb_single_value_scaled, xub_single_value_scaled, alpha=alpha, leng_sb=[leng_lb,leng_ub], nu=nu)
# write the model out
sm_filename='/glade/work/yifanc/NNA/optmz/surrogate_model/sm/pe_basin.iter_%s.sm'%(init_iteration)
pickle.dump(sm, open(sm_filename, 'wb'))

# perform optimization using the surrogate model
bestx_sm, besty_sm, x_sm, y_sm = \
    NSGA2.optimization(sm, nInput, nOutput, xlb_single_value_scaled, xub_single_value_scaled, \
                       pop, gen, crossover_rate, mu, mum)
D = NSGA2.crowding_distance(besty_sm)
idxr = D.argsort()[::-1][:N_resample]
x_resample = bestx_sm[idxr,:]
y_resample = np.zeros((N_resample,nOutput))

# plot the predicted pareto surface
y_predicted = sm.predict(x_resample)
# plt.scatter(y.T[0], y.T[1],c='r')
if nOutput == 2:
    plt.figure(figsize=[5,5],dpi=150)
    plt.scatter(y_predicted.T[0],y_predicted.T[1],label='pareto surface')
    plt.xlabel(evaluate_metric[0])
    plt.ylabel(evaluate_metric[1])
    plt.scatter(metric_baseline_mean_norm.loc['kge_daily','baseline'],metric_baseline_mean_norm.loc['swe_metric','baseline'], label='baseline')
    plt.savefig(figure_dir + 'predicted_pareto_surface.iter_%s.png'%(init_iteration),dpi=200,bbox_inches='tight')
if nOutput == 1:
    plt.figure(figsize=[5,5],dpi=150)
    kge_daily_b = 1-metric_baseline_mean.loc['kge_daily','baseline']
    sn.distplot(y_predicted,label='predicted optimized')
    fig = plt.gca()
    ylim = fig.get_ylim()
    plt.plot([kge_daily_b,kge_daily_b],ylim, label='default')
    plt.ylim(ylim)
    plt.xlabel('flow objective (smaller better)')
    plt.legend()
    plt.savefig(figure_dir + 'predicted_pareto_surface.iter_%s.png'%(init_iteration),dpi=200,bbox_inches='tight')

# scaled the x_sample to (0,1)
x_sample_scaled = (x_resample - xlb_single_value_scaled)/(xub_single_value_scaled-xlb_single_value_scaled)
perturbed_param = []
for i in range(N_resample):
    perturbed_param.append(x_sample_scaled[i,:] * (xub - xlb) + xlb)
perturbed_param = np.array(perturbed_param)

# create test id
test_id_list=[]
for id_ in range(N_resample):
    test_id = '%s_%s_%s'%(location, "%i"%(iteration), "%04i"%(id_))
    test_id_list.append(test_id)
psets_df = pd.DataFrame(perturbed_param, columns=pf[var_param].values, index=test_id_list)
psets_N_df = psets_df[N_var_list]
psets_P_df = psets_df[P_var_list]
# output files
psets_df.to_csv('/glade/work/yifanc/NNA/optmz/workflow/%s_%s.param_list.txt'%(location, "%i"%(iteration)))
main_run = '/glade/work/yifanc/NNA/optmz/workflow/%s_%s.main_run.txt'%(location, "%i"%(iteration))
with open(main_run,"w") as file:
    case_list = '\n'.join(psets_df.index.values)
    case_list += '\n'
    file.write(case_list)
file.close()

# assign the basepftfile
param_output_dir = "/glade/work/yifanc/NNA/optmz/paramfile/%s/"%(location)
if not path.isdir(param_output_dir):
    os.mkdir(param_output_dir)
# determine whether we need to generate new parameter files
if len(P_var_list)>0:
    for index, row in psets_P_df.iterrows():
        # open the default file
        tmp = xr.open_dataset(basepftfile)
        pftfile = param_output_dir+index+".nc"
        for var in P_var_list:
            if pf[pf[var_param]==var][var_min].values[0] == 'pft':
                var_value = tmp[var].values
                if len(var_value.shape) == 1:
                    var_value[pft_type_sel] = row[var]
                else:
                    var_value[:,pft_type_sel] = row[var]
                tmp[var] = xr.DataArray(var_value,dims=tmp[var].dims)
            else:
                if type(row[var]) == np.ndarray:
                    dims = tmp[var].dims
                    tmp[var] = xr.DataArray(row[var], dims=dims)
                else:
                    tmp[var] = row[var]
        print('Done working on '+pftfile)
        tmp.to_netcdf(pftfile,'w')

# write namelist files
namelist_dir='/glade/work/yifanc/NNA/optmz/namelist/%s/'%(location)
if not path.isdir(namelist_dir):
    os.mkdir(namelist_dir)
# determine whether we need to generate new namelist files
for index, row in psets_N_df.iterrows():
    nlfile = namelist_dir+index+".txt"
    with open(nlfile,"w") as file:
        output = "! user_nl_clm namelist options written by generate_params:\n"
        file.write(output)
for index, row in psets_N_df.iterrows():
    nlfile = namelist_dir+index+".txt"
    print('working on '+nlfile)
    for var in N_var_list:
        with open(nlfile,"a") as file: # key is using "a" for append option
            print(var+' modified')
            output = "%s=%s\n" % (var, float(row[var])) #round??
            file.write(output) 

