# ctsm_optz
This repo contains the workflow used for CTSM optimization using ASMO and PPE.

Details for this workflow can be referred to Cheng et al., 2023.
Cheng, Y., K. Musselman, S. Swenson, D. Lawrence, J. Hamman, K. Dagon, D. Kennedy, and A. Newman, 2023: Moving land models towards more actionable science: A novel application of the Community Terrestrial Systems Model across Alaska and the Yukon River Basin. Water Resources Research, doi:10.1029/2022WR032204.

The workflow for optimizing CTSM includes 4 main steps.
1) Select sensitive parameters using the Perturbed Parameter Ensemble (PPE) experiment
2) Parameter filtering - Simulating the response surface of objective functions to parameters
3) Optimizing parameters
4) Parameters regionalization

Please note that Step 4) is highly dependent on the targeted application so we will not include parameter regionalization in this repo. 

## Step 1 - selecting params using PPE

The script used in this study is modified from a template developed by Daniel Kennedy, Katie Dagon at CGD, NCAR. Here is an example of the template
https://github.com/djk2120/CLM5PPE/blob/master/pyth/oaat.ipynb

We also included the ipython notebook used in this study (PPE.select_sensitive_params.ipynb) but the PPE dataset might not be available for public use. 
Please reach out to Dr. Yifan Cheng (yifanc@ucar.edu) or Daniel Kennedy (djk2120@ucar.edu) for updated information


## Step 2 - simulating the response surface of objective functions to parameters

We first use script Initial_Sampling.ipynb to generate the perturbed parameter files and namelists for CLM.

Then we run CTSM using the perturbed parameters, run mizuRoute to generate flow data and calculated the corresponding objective functions. The script used in this step is: "workflow/run_optmz.CTSM.iter_0.sh" The corresponding configurations is "workflow/env/pe_basin.OM.temp.env".

Later, we further selected the parameters based on the response surface of objective functions to parameters. An example ipython notebook is: additional_parameter_filter.ipynb. Please remember to unzip "workflow_Step2.tar.gz" and "sm_metric_Step2.tar.gz" in "./data" folder for practice purposes.

## Step 3 - optimization

We still use script Initial_Sampling.ipynb to generate the initial perturbed parameter files and namelists for CLM (hereafter referred to as Iteration 0). Please remember to change the parameter file to "data/final_list_params_asmo.csv" (Variable: pf).

Then we run CTSM using using the perturbed parameters, run mizuRoute to generate flow data and calculate objective functions for Iteration 0. After Iteration 0 is finished, we use script "workflow/run_optmz.CTSM.sh" to optimize CTSM for iterations. Please note that we need to specify the starting and ending iteration IDs based on the limitations on the systems. For example, it takes roughly two hours to finish run CTSM for each iteration and the run time limit for PBS jobs at NCAR's Cheyenne system is 12 hours. So I choose to run 5 iterations at a time.

Finally, we use this script "examine_optimization_results.ipynb" to examine the optimization results. Please remember to unzip "workflow.tar.gz" and "sm_metric.tar.gz" in "./data" folder for practice purposes.


# Contact Information
Please reach out to Yifan Cheng @yifanc@ucar.edu if you have any questions concerning the workflow.
