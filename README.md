# OptFed


To recreate the results and plots from the article run the files in the numbered order:

01_fit.py

Create the model fit from the training data. If you run the script on a PC it will take several hours. 

02_fit_models.ipynb

It describes the model fit and compares it with RSM

03_optimize.ipynb

Calculate the optimal process based on the training data.

04_validation.ipynb

Results of the validation experiments and comparison to model predictions



The last 2 files are for the simulated validation. Note that this will not run on a PC in a reasonable time. It took about 1 day on 100 cores in a server environment.

05_sim_val.py

creates random data and fits the models for it.

06_sim_val_results.ipynb

calculates and compares optima for the random processes and the models fitted to them.


The product and biomass yields are calculated with fba_model/calc_yields.ipynb

The used software versions are found in OptFed.yaml.


