# CA plot routines
Python routines to plot adcirc results
```
git clone --recursive https://github.com/saeed-moghimi-noaa/ca_adcirc_plot
cd ca_adcirc_plot
```
## Contact
Saeed.Moghimi@noaa.gov

## Create conda env:
conda env create -f environment.yaml

## Link to sample model results:
https://drive.google.com/file/d/1dlzOGnwldQtADj4s37g0yZsOERgP9vBX/view?usp=sharing

## Link to sample observation:
https://drive.google.com/file/d/1iIjEZ5B7i9lzzyX5_s2kefbosZwNN1Q4/view?usp=sharing

## Howto

```
conda activate plot_adc
```

Edit base_info_flo.py to include location of obs and model result (runs) folders.

```
ln -s  base_info_flo.py   base_info.py
ipython --pylab
...
In [1]: run plot_hwm.py  
In [2]: run plot_maps.py 
In [3]: run plot_time_seris_csv.py
```