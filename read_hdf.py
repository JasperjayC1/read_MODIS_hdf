# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:51:02 2022

@author: DELL
"""
#%% 代码解读
# =============================================================================
# 1.本代码为解决读取MODIS hdf文件而服务
# 2.读取fPAR数据和PAR数据
# 3.hdf文件未存储经纬度信息，需要从xml文件中判断，并且尝试绘制全球地图进行验证
# 参考文献:https://blog.csdn.net/dym755833564/article/details/78143038
# 
# =============================================================================
#%% demo

## the unit of fPAR is mw-2 after multi scale_factor(only a try, not sure)
from pyhdf.SD import SD, SDC
import pprint
import pandas as pd
HDF_FILR_URL = r"E:\Research_life\DATA\MODIS FPAR\2001\GLASS09G02.V50.A2001057.2021020.hdf"
file = SD(HDF_FILR_URL)
 
print(file.info())

datasets_dic = file.datasets()
 
for idx,sds in enumerate(datasets_dic.keys()):
    print(idx,sds)

sds_obj = file.select('FAPAR') # select sds
 
fPAR = sds_obj.get() # get sds data

print(fPAR)


pprint.pprint(sds_obj.attributes())  # 重要 读取信息

fPAR = pd.DataFrame(fPAR)
## Filtering data
fPAR[fPAR>250] = np.nan

## multi scale_factor
fPAR = fPAR * 0.004

#%% !!case one(deal and compose fPAR during 2001 to 2016 in China) 8-days

## 解决思路：

import xarray as xr
import numpy as np
import scipy.io as scio


fPAR_data = np.zeros((736,160,126))

u = 2001001

k = 0

for i in range(2001,2017): #年份循环
    
    for j in range(46):
        
        if str(u)[4:7]=='369':
        
            u = u + 1000 - 368
    
        HDF_FILR_URL = r'E:\Research_life\DATA\MODIS FPAR\\'+ str(i) +'\GLASS09G02.V50.A'+str(u) +'.2021020'+ '.hdf'   
         
        file = SD(HDF_FILR_URL)
        
        sds_obj = file.select('FAPAR') # select sds
     
        fPAR = sds_obj.get() # get sds data
        
        fPAR = pd.DataFrame(fPAR)
        
        fPAR = fPAR.iloc[0:160,506:632]  #90N-10.5N    #73E-135.5E 
        
        fPAR_data[k:k+1,::] = fPAR  
        
        k += 1
        
        u += 8
    

## Filtering data
fPAR_data[fPAR_data>250] = np.nan

## multi scale_factor
fPAR_data = fPAR_data * 0.004  # fPAR_data即为所需



# =============================================================================
# 存为nc数据
# =============================================================================
# 新建字典
nc_dict = {
    # nc文件的维度信息
    "dims": {"time": 736,"lat": 160, "lon": 126,},
    # nc文件的维度信息的坐标信息（lat,lon,time等）
    "coords": {
        "lat": {
            "dims": ("lat",),
            "attrs": {
                "standard_name": "latitude",
                "long_name": "Latitude",
                "units": "degrees_north",
                "axis": "Y",
            },
            "data":np.arange(90,-90,-0.5)[0:160]
        },
        "lon": {
            "dims": ("lon",),
            "attrs": {
                "standard_name": "longitude",
                "long_name": "Longitude",
                "units": "degrees_east",
                "axis": "X",
            },
            "data":np.arange(-180,180,0.5)[506:632]
        },
        # "time": {
        #     "dims": ("time",),
        #     "attrs": {"standard_name": "time", "long_name": "Time"},
        #     "data":np.arange(np.datetime64('2013-01-01T00:00:00','s'),np.datetime64('2014-12-31T19:00:00','s'),np.timedelta64(6,'h'))
        # },
    # },
    
        "time": {
            "dims": ("time",),
            "attrs": {"standard_name": "time", "long_name": "Time"},
            "data":np.arange(736)
            # np.arange('2001', '2016', dtype='datetime64[Y]')
        },
    },
    
    # nc文件中的变量
    "data_vars": {
        "fPAR_data": {
            "dims": ("time","lat", "lon"),
            "attrs": {
                "long_name": "fPAR_data",
                "units": "mw-2",
                "precision": 2,
                "GRIB_id": 11,
                "GRIB_name": "ADSIF",
                "var_desc": "fPAR_data",
                "dataset": "fPAR_data",
                "level_desc": "Surface",
                "statistic": "Satellite-Based",
                "parent_stat": "Other",
                "actual_range": [0, 100000],
            },
            "data":fPAR_data,

        }
    },
    # nc文件的全局属性信息
    "attrs": {
        "Conventions": "COARDS",
        "title": "fPAR_data",
        "description": "Data is from OCO-2 Continuous.",
        "platform": "Model",
        "references": "http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.html",
    },
}

# 使用`from_dict`，将字典转化为xr.Dataset对象
dataset = xr.Dataset.from_dict(nc_dict)

# 将xr.Dataset对象保存为nc文件
dataset.to_netcdf(r"E:\Research_life\DATA\MODIS FPAR\fPAR_2001_2016.nc")

fPAR_data = xr.open_dataset(r"E:\Research_life\DATA\MODIS FPAR\fPAR_2001_2016.nc")

fPAR_data = fPAR_data.fPAR_data.interp(
    lat = np.arange(4,54,0.5),
    lon=np.arange(73.5,135.5,0.5),
    kwargs={
        "fill_value":"extrapolate"
        }
    )
#%% case two (transfer daily PAR data to 8-days data) 
import xarray as xr
import numpy as np
import scipy.io as scio
from pyhdf.SD import SD, SDC
import pprint
import pandas as pd

PAR_data = np.zeros((736,1190,1250))

u = 2001001

k = 0

for i in range(2001,2017): #年份循环
    
    for j in range(46): ## 每次读取八个文件,取最大值存为一个数组,每年的数组为[46,::]
                         
        PAR_doy_data = np.zeros((8,1190,1250))

        for m in range(8):
    
            try: # 2001的PAR数据不连续
    
                HDF_FILR_URL = r'E:\Research_life\DATA\PAR\\'+ str(i) + '\\' + str(i) + '\\' + str(i) +'\GLASS04B01.V42.A'+str(u) +'.2020313'+ '.hdf'   
                 
                file = SD(HDF_FILR_URL)
                
                sds_obj = file.select('PAR') # select sds
             
                PAR = sds_obj.get() # get sds data
                
                PAR = pd.DataFrame(PAR)
                
                PAR = PAR.iloc[400:1590,5060:6310]  #70N-10.5N    #73E-135.5E
            
                PAR_doy_data[m,::] = PAR
                
                u += 1
            
            except Exception:
                
                u += 1
                
                continue
        
        PAR_data[k:k+1,::] = np.nanmax(PAR_doy_data,axis=0)
        
        k += 1
        
    u = int(str(int(str(u)[0:4]) + 1) + '001')
                
    
    
## Filtering data
PAR_data[PAR_data<0] = np.nan

## multi scale_factor
PAR_data = PAR_data * 0.01  # PAR_data即为所需



# =============================================================================
# 存为nc数据
# =============================================================================
# 新建字典
nc_dict = {
    # nc文件的维度信息
    "dims": {"time": 736,"lat": 1190, "lon": 1250,},
    # nc文件的维度信息的坐标信息（lat,lon,time等）
    "coords": {
        "lat": {
            "dims": ("lat",),
            "attrs": {
                "standard_name": "latitude",
                "long_name": "Latitude",
                "units": "degrees_north",
                "axis": "Y",
            },
            "data":np.arange(90,-90,-0.05)[400:1590]
        },
        "lon": {
            "dims": ("lon",),
            "attrs": {
                "standard_name": "longitude",
                "long_name": "Longitude",
                "units": "degrees_east",
                "axis": "X",
            },
            "data":np.arange(-180,180,0.05)[5060:6310]
        },
        # "time": {
        #     "dims": ("time",),
        #     "attrs": {"standard_name": "time", "long_name": "Time"},
        #     "data":np.arange(np.datetime64('2013-01-01T00:00:00','s'),np.datetime64('2014-12-31T19:00:00','s'),np.timedelta64(6,'h'))
        # },
    # },
    
        "time": {
            "dims": ("time",),
            "attrs": {"standard_name": "time", "long_name": "Time"},
            "data":np.arange(736)
            # np.arange('2001', '2016', dtype='datetime64[Y]')
        },
    },
    
    # nc文件中的变量
    "data_vars": {
        "PAR_data": {
            "dims": ("time","lat", "lon"),
            "attrs": {
                "long_name": "PAR_data",
                "units": "wm-2",
                "precision": 2,
                "GRIB_id": 11,
                "GRIB_name": "ADSIF",
                "var_desc": "PAR_data",
                "dataset": "PAR_data",
                "level_desc": "Surface",
                "statistic": "Satellite-Based",
                "parent_stat": "Other",
                "actual_range": [0, 100000],
            },
            "data":PAR_data,

        }
    },
    # nc文件的全局属性信息
    "attrs": {
        "Conventions": "COARDS",
        "title": "PAR_data",
        "description": "Data is from GLASS.",
        "platform": "Model",
        "references": "http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.html",
    },
}

# 使用`from_dict`，将字典转化为xr.Dataset对象
dataset = xr.Dataset.from_dict(nc_dict)

# 将xr.Dataset对象保存为nc文件
dataset.to_netcdf(r'E:\Research_life\DATA\PAR\PAR_2001_2016_after_Filtering.nc')


PAR_data = xr.open_dataset(r'E:\Research_life\DATA\PAR\PAR_2001_2016_after_Filtering.nc')

PAR_data = PAR_data.PAR_data.interp(
    lat = np.arange(4,54,0.5),
    lon=np.arange(73.5,135.5,0.5),
    kwargs={
        "fill_value":"extrapolate"
        }
    )

#%% 绘图

# =============================================================================
# # 经绘图得到了经纬度范围（试出）
# lon = np.arange(-180,180,0.05)
# lat = np.arange(90,-90,-0.05)
# lon = lon[5060:6310]
# lat = lat[400:1590]
# =============================================================================

lon = np.arange(-180,180,0.05)
lat = np.arange(90,-90,-0.05)
lon = lon[5060:6310]
lat = lat[400:1590]

data = PAR_data[21,::]
#%%

proj = ccrs.PlateCarree()
fig = plt.figure(figsize = (16,9))
ax = fig.subplots(1,1,subplot_kw = {'projection':proj})
# ax.add_feature(cfeat.BORDERS.with_scale('50m'), linewidth=0.8, zorder=1)
ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
# ax.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=1)
# ax.add_feature(cfeat.LAKES.with_scale('50m'), zorder=1)
# ax.set_extent([70, 135, 14, 55],crs=proj)# 设置范围 

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
linewidth=1.2, color='k', alpha=0.5, linestyle='--')
gl.xlabels_top = False #关闭顶端标签
gl.ylabels_right = False #关闭右侧标签
gl.xformatter = LONGITUDE_FORMATTER #x轴设为经度格式
gl.yformatter = LATITUDE_FORMATTER #y轴设为纬度格式
gl.xlines = None  # 关闭x网格
gl.ylines = None  # 关闭y网格

# gl.xlabel_style = {'size': 14, 'color': 'gray','weight':'normal'}   # 修改主图x坐标字体特征
# gl.ylabel_style = {'size': 14, 'color': 'red', 'weight': 'bold'}   # 修改主图y坐标字体特征
gl.xlabel_style = {'size': 14.5, 'color': 'gray', 'weight': 'normal'}   # 修改主图x坐标字体特征
gl.ylabel_style = {'size': 14.5, 'color': 'darkred', 'weight': 'normal'}   # 修改主图y坐标字体特征

# 设置colorbar
# cbar_kwargs = {
# 'orientation': 'horizontal', # cb放在图下面
# 'label': 'Percentage',
# 'shrink': 0.8,
# }


# levels = np.arange(0,1.05,0.15)
# 画图
china = shpreader.Reader(r'E:\Research_life\毕设分区\矢量图层-20210720T085947Z-001\矢量图层\国界与省界\bou2_4l.dbf').geometries()
ax.add_geometries(china, ccrs.PlateCarree(),facecolor='none', edgecolor='black',zorder = 1)
# 颜色映射，以0为分界线
a = ax.contourf(lon,lat,data,cmap=cmaps.BlueYellowRed, transform=ccrs.PlateCarree())
cb = fig.colorbar(
    
        a, ax=ax, shrink=0.67, pad=0.08, orientation='horizontal'
        
    )

cb.ax.tick_params(labelsize=14)  #设置色标刻度字体大小。

font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
        }
cb.set_label('colorbar',fontdict=font) #设置colorbar的标签字体及其大小
# plt.title('CHINA is the best country in the world',fontsize=22)
