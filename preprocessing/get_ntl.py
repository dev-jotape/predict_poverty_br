import rioxarray
rds = rioxarray.open_rasterio(
	"../input/VNL_v21_npp_2021_global_vcmslcfg_c202205302300.average.dat.tif",
	cache=False
)
rds.name = "data"
print("finished import")
xmin, ymin, xmax, ymax = [-74.00, -34.00, -33.00, 6.00]
df = rds.squeeze().sel(x=slice(xmin, xmax)).to_dataframe().reset_index()
print("finished to df")
df = df[((df.y < 6.00) & (df.y > -34.00)) & ((df.x < -33.00) & (df.x > -74.00))]
print("finished filter")
nighttime_lights = df[['y','x','data']].rename({'y':'lat', 'x':'long', 'data': 'radiance'}, axis=1)
print("finished rename")
nighttime_lights.to_csv('../model/nightlights_br.csv', index=False)