# mp_forecast
NOAA weather forecast scraper from Mountain Project area pages.

```
# EXAMPLES
mpf = MtProjForecast(high_sierra_url)
areas = mpf.get_areas()
area_data = mpf.visit_areas(areas={'01 - Hoover Wilderness':areas['01 - Hoover Wilderness']},visit_sub_areas=True)

mpf = MtProjForecast(high_sierra_url)
gps = mpf.get_gps()
mpf.save_seven_day_images()

# JUPYTER NOTEBOOK
from IPython.display import Image,display
mpf = MtProjForecast(high_sierra_url)
gps = mpf.get_gps()
for with_images in mpf.get_seven_day_weather_forecast(gps,get_images=True):
	display(Image(data=requests.get(forecast_url_+with_images['image']).content))

# COMMAND LINE
python -m mp_forecast -u seki -s
```

