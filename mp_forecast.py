from bs4 import BeautifulSoup as bs
# from numpy.core.fromnumeric import sort
# from pandas.core.algorithms import isin
# from numpy.lib.arraysetops import isin
import requests
import re
# import numpy as np
from datetime import datetime
import pandas as pd
# from urllib.parse import urljoin
import errno
import time
import pickle
import os
# import glob
import argparse

california_url = 'https://www.mountainproject.com/area/105708959/california'
high_sierra_url = 'https://www.mountainproject.com/area/105791817/high-sierra'
seki_url = 'https://www.mountainproject.com/area/108147148/sequoia-kings-canyon-np'
forecast_url_ = 'https://forecast.weather.gov/'
forecast_url_base = 'https://forecast.weather.gov/MapClick.php?lon={longitude}&lat={latitude}#.YVIu8bhKg-U'
shasta_url = 'https://www.mountainproject.com/area/106464248/mt-shasta'
tahoe_url = 'https://www.mountainproject.com/area/105798291/lake-tahoe'
san_gorgonio_url = 'https://www.mountainproject.com/area/107446455/san-gorgonio-mountain'
castle_peak_url = 'https://www.mountainproject.com/area/121309470/castle-peak'
oregon_volcanoes_url = 'https://www.mountainproject.com/area/109532053/oregon-volcanoes'
wind_river_url = 'https://www.mountainproject.com/area/105823538/wind-river-range'
ritter_minarets_url = 'https://www.mountainproject.com/area/110847705/04-ritter-minarets'
joshua_tree_url = 'https://www.mountainproject.com/area/105720495/joshua-tree-national-park'
yosemite_url = 'https://www.mountainproject.com/area/105833388/yosemite-valley'
indian_creek_url = 'https://www.mountainproject.com/area/105716763/indian-creek'
cochise_url = 'https://www.mountainproject.com/area/105738034/cochise-stronghold'

saved_urls = dict(california_url=california_url, high_sierra_url=high_sierra_url, seki_url=seki_url,
				  shasta_url=shasta_url, tahoe_url=tahoe_url, san_gorgonio_url=san_gorgonio_url,
				  castle_peak_url=castle_peak_url, oregon_volcanoes_url=oregon_volcanoes_url,
				  wind_river_url=wind_river_url,ritter_minarets_url=ritter_minarets_url,
				  joshua_tree_url=joshua_tree_url,yosemite_url=yosemite_url,
				  indian_creek_url=indian_creek_url,cochise_url=cochise_url)
cwd = os.getcwd()
if not 'mp-forecast' in cwd:
	directory = os.path.join(cwd,'mp-forecast')
else:
	directory = cwd
def check_dir(path):
	'''
	Create any missing directories
	'''
	if not os.path.exists(path):
		print('Creating Directory {}'.format(path))
		try:
			os.makedirs(path)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise e
check_dir(os.path.join(directory,'Forecasts'))

class MtProjForecast():
	def __init__(self, base_url,detailed_forecast=True):
		'''
		mp_soup stores the BeautifulSoup object for the mountain project page
		f_soup stores the BeautifulSoup object for the forecast.gov page
		'''
		if base_url[:5] != 'https':
			url = saved_urls.get(base_url, None)
			if url is None:
				raise ValueError('url not found {}'.format(base_url))
		self.base_url = base_url
		self.new_mp_soup(url=base_url)
		self.f_soup = None
		self.detailed_forecast = detailed_forecast

	def get_url(self, url):
		req = requests.get(url)
		req.raise_for_status()
		return req

	def new_mp_soup(self, url=None):
		page = self.get_url(url if url else self.base_url)
		# page = requests.get(url if url else self.base_url)
		soup = bs(page.content, 'html.parser')
		self.mp_soup = soup

	def new_f_soup(self, url):
		page = self.get_url(url)
		# page = requests.get(url)
		soup = bs(page.content, 'html.parser')
		self.f_soup = soup

	def get_area_name(self, soup=None):
		if soup is None:
			soup = self.mp_soup
		an = soup.find('h3').string
		print(an)
		cleaned = clean_text(an)
		return cleaned if cleaned else clean_text(re.search(r'(?<=in )([^\t\n\r\f\v\d]*)', an).group())

	def area_or_route(self, soup=None):
		if soup is None:
			soup = self.mp_soup
		res = soup.find('h3').string
		return 'Area' if 'Area' in res else 'Route'

	def pickled(self, urls_filename, url_dir=directory):
		fp = join_pickle_file(urls_filename=urls_filename,url_dir=url_dir)
		return os.path.exists(fp)

	def pickle_areas(self, areas):
		if len(areas.keys()) == 1:
			fn = list(areas.keys())[0]
			if not self.pickled(fn):
				dump_urls(areas, fn)
				return
		for k, v in areas.items():
			if isinstance(v, dict):
				if not self.pickled(fn):
					dump_urls(areas, k)
				return
			if not self.pickled(fn):
				dump_urls(v, k)

	def unpickle_areas(self, area_name, url_dir=directory):
		return load_urls(urls_filename=area_name, url_dir=url_dir)

	def get_areas(self, url=None, soup=None, area_name=None):
		if not soup:
			page = self.get_url(url if url else self.base_url)
			# page = requests.get(url if url else self.base_url)
			soup = bs(page.content, 'html.parser')
		self.mp_soup = soup
		if self.area_or_route() == 'Route':
			return {}
		if not area_name:
			area_name = self.get_area_name()
		# print(dict(area_name=area_name))
		unpickled_areas = self.unpickle_areas(area_name)
		if unpickled_areas:
			# print(unpickled_areas)
			return unpickled_areas[area_name]
		links = {}
		for link in soup.find_all(class_="lef-nav-row"):
			links[clean_text(link.contents[1].text)
				  ] = link.contents[1].get('href')
		self.pickle_areas({area_name: links})
		return links

	def get_area_stats(self, url=None, soup=None):
		if not soup:
			if url:
				self.new_mp_soup(url=url)
		stats = {}
		stats.update(self.get_elevation())
		gps = self.get_gps()
		stats.update(gps)
		fd = {}
		if self.detailed_forecast:
			fd.update(self.get_weather_forecast_details(gps))
		else:
			fd.update(self.get_seven_day_weather_forecast_dict(gps))
		fd.update(self.get_about_forecast())
		stats.update({'Forecast': fd})
		return stats

	def visit_areas(self, areas=None, visit_sub_areas=False):
		if areas is None:
			areas = self.get_areas()
		if not areas:
			return {self.get_area_name(): self.get_area_stats()}
		area_stats = {}
		for k, v in areas.items():
			self.new_mp_soup(url=v)
			stats = self.get_area_stats()
			# stats = {}
			# stats.update(self.get_elevation())
			# gps = self.get_gps()
			# stats.update(gps)
			# fd = {}
			# fd.update(self.get_weather_forecast_details(gps))
			# fd.update(self.get_about_forecast())
			# stats.update({'Forecast': fd})
			sub_areas = self.get_areas(url=v)
			if sub_areas:
				if not visit_sub_areas:
					stats.update({'Sub Areas': sub_areas})
				else:
					sub_visits = self.visit_areas(
						areas=sub_areas, visit_sub_areas=visit_sub_areas)
					stats.update({'Sub Areas': sub_visits})
			area_stats[k] = stats

		return area_stats

	def get_area_forecasts(self, area_stats):
		self.area_forecasts = [AreaForecast(location=k, gps=v['GPS'], elevation=v['Elevation'],
											forecast=v['Forecast'], sub_areas=v.get('Sub Areas', {}),detailed_forecast=self.detailed_forecast) for k, v in area_stats.items()]

	def get_elevation(self, soup=None):
		if soup is None:
			soup = self.mp_soup
		details = soup.find('table', class_="description-details")
		# elev = details.find('td', class_="text-nowrap")
		# return {re.sub(r'\W', '', elev.next.string): re.sub(r'\W', '', elev.next.next.next.string)}
		elevation = None
		comp = re.compile('(\d*[\,]?\d* ft{1})')
		for t in details.find_all('td'):
			ts = t.string
			if ts:
				# if 'Elevation' in ts:
				# 	ele_key = ts
				matched = comp.search(ts)
				if matched:
					elevation = matched.group()
				# if ' ft' in ts:
				# 	elevation = ts
		# if not elevation:
		# 	backup_method = details.find('td', class_="text-nowrap").next.next.next.string
		# 	if comp.search(backup_method):
		# 		elevation = comp.search(backup_method).group()
		ele = {'Elevation':elevation if elevation else 'Not Available'}
		# print(ele)
		return ele

	def get_gps(self, soup=None):
		if soup is None:
			soup = self.mp_soup
		details = soup.find('table', class_="description-details")
		# elev = details.find('td', class_="text-nowrap")
		for t in details.find_all('td'):
			if 'GPS' in t.string:
				gps = t
				break
		return {re.sub(r'\W', '', gps.string): re.sub('[ \t\n\r\f\v]', '', gps.next_sibling.next_sibling.contents[0])}

	def get_weather_forecast_details(self, gps):
		latitude, longitude = gps['GPS'].split(',')
		f_url = forecast_url_base.format(
			longitude=longitude, latitude=latitude)
		self.new_f_soup(f_url)
		# forecast = requests.get(f_url)
		# forecast.raise_for_status()
		# self.f_soup = bs(forecast.content, 'html.parser')
		# self.f_soup = fsoup
		f_body = self.f_soup.find(id='detailed-forecast-body')
		details = []
		for det in f_body.find_all('div'):
			if det.string:
				details.append(det.string)
		return dict(chunks(details, 2))

	def get_seven_day_weather_forecast_dict(self, gps, get_images=False):
		return dict([f.split(': ') for f in self.get_seven_day_weather_forecast(self, gps=gps, get_images=False)])

	def get_seven_day_weather_forecast(self, gps, get_images=False):
		'''
		EXAMPLE
		for f in get_seven_day_weather_forecast(gps):
										print(f)
		'''
		latitude, longitude = gps['GPS'].split(',')
		f_url = forecast_url_base.format(
			longitude=longitude, latitude=latitude)
		self.new_f_soup(f_url)
		# forecast = requests.get(f_url)
		# forecast.raise_for_status()
		# self.f_soup = bs(forecast.content, 'html.parser')
		# self.f_soup = fsoup
		f_body = self.f_soup.find(id='seven-day-forecast-container')
		details = []
		if not get_images:
			for ft in f_body.find_all('li', class_='forecast-tombstone'):
				yield ft.find_all('p')[1].next_element['alt']
		else:
			for ft in f_body.find_all('li', class_='forecast-tombstone'):
				ne = ft.find_all('p')[1].next_element
				sd = ne.next_element
#                 print(sd.__dict__)
				yield {'description': ne['alt'],
					   'image': forecast_url_+ne['src'],
					   'short desc': ft.find('p').next.string,
					   'Temp': sd.next_element.string, }

	def display_seven_day_images(self):
		'''
		DISPLAY IMAGES IN JUPYTER NOTEBOOK
		'''
		from IPython.display import Image, display
		gps = self.get_gps()
		for with_images in self.get_seven_day_weather_forecast(gps, get_images=True):
			display(Image(data=requests.get(with_images['image'],stream=True).content))

	def save_seven_day_images(self):
		gps = self.get_gps()
		fd = []
		for with_images in self.get_seven_day_weather_forecast(gps, get_images=True):
			#             path = os.path.split(with_images['image'])[-1]
			#             file_ext = os.path.splitext(path)[-1]
			save_path = with_images['Temp'] + '.png'
			print(with_images['image'])
			if not os.path.exists(save_path):
				with open(save_path, 'wb') as handle:
					response = requests.get(with_images['image'], stream=True)
					response.raise_for_status()
					for block in response.iter_content(1024):
						if not block:
							break
						handle.write(block)
			fd.append(with_images)
		return fd

	def get_about_forecast(self, soup=None):
		if soup is None:
			soup = self.f_soup
		f_body = soup.find(id='about_forecast')
		fts = {}
		ft_rows = f_body.find_all('div', class_='fullRow')[:3]
		fts[ft_rows[0].find(class_='left').next.string] = ft_rows[0].find(
			class_='right').next_element
		fts[ft_rows[1].find(class_='left').next.string] = ft_rows[1].find(
			class_='right').next_element
		fts[ft_rows[2].find(class_='left').next.string] = ft_rows[2].find(
			class_='right').next_element
		return fts

	def main(self, **kwargs):
		# print(kwargs)
		self.get_area_forecasts(self.visit_areas(
			areas=kwargs.get('areas', None), visit_sub_areas=kwargs.get('visit_sub_areas', False)))
		if kwargs.get('save', None):
			for af in self.area_forecasts:
				af.main(save=kwargs['save'])


class AreaForecast():
	def __init__(self, location, gps, elevation, forecast, sub_areas={},detailed_forecast=True):
		self.location = location
		self.gps = gps
		self.elevation = elevation
		self.forecast = forecast
		self.detailed_forecast = detailed_forecast
		self.sub_areas = None
		if sub_areas:
			# print(sub_areas)
			
			try:
				self.sub_areas = [AreaForecast(location=k, gps=v['GPS'], elevation=v['Elevation'],
										   forecast=v['Forecast'], sub_areas=v.get('Sub Areas', {})) for k, v in sub_areas.items() if not isinstance(v,str)]
			except (TypeError,ValueError) as e:
				print(sub_areas)
				raise e

	def area_series(self, pop_forecast=True, transpose=True):
		area = self.__dict__.copy()
		area.pop('forecast')
		area.pop('sub_areas')
		df = pd.DataFrame([area], index=['Area'])
		if transpose:
			df = df.T
		if pop_forecast:
			return df
		return pd.concat([df.rename(columns={'Area': 'Area Forecast'}), self.forecast_series().rename(columns={'Forecast': 'Area Forecast'})], sort=False)

	def forecast_series(self, transpose=True):
		df = pd.DataFrame([self.forecast], index=['Forecast'])
		if transpose:
			return df.T
		return df

	def get_sub_areas(self):
		for sa in self.sub_areas:
			yield sa

	def forecast_dataframe(self, save=False):
		fs = self.forecast_series(transpose=False)
		as_ = self.area_series()
		df = pd.concat([as_, fs], sort=False)
		if save:
			self.save_csv(df)
		return df

	def all_areas_dataframe(self, save=False):
		dfs = []
		df = self.forecast_dataframe()
		if not self.sub_areas:
			if save:
				self.save_csv(df)
			return df
		dfs.append(df)
		for area in self.get_sub_areas():
			df = area.forecast_dataframe()
			dfs.append(df)
		df = pd.concat(dfs, sort=False)
		if save:
			self.save_csv(df)
		return df

	def save_csv(self, srdf):
		dst = os.path.join(directory,'Forecasts','{} {} Forecast {}.csv'.format(self.location,'Simplified' if self.detailed_forecast else 'Detailed',datetime.today().date()))
		if isinstance(srdf, pd.Series):
			srdf = srdf.to_frame().to_csv(dst)
		elif isinstance(srdf, (pd.DataFrame, pd.MultiIndex)):
			srdf.to_csv(dst)
		else:
			raise TypeError('srdf is not of the correct type')

	def main(self, **kwargs):
		# print(kwargs)
		self.all_areas_dataframe(save=kwargs.get('save', False))

	def pickled(self, urls_filename, url_dir=directory):
		fp = join_pickle_file(url_dir, urls_filename)
		return os.path.exists(fp)

	def pickle_areas(self, areas):
		if len(areas.keys()) == 1:
			fn = list(areas.keys())[0]
			if not self.pickled(fn):
				dump_urls(areas, fn)
				return
		for k, v in areas.items():
			if isinstance(v, dict):
				if not self.pickled(fn):
					dump_urls(areas, k)
				return
			if not self.pickled(fn):
				dump_urls(v, k)

	def unpickle_areas(self, area_name, url_dir=directory):
		return load_urls(urls_filename=area_name, url_dir=url_dir)


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


def load_urls(urls_filename, url_dir=directory):
	""" Returns dictionary of mountain urls saved a pickle file """

	full_path = join_pickle_file(urls_filename=urls_filename,url_dir=url_dir)
	if not os.path.exists(full_path):
		return None
	with open(full_path, 'rb') as file:
		urls = pickle.load(file)
	return urls


def join_pickle_file(urls_filename, url_dir=directory):
	print(urls_filename)
	full_path = os.path.join(os.path.normpath(url_dir), urls_filename) + '.pickle'
	print(full_path)
	return full_path


def dump_urls(mountain_urls, urls_filename, url_dir=directory):
	""" Saves dictionary of mountain urls as a pickle file """

	full_path = join_pickle_file(urls_filename=urls_filename,url_dir=url_dir)

	with open(full_path, 'wb') as file:
		pickle.dump(mountain_urls, file)


def clean_text(text):
	# print(text)
	clean_text  =re.sub('Areas in|Routes in','',text).strip()
	if clean_text:
		# print(clean_text)
		text = re.sub('[\\\/]',' ',clean_text)
	matchs = re.findall(r'([^\t\n\r\f\v\d\-\,]*)', text)
	if not matchs:
		return text
	matched = [m for m in matchs if m and not 'Area' in m and not 'Route' in m]
	# print(matched)
	if not matched:
		return text
	mt = max(matched, key=len).strip()
	# print(mt)
	return mt



'''
EXAMPLES
mpf = MtProjForecast(high_sierra_url)
areas = mpf.get_areas()
area_data = mpf.visit_areas(areas={'01 - Hoover Wilderness':areas['01 - Hoover Wilderness']},visit_sub_areas=True)

mpf = MtProjForecast(high_sierra_url)
gps = mpf.get_gps()
mpf.save_seven_day_images()

IF IN JUPYTER NOTEBOOK
from IPython.display import Image,display
mpf = MtProjForecast(high_sierra_url)
gps = mpf.get_gps()
for with_images in mpf.get_seven_day_weather_forecast(gps,get_images=True):
	display(Image(data=requests.get(forecast_url_+with_images['image']).content))

'''


def args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-u', '--url', action='store', default=high_sierra_url,
						type=str, help='URL for the area to get information from. Will additionally check saved_urls for URL variable names eg: oregon_volcanoes_url')
	parser.add_argument('-v', '--visit_sub_areas',
						action='store_true', help='get information for sub areas')
	parser.add_argument('-s', '--save',
						action='store_true', help='Save forecast information to CSV')
	parser.add_argument('-df', '--detailed_forecast',
						action='store_false', help='Save the simplified forecast information')
						
	args = parser.parse_args()
	return vars(args)


def main(**kwargs):
	
	url = kwargs['url']
	if url[:5] != 'https':
		url = saved_urls.get(url, None)
		if url is None:
			raise ValueError('url not found {}'.format(kwargs['url']))
	mpf = MtProjForecast(url)
	mpf.main(**kwargs)
	
	return mpf


if __name__ == '__main__':

	start = time.time()
	main(**args())
	end = time.time()
	print('Process took {} minutes'.format((end - start)/60))
