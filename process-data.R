library(Rtsne)
library(data.table)
library(dplyr)
library(tidyr)
library(tidyverse)
library(ggplot2)
library(xgboost)
library(lubridate)
library(compiler)
library(zoo)
library(xts)
library(caret)
library(mlr)
library(gbm)
library(forecast)
library(grid)
library(leaps)
library(glmnet)
library(purrr)
library(purrrlyr)
library(zoo)
library(feather)
library(doParallel)
library(parallel)
library(foreach)
library(moments)
# Calculate the number of cores
no_cores <- max(7, detectCores() - 1)
registerDoParallel(no_cores)
daily.directory <- file.path('.', 'data', 'daily_dataset')
halfhourly.directory <- file.path('.', 'data', 'halfhourly_dataset')
hhblock.directory <- file.path('.', 'data', 'hhblock_dataset')


daily.files <- list.files(path=daily.directory, pattern="*csv", full.names=TRUE, recursive=FALSE)
halfhourly.files <- list.files(path=halfhourly.directory, pattern = '*.csv', full.names = T, recursive = F)
hhblock.files <- list.files(path=hhblock.directory, pattern='*.csv', full.names = T, recursive = F)


daily.data <- foreach(i = daily.files, .combine = rbind) %dopar% 
{
  d <- data.table::fread(i, sep = ',', stringsAsFactors = F)
  d$day = as.POSIXct(d$day, tryFormats = '%Y-%m-%d')
  d
}

daily.data <- setDT(daily.data)
glimpse(daily.data)

halfhourly.data <- foreach(i = halfhourly.files, .combine = rbind) %dopar% 
{
  d <- data.table::fread(i, sep = ',', stringsAsFactors = F)
  d$tstp <- as.POSIXct(d$tstp)
  d$`energy(kWh/hh)` = as.numeric(d$`energy(kWh/hh)`)
  d$Date = as.Date(d$tstp)
  d
}

halfhourly.data <- setDT(halfhourly.data)
glimpse(halfhourly.data)

hhblock.data <- foreach(i = hhblock.files, .combine = rbind) %dopar% 
{
  d <- data.table::fread(i, sep = ',', stringsAsFactors = F)
  d$day <- as.POSIXct(d$day)
  d
}

hhblock.data <- setDT(hhblock.data)
glimpse(hhblock.data)
uk_bank_holidays = read.csv('data/uk_bank_holidays.csv', sep=',', stringsAsFactors = F)
uk_bank_holidays$Bank.holidays = as.POSIXct(us_bank_holidays$Bank.holidays)
glimpse(uk_bank_holidays)

weather_daily_darksky = read.csv('data/weather_daily_darksky.csv', sep=',', stringsAsFactors = F)
date_time_fields = c("temperatureMaxTime", "temperatureMinTime", "apparentTemperatureMinTime", "apparentTemperatureHighTime","time", "sunsetTime",
                    "uvIndexTime"  ,"sunriseTime","temperatureHighTime", "temperatureLowTime",  "apparentTemperatureMaxTime",
                    "apparentTemperatureLowTime" )

weather_daily_darksky$uvIndexTime[weather_daily_darksky$uvIndexTime == ''] = weather_daily_darksky$time[weather_daily_darksky$uvIndexTime == ''] 

t = foreach(d = date_time_fields) %do%
{
  weather_daily_darksky[,d] =  as.POSIXct(weather_daily_darksky[,d])
  NULL
}


glimpse(weather_daily_darksky)


weather_hourly_darksky <- fread('data/weather_hourly_darksky.csv', sep = ',', stringsAsFactors = F)
weather_hourly_darksky$time = as.POSIXct(weather_hourly_darksky$time)
glimpse(weather_hourly_darksky)

write_feather(daily.data, 'data/daily_data.feather')
write_feather(halfhourly.data, 'data/halfhourly_data.feather')
write_feather(hhblock.data, 'data/hhblock_data.feather')
write_feather(us_bank_holidays, 'data/us_bank_holidays.feather')
write_feather(weather_daily_darksky, 'data/weather_daily_darksky.feather')
write_feather(weather_hourly_darksky, 'data/weather_hourly_darksky.feather')


daily.data = setDT( read_feather('data/daily_data.feather'))
halfhourly.data = setDT( read_feather('data/halfhourly_data.feather'))
hhblock.data = setDT( read_feather('data/hhblock_data.feather'))
us_bank_holidays= setDT( read_feather('data/us_bank_holidays.feather'))
weather_daily_darksky = setDT( read_feather('data/weather_daily_darksky.feather'))
weather_hourly_darksky = setDT( read_feather( 'data/weather_hourly_darksky.feather'))

house_hold_information = read.csv('data/informations_households.csv')
acorn = read.csv('data/acorn_details.csv')

get_summary <- function(v, tstp)
{
  l = as.list(numeric(3))
  v = as.numeric(v)
  l[3]= as.integer(24)
  
  if(!purrr::is_empty(v[v>0]))
  {
    max.usage.index = which(v == max(v[v>0]))
    max.usage = as.integer(hour(tstp[max.usage.index]))
  
    v = v[v>0]
    
    l = list(skewness(v, na.rm=T),
             kurtosis(v, na.rm=T),
             max.usage)
    
  }
  
  return (l)
}
get.summary = cmpfun(get_summary)

daily.data.mo = halfhourly.data[,get.summary(`energy(kWh/hh)`, tstp), by=list(LCLid, Date)]
daily.data.mo$V3 = as.factor(daily.data.mo$V3)
names(daily.data.mo) = c('LCLid', 'Date', 'energy_skewness', "energy_kurtosis", 'energey_max_usage_hour')
glimpse(daily.data.mo)

d = daily.data
d$day = as.Date(d$day)
ss = d %>% inner_join(daily.data.mo, by = c('LCLid' = 'LCLid', 'day' = 'Date'))    

glimpse(ss)

glimpse(weather_hourly_darksky)
weather_hourly_darksky$Date = as.Date(weather_hourly_darksky$time)

weather_summary <- function(temperature)
{
  l = list(skewness(temperature), kurtosis(temperature))
  l
}
weather.summary = cmpfun(weather_summary)

weather_sum = weather_hourly_darksky[, weather.summary(temperature), by=Date]
names(weather_sum) = c('Date', 'temperature_skewness', 'temperature_kurtosis')
glimpse(weather_daily_darksky)
weather_daily_darksky$time = as.Date(weather_daily_darksky$time)
weather_daily = weather_daily_darksky %>% inner_join(weather_sum, by=c('time' = 'Date')) %>% 
  mutate(day_length = as.numeric(difftime(sunsetTime, sunriseTime), units = "secs")/(24.0*60*60))

date_time_fields = c("temperatureMaxTime", "temperatureMinTime", "apparentTemperatureMinTime", "apparentTemperatureHighTime","sunsetTime",
                     "uvIndexTime"  ,"sunriseTime","temperatureHighTime", "temperatureLowTime",  "apparentTemperatureMaxTime",
                     "apparentTemperatureLowTime" )

for(i in 1:NROW(date_time_fields))
{
  d = date_time_fields[i]
  weather_daily[,gsub('Time', 'Hour', d)] = hour(weather_daily[,d])
}
weather_daily = weather_daily %>% select(-date_time_fields)
glimpse(weather_daily)

glimpse(house_hold_information)

us_bank_holidays$Bank.holidays = as.Date(us_bank_holidays$Bank.holidays)

total_data = ss %>% inner_join(weather_daily, by = c('day' = 'time' )) %>% inner_join(house_hold_information, by = 'LCLid') %>% select(-file) %>%
  left_join(us_bank_holidays, by=c('day' = 'Bank.holidays')) %>% mutate(day.of.week = weekdays(day))


glimpse(total_data)

factor_files = c("energey_max_usage_hour", "icon", "temperatureMaxHour", "temperatureMinHour", "apparentTemperatureMinHour",
                 "apparentTemperatureHighHour","sunsetHour", "uvIndexHour", "sunriseHour", "temperatureHighHour", "temperatureLowHour", "apparentTemperatureMaxHour",
                 "apparentTemperatureLowHour", "stdorToU", "Acorn", "Acorn_grouped", "Type", "day.of.week")

unique(total_data$Type)


total_data$Type[is.na(total_data$Type) ] = 'Normal'