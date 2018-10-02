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
us_bank_holidays = read.csv('data/uk_bank_holidays.csv', sep=',', stringsAsFactors = F)
us_bank_holidays$Bank.holidays = as.POSIXct(us_bank_holidays$Bank.holidays)
glimpse(us_bank_holidays)

weather_daily_darksky = read.csv('data/weather_daily_darksky.csv', sep=',', stringsAsFactors = F)
date_time_fields = c("temperatureMaxTime", "temperatureMinTime", "apparentTemperatureMinTime", "apparentTemperatureHighTime","time", "sunsetTime",
                    "uvIndexTime"  ,"sunriseTime","temperatureHighTime", "temperatureLowTime",  "apparentTemperatureMaxTime",
                    "apparentTemperatureLowTime" )

weather_daily_darksky$uvIndexTime[weather_daily_darksky$uvIndexTime == ''] = weather_daily_darksky$time[weather_daily_darksky$uvIndexTime == ''] 

t = foreach(d = date_time_fields) %do%
{
  weather_daily_darksky[,d] = as.POSIXct(weather_daily_darksky[,d])
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
  v = as.numeric(v)
  v = v[v>0]
  
  l = list(mean(v,na.rm=T), median(v,na.rm=T), sd(v,na.rm=T), max(v,na.rm=T), min(v,na.rm=T), NROW(v), sum(v, na.rm=T))
  return (l)
}

daily.data.mo = halfhourly.data[,get_summary(`energy(kWh/hh)`, tstp), by=list(LCLid, Date)]

d = daily.data
d$day = as.Date(d$day) + days(1)
ss = d %>% inner_join(daily.data.mo, by = c('LCLid' = 'LCLid', 'day' = 'Date'))    

glimpse(ss)
