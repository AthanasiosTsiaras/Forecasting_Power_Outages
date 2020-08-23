setwd("~/Desktop/PPL Electric")
rm(list = ls())

library(tidyverse) 
library(geosphere)
library(caret)
set.seed(69)

## Relevant columns from outage.csv
predict_columns <- c("TR_TIME", "TR_TIME_CLOCK_FK", "FAIL_FAC_ID", "FAC_TYPE_DE", "FACILE_DESC", 
                     "FEED", "LONG", "LAT","LINE", "UPSTR_MUL_DEV", "UPSTRDEV_TYPE", "MWF_REGION", "REASON_DE")

outage <- read.csv("data/outage.csv", header = TRUE)[, predict_columns]
outage <- filter(outage, REASON_DE == "Weather") # only need the weather related outages

# Setting a datetime column named Date and dropping the rest
outage$TR_TIME <- as.Date(outage$TR_TIME, format="%d-%b-%y")
outage$TR_TIME_CLOCK_FK <- format(as.POSIXct((outage$TR_TIME_CLOCK_FK), origin = "1970-01-01", tz = "UTC"), "%H")
outage$Date <- as.POSIXct(paste(as.character(outage$TR_TIME), outage$TR_TIME_CLOCK_FK), format=c('%Y-%m-%d %H'))
drops <- c("TR_TIME", "TR_TIME_CLOCK_FK", "REASON_DE")
outage <- outage[ , !(names(outage) %in% drops)]

# Removing NAs
outage <- outage[complete.cases(outage), ]

# We need to merge weather and outage by Date and METAR (weather station), so:

# 1. Function to find nearest METAR for every outage

Regions <- read.csv("data/regions.csv", header = TRUE)
region_assign <- function(Regions, Outage_row){
        dist <- c()
        for (metar_rows in 1:length(Regions$Metar)){
                dist <-c(dist, distm(c(Outage_row$LONG, Outage_row$LAT), c(Regions[metar_rows, 2], Regions[metar_rows, 3]), fun = distHaversine))
        }
        
        return(as.character(Regions[which.min(dist), 1]) )
}

outage_reg <- c()
for (row in 1:length(outage[,1])){ # Applying the function to every row of the data frame
        
        outage_reg <- c(outage_reg, region_assign(Regions, outage[row, ]))
        
}


outage$METAR_ID <-outage_reg
# mapply(distm, x = c(outage$LAT, Regions$Lat), y = c(outage$LAT, Regions$Lat))

# 2. Reading the weather dataset

weather <- read.csv("data/weather.csv", header = TRUE)
weather$TIME_GMT <- as.Date(weather$TIME_GMT, format="%d-%b-%y")
weather <- filter(weather, WEATHER_RECORD_TYPE == "METAR") # Only need METAR weather records

# Creating a Date column identical to that of outage (merging later)
weather$Date <- as.POSIXct(paste(as.character(weather$TIME_GMT), weather$HOUR_OF_DAY_ID), format=c('%Y-%m-%d %H'))
weather_col <- c("Date" ,"METAR_ID", "AIR_TEMPERATURE_F", "AIR_TEMPERATURE_FEELS_LIKE_F", 
                 "DEWPOINT_TEMPERATURE_F", "RELATIVE_HUMIDITY", "WIND_SPEED_MPH", "WIND_GUST_MPH", "WIND_DIRECTION_DEG",
                 "CLOUD_AMOUNT", "PRECIPITATION_PAST_1HOUR_IN", "SNOWFALL_PAST_1HOUR_IN", "ATMOSPHERIC_PRESSURE_MB", 
                 "DI_NORL_RADIATION", "DIFF_HOZ_RADIATION", "DOWN_SLR_RAD", "LEAF_COVERAGE")

weather <- weather[, weather_col] # Keeping relevant columns

historic <- outage %>% complete(Date = seq(from = min(Date), to = max(Date), by="hour"), FAIL_FAC_ID)
historic$Y <- ifelse(is.na(historic$FAC_TYPE_DESC), 0, 1)
historic$Y <- as.factor(historic$Y)
historic <- historic %>% arrange(FAIL_FAC_ID, desc(Y)) %>% fill(names(outage[, c(2:10, 12)]), .direction = "down")


historic <- historic[complete.cases(historic),]

final <- inner_join(weather, historic, by = c("Date", "METAR_ID"))

final <- final %>% select(-c( DI_NORL_RADIATION, DIFF_HOZ_RADIATION,
                              DOWN_SLR_RAD, UPSTR_MUL_DEV)) # With Region


final$METAR_ID <- as.factor(final$METAR_ID)
final$MWF_REGION <- as.factor(final$MWF_REGION)
final$FAIL_FAC_ID <- as.factor(final$FAIL_FAC_ID)
final$FEEDER <- as.factor(final$FEED)
final$LINE <- as.factor(final$LINE)

final$Date <- substr(final$Date, 1, 10)
final <- final %>% group_by(Date, METAR_ID, MWF_REGION, LEAF_COVERAGE, FAIL_FAC_ID, FAC_TYPE_DESC,FAC_STYPE_DES,
                            FEEDER, LONGITUDE, LATITUDE, LINE, UPSTREAM_DEV_TYPE, Y) %>% 
        summarise(
        
        air_temp_min = min(AIR_TEMPERATURE_F),
        air_temp_ave = mean(AIR_TEMPERATURE_F),
        air_temp_max = max(AIR_TEMPERATURE_F),
        
        air_temp_feel_min = min(AIR_TEMPERATURE_FEELS_LIKE_F),
        air_temp_feel_ave = mean(AIR_TEMPERATURE_FEELS_LIKE_F),
        air_temp_feel_max = max(AIR_TEMPERATURE_FEELS_LIKE_F),
        
        dew_temp_min = min(DEWPOINT_TEMPERATURE_F),
        dew_temp_ave = mean(DEWPOINT_TEMPERATURE_F),
        dew_temp_max = max(DEWPOINT_TEMPERATURE_F),
        
        av_humid = mean(RELATIVE_HUMIDITY),
        
        wind_speed_min = min(WIND_SPEED_MPH),
        wind_speed_ave = mean(WIND_SPEED_MPH),
        wind_speed_max = max(WIND_SPEED_MPH),
        
        wind_gust_min = min(WIND_GUST_MPH),
        wind_gust_ave = mean(WIND_GUST_MPH),
        wind_gust_max = max(WIND_GUST_MPH),
        
        wind_direction_av = mean(WIND_DIRECTION_DEG),
        
        cloud_amnt =  mean(CLOUD_AMOUNT),
        
        precipitation = sum(PRECIPITATION_PAST_1HOUR_IN),
        snowfall = sum(SNOWFALL_PAST_1HOUR_IN),
        
        atm_press_min = min(ATMOSPHERIC_PRESSURE_MB),
        atm_press_ave = mean(ATMOSPHERIC_PRESSURE_MB),
        atm_press_max = max(ATMOSPHERIC_PRESSURE_MB)
)

final2 <- downSample(x = final[, !names(final) %in% ("Y")], y = final$Y, list = FALSE, yname = "Y")
final2 <- final2[, !names(final2) %in% c("Date")]

write.csv(final, "data/final_unbalanced.csv", row.names=FALSE)
write.csv(final2, "data/final_extra.csv", row.names=FALSE)

