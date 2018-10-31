library(data.table)
library(dplyr)
library(xgboost)
library(ggplot2)

train <- fread("/Users/katie/Desktop/R extra project/adtracking_dataset.csv", 
               select =c("ip", "app", "device", "os", "channel", "click_time", "is_attributed"),
               showProgress=F,
               colClasses=c("ip"="numeric","app"="numeric","device"="numeric","os"="numeric","channel"="numeric","click_time"="character","is_attributed"="numeric"))

#test <- fread("/Users/katie/Desktop/R extra project/test.csv", 
              #select =c("ip", "app", "device", "os", "channel", "click_time"),
              #showProgress=F,
              #colClasses=c("ip"="numeric","app"="numeric","device"="numeric","os"="numeric","channel"="numeric","click_time"="character"))

set.seed(1234)

train <- train[c(which(train$is_attributed == 1),sample(which(train$is_attributed == 0),9773,replace = F)), ]
str(train)

y <- train$is_attributed

#write.csv(y,'/Users/katie/Desktop/train_y.csv')

n_train = nrow(train)
#dat_combined <- rbind(train,test[,-1],fill = T)
#rm(train,test)
#invisible(gc())

train[, ':='(hour = hour(click_time))
             ][, ip_count := .N, by = "ip"
               ][, app_count := .N, by = "app"
                 ][, channel_count := .N, by = "channel"
                   ][, device_count := .N, by = "device"
                     ][, os_count := .N, by = "os"
                       ][, app_count := .N, by = "app"
                         ][, ip_app := .N, by = "ip,app"
                           ][, ip_dev := .N, by = "ip,device"
                             ][, ip_os := .N, by = "ip,os"
                               ][, ip_channel := .N, by = "ip,channel"
                                 ][,ip_hour := .N, by = "ip,hour"
                                   ][,app_device := .N, by = "app,device"
                                     ][,app_channel := .N, by = "app,channel"
                                       ][,channel_hour := .N, by = "channel,hour"
                                         ][,ip_app_channel := .N, by = "ip,app,channel"
                                           ][,app_channel_hour := .N, by = "app,channel,hour"
                                             ][,ip_app_hour := .N, by = "ip,app,hour"
                                               ][, c("ip","click_time", "is_attributed") := NULL]

#write.csv(train, '/Users/katie/Desktop/train_interaction.csv')

invisible(gc())

train[, lapply(.SD, uniqueN), .SDcols = colnames(train)] %>%
  melt(variable.name = "features", value.name = "unique_values") %>%
  ggplot(aes(reorder(features, -unique_values), unique_values)) +
  geom_bar(stat = "identity", fill = "lightblue") + 
  scale_y_log10(breaks = c(50,100,250, 500, 10000, 50000)) +
  geom_text(aes(label = unique_values), vjust = 1.6, color = "black", size=2) +
  theme_minimal() +
  labs(x = "features", y = "Number of unique values")

within_train_index <- sample(c(1:n_train),0.7*n_train,replace = F) ## split the training dataset into train & validation
processed_train_train = train[1:n_train,][within_train_index]
y1 = y[1:n_train][within_train_index]
processed_train_val = train[1:n_train,][-within_train_index]
y2 = y[1:n_train][-within_train_index]
processed_test = train[-c(1:n_train),]
rm(train)
rm(y)

invisible(gc())

model_train <- xgb.DMatrix(data = data.matrix(processed_train_train), label = y1)
rm(processed_train_train)
invisible(gc())
model_val <- xgb.DMatrix(data = data.matrix(processed_train_val), label = y2)
rm(processed_train_val)
invisible(gc())
xgb_test <- xgb.DMatrix(data = data.matrix(processed_test))
rm(processed_test)
invisible(gc())

params <- list(objective = "binary:logistic",
               booster = "gbtree",
               eval_metric = "auc",
               nthread = 7,
               eta = 0.05,
               max_depth = 10,
               gamma = 0.9,
               subsample = 0.8,
               colsample_bytree = 0.8,
               scale_pos_weight = 50,
               nrounds = 100)

myxgb_model <- xgb.train(params, model_train, params$nrounds, list(val = model_val), print_every_n = 20, early_stopping_rounds = 50)
imp <- xgb.importance(colnames(model_train), model=myxgb_model)
xgb.plot.importance(imp, top_n = 15)

cv <- xgb.cv(data = model_train, nrounds = 100, nthread = 7, nfold = 10, metrics = "auc",
             max_depth = 10, eta = 0.05, objective = "binary:logistic")
