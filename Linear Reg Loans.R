
setwd('C:/Users/Nithin/Downloads/LR_Loans')

#**************************************************************************************************************#

#########################
#-->Required Packages<--#
#########################
require(dplyr)
require(stringr)
require(fastDummies)
require(ggplot2)
require(caret)
require(car)
require(Metrics)
require(MLmetrics)
require(sqldf)

#**************************************************************************************************************#

################
#-->Datasets<--#
################

Loans <- read.csv('LoansData.csv')

#**************************************************************************************************************#

#################
#-->Data Prep<--#
#################

str(Loans)

Loans$Interest.Rate <- as.numeric(str_replace(Loans$Interest.Rate,'%',''))
Loans$Loan.Length <- as.numeric(str_replace(Loans$Loan.Length,' months',''))
Loans$Debt.To.Income.Ratio <- as.numeric(str_replace(Loans$Debt.To.Income.Ratio,'%',''))

Loans$Employment.Length <- str_replace(Loans$Employment.Length,' years','')
Loans$Employment.Length <- str_replace(Loans$Employment.Length,' year','')
Loans$Employment.Length <- str_replace(Loans$Employment.Length,'< ','')
Loans$Employment.Length <- as.numeric(str_replace(Loans$Employment.Length,'\\+',''))

Fico <- data.frame(str_split_fixed(Loans$FICO.Range,'-',2))
Fico$X1 <- as.numeric(Fico$X1)
Fico$X2 <- as.numeric(Fico$X2)
Loans['Fico_avg'] <- (Fico$X1 + Fico$X2)/2
rm(Fico)

#Removed Columns
Loans$LoanID <- NULL
Loans$FICO.Range <- NULL

#*#**************************************************************************************************************#

#############
#--> UDF <--#
#############

#cont_var_summary
cont_var_summary <- function(x){
  n = length(x)
  nmiss = sum(is.na(x))
  nmiss_pct = mean(is.na(x))
  sum = sum(x, na.rm=T)
  mean = mean(x, na.rm=T)
  median = quantile(x, p=0.5, na.rm=T)
  std = sd(x, na.rm=T)
  var = var(x, na.rm=T)
  range = max(x, na.rm=T)-min(x, na.rm=T)
  pctl = quantile(x, p=c(0, 0.01, 0.05,0.1,0.25,0.5, 0.75,0.9,0.95,0.99,1), na.rm=T)
  return(c(N=n, Nmiss =nmiss, Nmiss_pct = nmiss_pct, sum=sum, avg=mean, meidan=median, std=std, var=var, range=range, pctl=pctl))
}

#outlier_treatment
outlier_treatment <- function(x){
  UC = quantile(x, p=0.99, na.rm=T)
  LC = quantile(x, p=0.01, na.rm=T)
  x = ifelse(x>UC, UC, x)
  x = ifelse(x<LC, LC, x)
  return(x)
}

#missing_value_treatment continuous
missing_value_treatment = function(x){
  x[is.na(x)] = mean(x, na.rm=T)
  return(x)
}

#mode for categorical
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#missing_value_treatment categorical
missing_value_treatment_categorical <- function(x){
  x[is.na(x)] <- Mode(na.omit(x))
  return(x)
}

#*#**************************************************************************************************************#

#######################
#-->Data Treatment <--#
#######################

cont_col <- c(colnames(select_if(Loans,is.numeric)),colnames(select_if(Loans,is.integer)))
cat_col <- colnames(select_if(Loans,is.character))

Loans_cont <- Loans[,cont_col]
Loans_cat <- Loans[,cat_col]

#Outlier Treatment & Missing Value treatment for continuous variables

num_sum <- data.frame(t(round(apply(Loans_cont,2,cont_var_summary),2)))

Loans_cont <- data.frame(apply(Loans_cont,2,outlier_treatment))
Loans_cont <- data.frame(apply(Loans_cont,2,missing_value_treatment))

#Mode Treatment for categorical variables

Loans_cat <- data.frame(apply(Loans_cat,2,missing_value_treatment_categorical))

#*#**************************************************************************************************************#

##########################
#--> Dummies Creation <--#
##########################

Loans_cat <- fastDummies::dummy_cols(Loans_cat,remove_first_dummy = TRUE)

Loans_cat <- select(Loans_cat,-cat_col)

Loans_clean <- cbind(Loans_cont,Loans_cat)

#*#**************************************************************************************************************#

#####################
#--> ASSUMPTIONS <--#
#####################

#target should be ND

ggplot(Loans_clean) + aes(Interest.Rate) + geom_histogram(bins = 10,fill = 'blue',color = 'white')

#To normalise we take Log

Loans_clean['ln_Intrest_rate'] <- log(Loans_clean$Interest.Rate)

ggplot(Loans_clean) + aes(ln_Intrest_rate) + geom_histogram(bins = 10,fill = 'blue',color = 'white')

#Corelation Between x & y ,x & x variables

corel_matrix <- data.frame(round(cor(Loans_clean),2))

#Amount.Funded.By.Investors has high corelation so to be dropped

#*#**************************************************************************************************************#

###########################
#--> Feature Reduction <--#
###########################

feat <- data.matrix(select(Loans_clean,-Interest.Rate))
target <- data.matrix(select(Loans_clean,Interest.Rate))

set.seed(12345)

#--> Stepwise <--#

#Full & Empty model
m_full <- lm(Interest.Rate~.,Loans_clean)
m_null <- lm(Interest.Rate~1,Loans_clean)

stepwise_feat <- step(m_null,scope = list(upper = m_full),data = Loans_clean, direction = 'both')

#ln_Intrest_rate + Loan.Length + Amount.Funded.By.Investors + 
#Amount.Requested + Open.CREDIT.Lines + Employment.Length + 
#State_CA + Revolving.CREDIT.Balance + Monthly.Income + Loan.Purpose_medical + 
#State_DE + Home.Ownership_OWN + Loan.Purpose_moving + Loan.Purpose_other + 
#State_MI + State_VT

#--> RFE <--#

rfe_model <- rfe(feat,target,sizes = c(1:78),rfeControl=rfeControl(functions = lmFuncs))

rfe_feat <- update(rfe_model,feat,target,size = 10)
rfe_feat[["bestVar"]]

#'ln_Intrest_rate','State_IA''State_VT','Home.Ownership_OTHER',
#'State_DE','Home.Ownership_NONE','State_IN''State_WY',
#'State_SD''State_UT'  

feat_selected <- c(
  'ln_Intrest_rate','State_IA','State_VT','Home.Ownership_OTHER',
  'State_DE','Home.Ownership_NONE','State_IN','State_WY',
  'State_SD','State_UT',
  
  'ln_Intrest_rate','Loan.Length','Amount.Funded.By.Investors',
  'Amount.Requested','Open.CREDIT.Lines','Employment.Length',
  'State_CA','Revolving.CREDIT.Balance','Monthly.Income','Loan.Purpose_medical',
  'State_DE','Home.Ownership_OWN','Loan.Purpose_moving','Loan.Purpose_other',
  'State_MI','State_VT',
  
  'Interest.Rate'
  )

feat_selected <- feat_selected[!duplicated(feat_selected)]

Loans_clean_selected <- Loans_clean[,feat_selected]

#--> LASSO <--#
lasso = train(Interest.Rate~.,
              data=Loans_clean_selected,method='glmnet',
              trControl = trainControl(method="none"),
              tuneGrid=expand.grid(alpha=1,lambda=0.02))

coef(lasso$finalModel, s = lasso$bestTune$lambda)

#'State_IA','Home.Ownership_OTHER','Home.Ownership_NONE','State_IN','State_WY',
#'State_SD','State_UT','Amount.Funded.By.Investors'

lasso_rej <- c('State_IA','Home.Ownership_OTHER','Home.Ownership_NONE','State_IN','State_WY',
               'State_SD','State_UT','Amount.Funded.By.Investors')

Loans_clean_selected <- dplyr::select(Loans_clean_selected,-lasso_rej)

#--> VIF <--#
m_full <- lm(Interest.Rate~.,data = Loans_clean_selected)
vif(m_full)

#*#**************************************************************************************************************#

########################
#--> Data Splitting <--#
########################

samp <- sample(1:nrow(Loans_clean_selected), floor(nrow(Loans_clean_selected)*0.7))

dev <-Loans_clean_selected[samp,]
val <-Loans_clean_selected[-samp,]

#*#**************************************************************************************************************#

########################
#--> Model Building <--#
########################

M0 <- lm(Interest.Rate~ln_Intrest_rate+
           State_VT+
           Loan.Length+
           Amount.Requested+
           Open.CREDIT.Lines+
           Employment.Length+
           State_CA+
           Revolving.CREDIT.Balance+
           Monthly.Income+
           Loan.Purpose_medical,data = dev)

summary(M0)

#--> Columns Removed <--#
#Home.Ownership_OWN
#State_MI
#Loan.Purpose_moving
#Loan.Purpose_other
#State_DE

dev <- data.frame(cbind(dev,pred = predict(M0)))
val <- data.frame(cbind(val,pred = predict(M0, newdata = val)))

#*#*#**************************************************************************************************************#

#*#####################
#--> Model Scoring <--#
#######################

#--> MAPE <--#
mape(dev$Interest.Rate,dev$pred)
mape(val$Interest.Rate,val$pred)

#--> RMSE <--#
rmse(dev$Interest.Rate,dev$pred)
rmse(val$Interest.Rate,val$pred)

#--> R^2 <--#
MLmetrics::R2_Score(dev$pred,dev$Interest.Rate)
MLmetrics::R2_Score(val$pred,val$Interest.Rate)

#*#*#**************************************************************************************************************#

#*#######################
#--> Cook's Distance <--#
#########################

#To Reduce Error

cd <- cooks.distance(M0)

plot(cd,pch = '*',cex = 2,main = 'Influencers')
abline(h = 4/nrow(dev),col = 'red')

#Remove Influential outliers
influerncers <- as.numeric(names(cd)[cd>(4/nrow(dev))])

dev2 <- dev[-influerncers,]
dev2$pred <- NULL
val$pred <- NULL

M1 <- lm(Interest.Rate~ln_Intrest_rate+
           State_VT+
           Loan.Length+
           Amount.Requested+
           Open.CREDIT.Lines+
           Employment.Length+
           State_CA+
           Revolving.CREDIT.Balance+
           Monthly.Income+
           Loan.Purpose_medical,data = dev2)

summary(M1)

dev2 <- data.frame(cbind(dev2,pred = predict(M1)))
val2 <- data.frame(cbind(val,pred = predict(M1, newdata = val)))

#*#*#**************************************************************************************************************#

#*#####################
#--> Model Scoring <--#
#######################

#--> MAPE <--#
mape(dev2$Interest.Rate,dev2$pred)
mape(val2$Interest.Rate,val2$pred)

#--> RMSE <--#
rmse(dev2$Interest.Rate,dev2$pred)
rmse(val2$Interest.Rate,val2$pred)

#--> R^2 <--#
MLmetrics::R2_Score(dev2$pred,dev2$Interest.Rate)
MLmetrics::R2_Score(val2$pred,val2$Interest.Rate)

#*#*#**************************************************************************************************************#

#*#######################
#--> Decile Analysis <--#
#########################

dev2.1 <- dev2

#Deciles
dec <- quantile(dev2.1$pred,probs = seq(0.1,0.9,by=0.1))

#intervals
dev2.1$decile <- findInterval(dev2.1$pred,c(-Inf,dec,Inf))

#to check deciles
xtabs(~decile,dev2.1)

dev2_1 <- dev2.1[,c("decile","Interest.Rate","pred")]
colnames(dev2_1) <- c('decile','Interest_rate','pred')

dev_dec <- sqldf::sqldf(
                        " select decile,
                         count(decile) cnt,
                         avg(pred) as avg_pred_Y,
                         avg(Interest_rate) avg
                         from dev2_1
                         group by decile 
                         order by decile")

writexl::write_xlsx(dev_dec,'DA_dev.xlsx')

val2.1 <- val2

#Deciles
dec <- quantile(val2.1$pred,probs = seq(0.1,0.9,by=0.1))

#intervals
val2.1$decile <- findInterval(val2.1$pred,c(-Inf,dec,Inf))

#to check deciles
xtabs(~decile,val2.1)

val2_1 <- val2.1[,c("decile","Interest.Rate","pred")]
colnames(val2_1) <- c('decile','Interest_rate','pred')

val_dec <- sqldf::sqldf(
  " select decile,
                         count(decile) cnt,
                         avg(pred) as avg_pred_Y,
                         avg(Interest_rate) avg
                         from val2_1
                         group by decile 
                         order by decile")                      

writexl::write_xlsx(val_dec,'DA_val.xlsx')

#*#*#**************************************************************************************************************#

#*#########################
#--> Model Diagnostics <--#
###########################

coefficients(M1) # model coefficients
confint(M1, level=0.95) # CIs for model parameters 
fitted(M1) # predicted values
residuals(M1) # residuals
anova(M1) # anova table 
vcov(M1) # covariance matrix for model parameters 
influence(M1) # regression diagnostics

#*#*#**************************************************************************************************************#