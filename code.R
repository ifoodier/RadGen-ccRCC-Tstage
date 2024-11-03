save(dat3,file="data0418.rdata")
load("data0418.rdata")

###### 3 分期 ######
##### 3.1模型 #####
## 构建数据
dat4 <- dat3 %>% select(分期,perinephric.stranding,MAPS,REL评分,脂蛋白.术前,低密度脂蛋白胆固醇.术前,
                          高密度脂蛋白胆固醇.术前,纤维蛋白原.术前,血小板计数.术前,TAM.术前,
                          白细胞.术前,单核细胞.术前,中性粒细胞.术前,血肌酐SCr.术后,尿素Urea.术后,
                          肾小球滤过率eGFR.术后) %>% mutate(REL评分=(REL评分<=6 & REL评分>=0)*1+(REL评分<=9 & REL评分>=7)*2++(REL评分>=10)*3) %>% 
  na.omit()
x2 <- colnames(dat4)[-1]
summary(dat4)

## 逐步回归
full=as.formula(paste0("分期","~", paste(x2, collapse= "+")))
m1=glm(full,data=dat4,family = binomial(link = "logit"))
m2=step(m1, direction = c("backward"),k=log(nrow(dat4)))
tmp <- summary(m2)$coefficients
summary(m2)$coefficients
stepname <- row.names(tmp)[-1]
step=as.formula(paste0("分期","~", paste(stepname, collapse= "+")))
m3=glm(step,data=dat4,family = binomial(link = "logit"))
summary(m3)


## 预测
pred3 <- predict(m3,dat4,type = c("response"))
mod3 <- roc(dat4$分期,pred3)
mod1.1 <- roc(dat4$分期,dat4$perinephric.stranding)
mod1.2 <- roc(dat4$分期,dat4$REL评分)
mod1.3 <- roc(dat4$分期,dat4$血小板计数.术前)


##### 3.2delong test #####
roc.test(mod3,mod1.1)
roc.test(mod3,mod1.2)
roc.test(mod3,mod1.3)

roc.test(mod1.1,mod1.2)
roc.test(mod1.1,mod1.3)
roc.test(mod1.2,mod1.3)

##### 3.3nom #####
ddist=datadist(dat4)
options(datadist="ddist")

step = as.formula(paste("分期~", paste(result3$var, collapse = "+")))
mod=lrm(step,dat4,x = TRUE, y = TRUE)

mod <- Newlabels(mod, c(REL评分="RNS",perinephric.stranding="PFS",
                        血小板计数.术前="Platelet counts"))

nom=nomogram(mod,fun=plogis,lp=F,fun.at=c(seq(0.1,0.9,by=0.1)),funlabel="Predicted Risk")
tiff(filename = "1130结果/nomogram-分期.tiff",width = 800, height = 800, units = "px",family="serif", pointsize = 18)
plot(nom,fun.side = c(1,3,1,3,1,3,1,3,1)) 
dev.off()
##### 3.4cal+hl #####
cal_train=calibrate(mod,method="boot",B=1000) 
tiff(filename = "1130结果/calibrate-分期.tiff",width = 800, height = 800, units = "px",family="serif", pointsize = 18)
plot(cal_train,xlab="Nomogram Predicted Probability",ylab="Actual Probability",
     subtitles=F,xlim=c(0,1),ylim=c(0,1),cex.axis=1.5,cex.lab=1.3,legend=T) 
dev.off()

hl3 <- hoslem.test(dat4$分期, pred3, g=10)
hl3

##### 3.5roc #####
cc1=NULL
cc2=NULL

ff=paste0(c("Integrated model:",sprintf("%0.3f", mod3$auc)," (",
            sprintf("%0.3f", ci.auc(mod3)[1]),"-",
            sprintf("%0.3f", ci.auc(mod3)[3]),")"),collapse="")
cc1=c(cc1,ff)
cc2=c(cc2,2)

ff=paste0(c("PFS:",sprintf("%0.3f", mod1.1$auc)," (",
            sprintf("%0.3f", ci.auc(mod1.1)[1]),"-",
            sprintf("%0.3f", ci.auc(mod1.1)[3]),")"),collapse="")
cc1=c(cc1,ff)
cc2=c(cc2,3)

ff=paste0(c("RNS:",sprintf("%0.3f", mod1.2$auc)," (",
            sprintf("%0.3f", ci.auc(mod1.2)[1]),"-",
            sprintf("%0.3f", ci.auc(mod1.2)[3]),")"),collapse="")
cc1=c(cc1,ff)
cc2=c(cc2,4)
ff=paste0(c("Platelet counts:",sprintf("%0.3f", mod1.3$auc)," (",
            sprintf("%0.3f", ci.auc(mod1.3)[1]),"-",
            sprintf("%0.3f", ci.auc(mod1.3)[3]),")"),collapse="")
cc1=c(cc1,ff)
cc2=c(cc2,5)

tiff(filename = "1130结果/ROC for 分期.tiff",width = 800, height = 800, units = "px",family="serif", pointsize = 18)
roc=plot.roc(mod3,col=2,
             legacy.axes=TRUE,cex.axis=1.0,cex.lab=1.0,main="")
lines.roc(mod1.1,col=3)
lines.roc(mod1.2,col=4)
lines.roc(mod1.3,col=5)
legend("bottomright",inset=.04,cc1,lty=1,col=cc2,title = "AUC(95%CI)",bty="n",cex=0.9)
dev.off()

#灵敏度+特异度
##总模型
plot.roc(mod3,col=2,legacy.axes=TRUE,cex.axis=1.0,cex.lab=1.0,main="", print.thres=TRUE)
##perinephric.stranding
plot.roc(mod1.1,col=2,legacy.axes=TRUE,cex.axis=1.0,cex.lab=1.0,main="", print.thres=TRUE)
##REL评分
plot.roc(mod1.2,col=2,legacy.axes=TRUE,cex.axis=1.0,cex.lab=1.0,main="", print.thres=TRUE)
##血小板计数.术前
plot.roc(mod1.3,col=2,legacy.axes=TRUE,cex.axis=1.0,cex.lab=1.0,main="", print.thres=TRUE)

##### 3.6cvroc #####
dat5 <- dat4 %>% rename(Y=分期) %>% mutate(pred=pred3) %>% select(Y,pred)

set.seed(1234)
ci_5_3 <- iid_example(data = dat5, V = 5)  
set.seed(1234)
ci_10_3 <- iid_example(data = dat5, V = 10)  

set.seed(1234)
cv_5_3 <- iid_example1(data = dat5, V = 5) 
set.seed(1234)
cv_10_3 <- iid_example1(data = dat5, V = 10) 

### 5 folds
cc1=NULL
cc2=NULL

ff=paste0(c("Integrated model:",sprintf("%0.3f", mod3$auc)," (",
            sprintf("%0.3f", ci.auc(mod3)[1]),"-",
            sprintf("%0.3f", ci.auc(mod3)[3]),")"),collapse="")
cc1=c(cc1,ff)
cc2=c(cc2,2)

ff=paste0(c("5 folds",":",sprintf("%0.3f", ci_5_3$cvAUC)," (",
            sprintf("%0.3f", ci_5_3$ci[1]),"-",
            sprintf("%0.3f", ci_5_3$ci[2]),")"),collapse="")
cc1=c(cc1,ff)
cc2=c(cc2,3)


tiff(filename = "1130结果/5 folds ROC for 分期.tiff",width = 800, height = 800, units = "px",family="serif", pointsize = 18)
plot(cv_5_3$perf, avg = "vertical",col = 3, lty = 1, main = "",xlab="1-specificity",ylab="Sensitivity",lwd=2)
lines(1-mod3$sensitivities,mod3$specificities,col=2,lwd=2)
abline(b=1,a=0,lwd=1)

legend("bottomright",inset=.04,cc1,lty=1,col=cc2,title = "AUC(95%CI)",bty="n",cex=0.9,lwd=2)
dev.off()

### 10 folds
cc1=NULL
cc2=NULL

ff=paste0(c("Integrated model:",sprintf("%0.3f", mod3$auc)," (",
            sprintf("%0.3f", ci.auc(mod3)[1]),"-",
            sprintf("%0.3f", ci.auc(mod3)[3]),")"),collapse="")
cc1=c(cc1,ff)
cc2=c(cc2,2)

ff=paste0(c("10 folds",":",sprintf("%0.3f", ci_10_3$cvAUC)," (",
            sprintf("%0.3f", ci_10_3$ci[1]),"-",
            sprintf("%0.3f", ci_10_3$ci[2]),")"),collapse="")
cc1=c(cc1,ff)
cc2=c(cc2,3)


tiff(filename = "1130结果/10 folds ROC for 分期.tiff",width = 800, height = 800, units = "px", family="serif",pointsize = 18)
plot(cv_10_3$perf, avg = "vertical",col = 3, lty = 1, main = "",xlab="1-specificity",ylab="Sensitivity",lwd=2)
lines(1-mod3$sensitivities,mod3$specificities,col=2,lwd=2)
abline(b=1,a=0,lwd=1)

legend("bottomright",inset=.04,cc1,lty=1,col=cc2,title = "AUC(95%CI)",bty="n",cex=0.9,lwd=2)
dev.off()

##### 3.7Bootstrap #####
dat6 <- dat4 %>% mutate(pred=pred3,label=分期)
##auc
set.seed(1234)
rs3<-boot(data=dat6,statistic = bs,R=1000)

cc1=NULL
cc2=NULL

ff=paste0(c("Integrated model:",sprintf("%0.3f", mod3$auc)," (",
            sprintf("%0.3f", ci.auc(mod3)[1]),"-",
            sprintf("%0.3f", ci.auc(mod3)[3]),")"),collapse="")
cc1=c(cc1,ff)
cc2=c(cc2,2)

ff=paste0(c("Bootstrap",":",sprintf("%0.3f", median(rs3$t))," (",
            sprintf("%0.3f", boot.ci(rs3,type = "perc",index = 1)$percent[4]),"-",
            sprintf("%0.3f", boot.ci(rs3,type = "perc",index = 1)$percent[5]),")"),collapse="")
cc1=c(cc1,ff)
cc2=c(cc2,3)

dat6$label=as.logical(dat6$label)
result.boot <- boot.roc(dat6$pred,dat6$label)
tiff(filename = "1130结果/Bootstrap roc for 分期.tiff",width = 800, height = 800, units = "px", family="serif",pointsize = 18) 
plot(result.boot$roc$FPR,result.boot$roc$TPR,type="s",col = 3, lty = 1, main = "",
     xlab="1-specificity",ylab="Sensitivity",lwd=2,
     ylim=c(0,1),xlim=c(0,1))

lines(1-mod3$sensitivities,mod3$specificities,col=2,lwd=2)
abline(b=1,a=0,lwd=1)

legend("bottomright",inset=.04,cc1,lty=1,col=cc2,title = "AUC(95%CI)",bty="n",cex=0.9,lwd=2)
dev.off()

##### 3.8DCA #####
dat4$pred3=pred3
N0=decision_curve(分期~pred3,data=dat4,family=binomial(link="logit"),
                    thresholds=seq(0,1,by=0.005),confidence.intervals=0.95,
                    study.design="case-control")
N1=decision_curve(分期~REL评分,data=dat4,family=binomial(link="logit"),
                    thresholds=seq(0,1,by=0.005),confidence.intervals=0.95,
                    study.design="case-control")  
N2=decision_curve(分期~perinephric.stranding,data=dat4,family=binomial(link="logit"),
                    thresholds=seq(0,1,by=0.005),confidence.intervals=0.95,
                    study.design="case-control") 
N3=decision_curve(分期~血小板计数.术前,data=dat4,family=binomial(link="logit"),
                    thresholds=seq(0,1,by=0.005),confidence.intervals=0.95,
                    study.design="case-control") 



# 画图
list=list(N0,N1,N2,N3)
tiff(filename = "1130结果/dca-分期.tiff",width = 800, height = 800, units = "px",family="serif", pointsize = 18)
plot_decision_curve(list,curve.names=c("Integrated model","RNS",
                                       "PFS","Platelet counts"),
                    cost.benefit.axis=FALSE,col=2:5,
                    confidence.intervals=FALSE,standardize=FALSE) #这条命令是画出来2条DCA，可以相互比较
dev.off()

tiff(filename = "1130结果/dca-分期+可信区间.tiff",width = 800, height = 800, units = "px", family="serif",pointsize = 18)
plot_decision_curve(N0,curve.names=c("Integrated model"),
                    cost.benefit.axis=FALSE,col=2,
                    confidence.intervals=T,standardize=FALSE) #这条命令是画出来2条DCA，可以相互比较
dev.off()