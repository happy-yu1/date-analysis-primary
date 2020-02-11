#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
matplotlib.use('TkAgg')#使用Windows不需要这个命令
import matplotlib.pyplot as plt
plt.rc('font', family='Microsoft YaHei') 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import pandas as pd    
import os
os.chdir('/Users/chenjiaju/Desktop/data')


# In[2]:


df=pd.read_csv('cs-training.csv')


# In[3]:


states={"Unnamed: 0":"用户ID",
        "SeriousDlqin2yrs":"好坏客户",
        "RevolvingUtilizationOfUnsecuredLines":"可用额度比值",
        "age":"年龄",
        "NumberOfTime30-59DaysPastDueNotWorse":"逾期30-59天笔数",
        "DebtRatio":"负债率",
        "MonthlyIncome":"月收入",
        "NumberOfOpenCreditLinesAndLoans":"信贷数量",
        "NumberOfTimes90DaysLate":"逾期90天笔数",
        "NumberRealEstateLoansOrLines":"固定资产贷款量",
        "NumberOfTime60-89DaysPastDueNotWorse":"逾期60-89天笔数",
        "NumberOfDependents":"家属数量"}
df.rename(columns=states,inplace=True)


# In[4]:


df


# In[5]:


print(df.columns)


# In[6]:


df.info()


# In[7]:


def missing_values_table(df):
        # 全部缺失值
        mis_val = df.isnull().sum()
        
        # 缺失值比例
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # 做成一个表
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # 改列名
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : '缺失值', 1 : '缺失比例'})
        
        # 对缺失值排序
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '缺失比例', ascending=False).round(1)
        
        # 打印出表
        print ("数据有" + str(df.shape[1]) + " 列.\n"      
            "其中 " + str(mis_val_table_ren_columns.shape[0]) +
              " 列含有缺失值.")
        
        # 返回确实行列
        return mis_val_table_ren_columns###显示缺失值和表示缺失值函数


# In[8]:


df=df.fillna(df.median())#补充中位数


# In[9]:


age_cut=pd.cut(df["年龄"],5)
age_cut_grouped=df["好坏客户"].groupby(age_cut).count()
age_cut_grouped1=df["好坏客户"].groupby(age_cut).sum()
df2=pd.merge(pd.DataFrame(age_cut_grouped), pd.DataFrame(age_cut_grouped1),right_index=True,left_index=True)
df2.rename(columns={"好坏客户_x":"好客户","好坏客户_y":"坏客户"},inplace=True)
df2.insert(2,"坏客户率",df2["坏客户"]/df2["好客户"])
ax1=df2[["好客户","坏客户"]].plot.bar()
ax1.set_xticklabels(df2.index,rotation=15)
ax1.set_ylabel("客户数")
ax1.set_title("年龄与好坏客户数分布图")


# In[10]:


ax11=df2["坏客户率"].plot()
ax11.set_ylabel("坏客户率")
ax11.set_title("坏客户率随年龄的变化趋势图")


# In[11]:


plt.rcParams["font.sans-serif"]='SimHei'
plt.rcParams['axes.unicode_minus'] = False
corr = df.corr()#计算各变量的相关性系数
xticks = list(corr.index)#x轴标签
yticks = list(corr.index)#y轴标签
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap="rainbow",ax=ax1,linewidths=.5, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
ax1.set_xticklabels(xticks, rotation=35, fontsize=10)
ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
plt.show()


# In[12]:


#数据分箱
cut1=pd.qcut(df["可用额度比值"],4,labels=False)
cut2=pd.qcut(df["年龄"],8,labels=False)
bins3=[-1,0,1,3,5,13]
cut3=pd.cut(df["逾期30-59天笔数"],bins3,labels=False)
cut4=pd.qcut(df["负债率"],3,labels=False)
cut5=pd.qcut(df["月收入"],4,labels=False)
cut6=pd.qcut(df["信贷数量"],4,labels=False)
bins7=[-1, 0, 1, 3,5, 20]
cut7=pd.cut(df["逾期90天笔数"],bins7,labels=False)
bins8=[-1, 0,1,2, 3, 33]
cut8=pd.cut(df["固定资产贷款量"],bins8,labels=False)
bins9=[-1, 0, 1, 3, 12]
cut9=pd.cut(df["逾期60-89天笔数"],bins9,labels=False)
bins10=[-1, 0, 1, 2, 3, 5, 21]
cut10=pd.cut(df["家属数量"],bins10,labels=False)


# In[13]:


#好坏客户比率
rate=df["好坏客户"].sum()/(df["好坏客户"].count()-df["好坏客户"].sum())

#定义woe计算函数
def get_woe_data(cut):
    grouped=df["好坏客户"].groupby(cut,as_index = True).value_counts()
    woe=np.log(pd.DataFrame(grouped).unstack().iloc[:,1]/pd.DataFrame(grouped).unstack().iloc[:,0]/rate)#计算每个分组的woe值
    return(woe)
cut1_woe=get_woe_data(cut1)
cut2_woe=get_woe_data(cut2)
cut3_woe=get_woe_data(cut3)
cut4_woe=get_woe_data(cut4)
cut5_woe=get_woe_data(cut5)
cut6_woe=get_woe_data(cut6)
cut7_woe=get_woe_data(cut7)
cut8_woe=get_woe_data(cut8)
cut9_woe=get_woe_data(cut9)
cut10_woe=get_woe_data(cut10)


# In[14]:


def mono_bin(Y, X, n = 20):
    r = 0
    bad=Y.sum()
    good=Y.count()-good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe']=np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)
    print("=" * 60)
    print(d4)
    return(d4)


# In[15]:


pinf = float('inf') #正无穷大
ninf = float('-inf') #负无穷大


# In[16]:


pinf = float('inf') #正无穷大
ninf = float('-inf') #负无穷大


# In[17]:


cut1=pd.qcut(df["可用额度比值"],4,labels=False)
cut2=pd.qcut(df["年龄"],8,labels=False)
bins3=[ninf, 0, 1, 3, 5, pinf]
cut3=pd.cut(df["逾期30-59天笔数"],bins3,labels=False)
cut4=pd.qcut(df["负债率"],3,labels=False)
cut5=pd.qcut(df["月收入"],4,labels=False)
cut6=pd.qcut(df["信贷数量"],4,labels=False)
bins7=[ninf, 0, 1, 3, 5, pinf]
cut7=pd.cut(df["逾期90天笔数"],bins7,labels=False)
bins8=[ninf, 0,1,2, 3, pinf]
cut8=pd.cut(df["固定资产贷款量"],bins8,labels=False)
bins9=[ninf, 0, 1, 3, pinf]
cut9=pd.cut(df["逾期60-89天笔数"],bins9,labels=False)
bins10=[ninf, 0, 1, 2, 3, 5, pinf]
cut10=pd.cut(df["家属数量"],bins10,labels=False)


# In[18]:


#好坏客户比率
rate=df["好坏客户"].sum()/(df["好坏客户"].count()-df["好坏客户"].sum())

#定义woe计算函数
def get_woe_data(cut):
    grouped=df["好坏客户"].groupby(cut,as_index = True).value_counts()
    woe=np.log(pd.DataFrame(grouped).unstack().iloc[:,1]/pd.DataFrame(grouped).unstack().iloc[:,0]/rate)#计算每个分组的woe值
    return(woe)
cut1_woe=get_woe_data(cut1)
cut2_woe=get_woe_data(cut2)
cut3_woe=get_woe_data(cut3)
cut4_woe=get_woe_data(cut4)
cut5_woe=get_woe_data(cut5)
cut6_woe=get_woe_data(cut6)
cut7_woe=get_woe_data(cut7)
cut8_woe=get_woe_data(cut8)
cut9_woe=get_woe_data(cut9)
cut10_woe=get_woe_data(cut10)


# In[19]:


#定义IV值计算函数
def get_IV_data(cut,cut_woe):
    grouped=df["好坏客户"].groupby(cut,as_index = True).value_counts()
    cut_IV=((pd.DataFrame(grouped).unstack().iloc[:,1]/df["好坏客户"].sum()-pd.DataFrame(grouped).unstack().iloc[:,0]/
             (df["好坏客户"].count()-df["好坏客户"].sum()))*cut_woe).sum()
    return(cut_IV)

#计算各分组的IV值
cut1_IV=get_IV_data(cut1,cut1_woe)
cut2_IV=get_IV_data(cut2,cut2_woe)
cut3_IV=get_IV_data(cut3,cut3_woe)
cut4_IV=get_IV_data(cut4,cut4_woe)
cut5_IV=get_IV_data(cut5,cut5_woe)
cut6_IV=get_IV_data(cut6,cut6_woe)
cut7_IV=get_IV_data(cut7,cut7_woe)
cut8_IV=get_IV_data(cut8,cut8_woe)
cut9_IV=get_IV_data(cut9,cut9_woe)
cut10_IV=get_IV_data(cut10,cut10_woe)

#各组的IV值可视化
df_IV=pd.DataFrame([cut1_IV,cut2_IV,cut3_IV,cut4_IV,cut5_IV,cut6_IV,cut7_IV,cut8_IV,cut9_IV,cut10_IV],index=df.columns[2:])
df_IV.plot(kind="bar")
for a,b in zip(range(10),df2.values):
    plt.text(a,b,'%.2f' % b, ha='center', va= 'bottom',fontsize=9)


# In[20]:


df_new=df.drop(["负债率","月收入","信贷数量","固定资产贷款量","家属数量","用户ID"],axis=1)


# In[21]:


def replace_data(cut,cut_woe):
    a=[]
    for i in cut.unique():
        a.append(i)
        a.sort()
    for m in range(len(a)):
        cut.replace(a[m],cut_woe.values[m],inplace=True)
    return cut

#进行替换
df_new["可用额度比值"]=replace_data(cut1,cut1_woe)
df_new["年龄"]=replace_data(cut2,cut2_woe)
df_new["逾期30-59天笔数"]=replace_data(cut3,cut3_woe)
df_new["逾期90天笔数"]=replace_data(cut7,cut7_woe)
df_new["逾期60-89天笔数"]=replace_data(cut9,cut9_woe)


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

x=df_new.iloc[:,1:]
y=df_new.iloc[:,0]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

#模型训练
model=LogisticRegression()
clf=model.fit(x_train,y_train)
print("测试成绩:{}".format(clf.score(x_test,y_test)))
y_pred=clf.predict(x_test)
y_pred1=clf.decision_function(x_test)

#绘制ROC曲线以及计算AUC值
fpr, tpr, threshold = roc_curve(y_test, y_pred1)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',
          label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_curve')
plt.legend(loc="lower right")
plt.show()


# In[23]:


coe=model.coef_


# In[24]:


import numpy as np
factor = 20 / np.log(2)
offset = 600 - 20 * np.log(20) / np.log(2)

#定义变量分数计算函数
def get_score(coe,woe,factor):
    scores=[]
    for w in woe:
        score=round(coe*w*factor,0)
        scores.append(score)
    return scores

#计算每个变量得分
x1 = get_score(coe[0][0], cut1_woe, factor)
x2 = get_score(coe[0][1], cut2_woe, factor)
x3 = get_score(coe[0][2], cut3_woe, factor)
x7 = get_score(coe[0][3], cut7_woe, factor)
x9 = get_score(coe[0][4], cut9_woe, factor)

#打印输出每个特征对应的分数
print("可用额度比值对应的分数:{}".format(x1))
print("年龄对应的分数:{}".format(x2))
print("逾期30-59天笔数对应的分数:{}".format(x3))
print("逾期90天笔数对应的分数:{}".format(x7))
print("逾期60-89天笔数对应的分数:{}".format(x9))


# In[25]:


w=pd.read_csv('cs-training.csv')
w['age'].hist(bins=50)


# In[26]:


def cap(x,quantile=[0.01,0.99]):
    Q01,Q99=x.quantile(quantile).values.tolist()
    if Q01>x.min():
        x=x.copy()
        x.loc[x<Q01]=Q01
    if Q99<x.max():
        x=x.copy()
        x.loc[x>Q99]=Q99
    return(x)


# In[27]:


w['age'].hist(bins=50)

