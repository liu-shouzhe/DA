# In this ab test, we use the dataset of a grocery website
# the aim is to verify if different layout of web page on serval different servers could affect the ratio of user clicked on the loyalty program page
# we use to sided test method because we do not know different lay-outs have bad or good effect
# 假设server1上的页面布局是旧的，而server2和server3上的页面布局是两种不同的新布局
# 对照组（control组）：看到的是旧版页面     实验组（treatment组）：看到的是新版页面
# 那么可以设计原假设H0为 P = P0 即转化率不变，备择假设为 P != P0 即转化率改变，显著性水平α = 0.1，统计功效 1-β = 0.8
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt

# 引入数据
df = pd.read_csv('grocerywebsiteabtestdata.csv')
# print(df.head())

# 访问页面的dataframe
group_v = df[df['VisitPageFlag'] == 1].groupby('ServerID').aggregate('count')
# 取出对照组和实验组访问页面的人数
group_v_c = group_v.iloc[0,2]
group_v_t = group_v.iloc[1,2] + group_v.iloc[2,2]

# 访问页面的dataframe
group_uv = df[df['VisitPageFlag'] == 0].groupby('ServerID').aggregate('count')
# 取出对照组和实验组访问页面的人数
group_uv_c = group_uv.iloc[0,2]
group_uv_t = group_uv.iloc[1,2] + group_uv.iloc[2,2]
# print(group_uv_t)

# 计算方差
# 这里因为没有经验值，我们直接用对照组的频率作为概率
pa = group_v_c/(group_v_c + group_uv_c)
print(pa)
# 这里pb是估计值，我们期望转化率增加2%
pb = pa + 0.02

# 上面的所有计算实际都是不必要的，在实际生产中，应当提供PA的经验值和PB的预计值
# 样本容量根据计算方法计算
effect_size = sms.proportion_effectsize(pa, pb)
required_n = sms.NormalIndPower().solve_power(
    effect_size,
    power=0.8,
    alpha=0.1,
    ratio=1
)
# 这里可以看出样本容量对照组和实验组各2200条就足够了，需求远小于数据集的数据数量
# 在实际生产中，我们是在收集数据前完成样本预估和时间成本估计的，故这一步是非常必要的
# print(required_n)

# 完成这一步后,可以在线上完成AB实验设计或者自己设计AB测试


# 下面真正的ABtest就开始了
# 数据已经在上方引入,不再重复

# print(df.info())
# # 检查数据是否有缺失值,确定数据类型
# # 发现无缺失值,不需要用isnull函数查看了

# # 检查数据是否有重复值
# print(df.duplicated().sum())

# 这里发现IP地址大量重复，对于AB测试，删除这些重复的用户是最好的
# print(df['IP Address'].duplicated().sum())
del_IP = df[df['IP Address'].duplicated()]['IP Address'].values
df['IP Address'].isin(del_IP)
df_clear = df[~df['IP Address'].isin(del_IP)]
# print(df_clear)

# 抽样，实际生产中无需抽样，这里因为数据较多，抽取部分做为用例
control_sample = df_clear[df_clear['ServerID'] == 1].sample(n=round(required_n), random_state=22)
treatment_sample = df_clear[(df_clear['ServerID'] == 2) | (df_clear['ServerID'] == 3)].sample(n=round(required_n), random_state=22)
control_sample.insert(loc=5, column='group', value='control')
treatment_sample.insert(loc=5, column='group', value='treatment')
abtest = pd.concat([control_sample, treatment_sample], axis=0)
# drop 表示直接删除原来的索引， inplace 表示直接在原位置进行替换
abtest.reset_index(drop=True, inplace=True)

# 检查数据
# print(abtest)

# 计算转化率和方差
conversion_rate = abtest.groupby('group')['VisitPageFlag'].agg([np.mean, np.std])
conversion_rate.columns = ['conversion_rate', 'std_deviation']
# print(conversion_rate)

# 假设检验，使用Z检验
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# 取出需要的数据列
control_converted = control_sample['VisitPageFlag']
treatment_converted = treatment_sample['VisitPageFlag']

# 求算总数和转换成功的用户数
nobs = [control_converted.count(), treatment_converted.count()]
success = [control_converted.sum() ,treatment_converted.sum()]

z_stat, pval = proportions_ztest(success, nobs=nobs)
(lower_con, lower_tre), (upper_con, upper_tre) = proportion_confint(success, nobs=nobs, alpha=0.1)

print(z_stat, pval)
print(lower_con, upper_con)
print(lower_tre, upper_tre)
# 可以得知， p值是远小于显著水平α的，故应该拒绝原假设，即新版页面和原页面有显著差异
# 原转化率为pa = 0.0676, 我们期望的转化率为0.0876
# 对照组的实验包含了原转化率而不包含期望转化率,置信区间应当是合适的
# 然而实验组的置信区间远小于原转化率,更小于期望转化率,故实际上新版的页面大大降低了转化率

