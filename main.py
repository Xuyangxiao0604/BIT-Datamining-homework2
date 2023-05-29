import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from collections import defaultdict


# 训练集和测试集数据路径
data_path = 'anonymous-msweb.data'
test_path = 'anonymous-msweb.test'

data = open(data_path).readlines()
A_count = 0  # 记录属性数
C_count = 0  # 记录case数
V_count = 0  # 记录投票数
Attributes = {}  # 属性与编号映射字典
Cases = []  # 案例号
Votes = []  # 各案例投票

# 统计data数据
for i in range(len(data)):
    item = data[i].split(',')
    if item[0] == 'A':
        A_count += 1
        Attributes[int(item[1])] = item[3]
    if item[0] == 'C':
        C_count += 1
        Cases.append(int(item[1].split('"')[1]))
        vote = []
        while (i < len(data) - 1):
            i += 1
            if data[i][0] != 'V':
                break
            V_count += 1
            vote.append(int(data[i].split(',')[1]))
        Votes.append(vote)

# 统计test数据
data = open(test_path).readlines()
for i in range(len(data)):
    item = data[i].split(',')
    if item[0] == 'A':
        A_count += 1
        Attributes[int(item[1])] = item[3]
    if item[0] == 'C':
        C_count += 1
        Cases.append(int(item[1].split('"')[1]))
        vote = []
        while (i < len(data) - 1):
            i += 1
            if data[i][0] != 'V':
                break
            V_count += 1
            vote.append(int(data[i].split(',')[1]))
        Votes.append(vote)

    
# 打印基本信息
print('数据集属性A数量:', A_count)
print('数据集属性C数量:', C_count)
print('数据集属性V数量:', V_count)

print("\n去重之后：")
# 统计属性种数
arr_unique = set()
for attributes in Attributes: # type: ignore
    arr_unique.add(attributes)
print('所有的属性有 %d 种' % len(arr_unique))

# 统计属性种数
case_unique = set()
for case in Cases:
    case_unique.add(case)
print('所有的投票用户有 %d 个' % len(case_unique))

# 统计投票属性种数
vote_unique = set()
for Vote in Votes:
    for item in Vote:
        vote_unique.add(item)
print('所有被投票的属性有 %d 种' % len(vote_unique))

# 数据可视化
votes_every_attr = defaultdict(int)  # 每个属性获得的投票数
votes_every_case = []  # 每个用户的投票数
# 遍历数据项
for item in Votes:
    votes_every_case.append(len(item))
    for attr in item:
        votes_every_attr[attr] += 1
# 画出不同投票数的用户的分布
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

lens, counts = np.unique(np.array(votes_every_case), return_counts=True)
plt.bar(lens, counts / len(Votes) * 100)
plt.title('用户投票数分布')
plt.xlabel('投票数量')
plt.ylabel('所占用户百分比%')
# 画出不同属性的投票数分布
plt.figure(2)
attrs = list(votes_every_attr.keys())
vote_counts = list(votes_every_attr.values())
plt.bar(attrs, np.array(vote_counts) / len(Votes) * 100)
plt.title('属性投票数分布')
plt.xlabel('属性编号')
plt.ylabel('所占投票数百分比%')
plt.show(block=True)


te = TransactionEncoder()  
df_tf = te.fit_transform(Votes)  
df = pd.DataFrame(df_tf, columns = te.columns_) # type: ignore


# 计算频繁项集
frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
print("\n频繁项集如下：")
print(frequent_itemsets)  



# 计算关联规则
association_rule = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.8)  # 关联规则发掘，置信度阈值为0.8 
print("\n关联规则如下：")
print(association_rule)
