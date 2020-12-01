# 训练样本, separate by space
train_samples = ["<=30 高 否 中 否",
                 "<=30 高 否 优 否",
                 "31~40 高 否 中 是",
                 ">40 中 否 中 是",
                 ">40 低 是 中 是",
                 ">40 低 是 优 否",
                 "31~40 低 是 优 是",
                 "<=30 中 否 中 否",
                 "<=30 低 是 中 是",
                 ">40 中 是 中 是",
                 "<=30 中 是 优 是",
                 "31~40 中 否 优 是",
                 "31~40 高 是 中 是",
                 ">40 中 否 优 否"]
# 待分类样本
test_sample = '<=30 中 是 中'

# convert training data and test data
from tabulate import tabulate  # to print lists in a pretty form

# 描述属性分别用数字替换
# 年龄, <=30-->0, 31~40-->1, >40-->2
# 收入, '低'-->0, '中'-->1, '高'-->2
# 是否学生, '是'-->0, '否'-->1
# 信誉: '中'-->0, '优'-->1
# 类别属性用数字替换
# 购买电脑是-->0, 不购买电脑否-->1
MAP_text2num = [{'<=30': 0, '31~40': 1, '>40': 2},
                {'低': 0, '中': 1, '高': 2},
                {'是': 0, '否': 1},
                {'中': 0, '优': 1},
                {'否': 0, '是': 1}]

# 下面步骤将文字，转化为对应数字
train_samples = [sample.split(' ') for sample in train_samples]
print(f"split train_samples:\n {tabulate(train_samples)}", '\n')
train_samples = [[MAP_text2num[i][attr] for i, attr in enumerate(sample)] for sample in train_samples]
print(f"split train_samples after mapping:\n {tabulate(train_samples)}", '\n')
# convert the test sample
test_sample = [MAP_text2num[i][attr] for i, attr in enumerate(test_sample.split(' '))]
print(f"split test sample after mapping:\n {test_sample}", '\n')

# statistics on the train set
# 单个样本的维度： 描述属性和类别属性个数
sample_dim = len(train_samples[0])

# 计算每个属性有哪些取值, initializing attr_list
attr_values_list = [[] for attr_index in range(sample_dim)]
# attr_values_list = [[]] * sample_dim # all elements will be the same
print(f"empty attr_values_list:\n{attr_values_list}\n")

for sample in train_samples:
    for attr_index in range(0, sample_dim):
        if sample[attr_index] not in attr_values_list[attr_index]:
            attr_values_list[attr_index].append(sample[attr_index])
print(f"attr_values_list:\n{tabulate(attr_values_list)}\n")
attr_values_list_sorted = [sorted(attr_values) for attr_values in attr_values_list]
# sort attributes
print(f"attr_values_list_sorted:\n{tabulate(attr_values_list_sorted)}\n")
# 每个属性取值的个数
attr_nums = [len(attr) for attr in attr_values_list_sorted]
print(f"attr_nums:\n{attr_nums}\n")

# 记录不同类别,category的样本个数
category_nums = [0] * attr_nums[-1]
print(f"all category_num:\n{category_nums}\n")
# 计算不同类别的样本个数, 是 or 否 for this example
for category_index, category in enumerate(attr_values_list_sorted[-1]):
    for sample in train_samples:
        if sample[-1] == category:
            category_nums[category_index] += 1
print(f"Times of sample for {attr_values_list_sorted[-1]} are:\n{category_nums}\n")

# 计算不同类别样本所占概率
category_prior_probability = [single_category_num / sum(category_nums) for single_category_num in category_nums]
print(f"Odds for {attr_values_list_sorted[-1]} are:\n{category_prior_probability}\n")

# 将用户按照购买电脑case分类
samples_grouped_by_category = {}
for category_name in attr_values_list_sorted[-1]:
    samples_grouped_by_category[category_name] = []
print(f"empty dictionary samples_grouped_by_category:\n{samples_grouped_by_category}\n")
for sample in train_samples:
    samples_grouped_by_category[sample[-1]].append(sample)
print(f"grouped samples_grouped_by_category:\n{tabulate(samples_grouped_by_category)}\n")

# 初始化后验概率
category_posterior_probability = []
for category_index in range(0, attr_nums[-1]):
    category_posterior_probability.append(category_prior_probability[category_index])
print(f"prior probabilities for {attr_values_list_sorted[-1]}:\n{category_posterior_probability}\n")

# 记录 每个类别的train样本中，取test样本的某个属性值的样本个数
train_sample_attr_nums = {}
for category_name in attr_values_list_sorted[-1]:
    train_sample_attr_nums[category_name] = [0] * (sample_dim - 1)
print(f"empty dictionary train_sample_attr_nums:\n{train_sample_attr_nums}\n")

# 计算 每个类别的训练样本中，取待分类样本的某个属性值的样本个数
for category_name in samples_grouped_by_category:
    samples_in_one_category = samples_grouped_by_category[category_name]
    for sample in samples_in_one_category:
        for attr_index in range(sample_dim - 1):
            if sample[attr_index] == test_sample[attr_index]:
                train_sample_attr_nums[category_name][attr_index] += 1
print(f"dictionary train_sample_attr_nums:\n{train_sample_attr_nums}\n")
# 字典转化为list
train_sample_attr_nums = list(train_sample_attr_nums.values())
print(f"list train_sample_attr_nums:\n{tabulate(train_sample_attr_nums)}\n")
# 计算后验概率
for category_index in range(0, attr_nums[-1]):
    train_test_attr_match_odds_list = \
        [_attr_num / category_nums[category_index] + 1 for _attr_num in train_sample_attr_nums[category_index]]
    for train_test_attr_match_odds in train_test_attr_match_odds_list:
        category_posterior_probability[category_index] *= train_test_attr_match_odds

print(f'posterior probabilities for {attr_values_list_sorted[-1]}:\n{category_posterior_probability}\n')

# 找到概率最大对应的那个类别，就是预测样本的分类情况
predict_class = category_posterior_probability.index(max(category_posterior_probability))
buy_or_not_dict = MAP_text2num[-1]
test_sample_predict = list(buy_or_not_dict.keys())[list(buy_or_not_dict.values()).index(predict_class)]
print(f'The test sample will buy or not: {test_sample_predict}')
