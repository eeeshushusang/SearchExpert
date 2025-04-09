import re, random

# 将标准答案和学生答案分别存储为多行字符串
standard_text = """
答案汇总:

"""

student_text = """
以下为学生答案:


"""


def parse_answers(text):
    pattern = r"(\d+)\.\s*([A-E])"
    matches = re.findall(pattern, text)
    answers = {int(num): ans for num, ans in matches}
    return answers


# 解析标准答案和学生答案
standard_answers = parse_answers(standard_text)
student_answers = parse_answers(student_text)

# 比较答案
correct = 0
incorrect_questions = []
total_questions = len(standard_answers)
answered_questions = len(student_answers)

for num in standard_answers:
    if num in student_answers:
        if student_answers[num] == standard_answers[num]:
            correct += 1
        else:
            incorrect_questions.append(num)
    else:
        incorrect_questions.append(num)

# 计算正确率
correct_rate = correct / answered_questions

# 输出结果
print(f"总题数: {total_questions}")
print(f"答对题数: {correct}")
print(f"答错题数: {answered_questions - correct}")
print(f"正确率: {correct_rate * 100:.2f}%")

# 输出答错的题号汇总
# print('答错的题号:', ', '.join(map(str, sorted(incorrect_questions))))


# Bootstrap 方差计算
def bootstrap_variance(standard_answers, student_answers, num_bootstrap=1000):
    """
    通过 Bootstrap 方法计算学生正确率的方差。

    :param standard_answers: 标准答案字典
    :param student_answers: 学生答案字典
    :param num_bootstrap: 重采样次数
    :return: 正确率的方差
    """
    correct_rates = []
    questions = list(standard_answers.keys())

    for _ in range(num_bootstrap):
        # 重采样问题，允许重复
        sampled_questions = random.choices(questions, k=total_questions)
        sampled_correct = 0
        for q in sampled_questions:
            if q in student_answers and student_answers[q] == standard_answers[q]:
                sampled_correct += 1
        sampled_rate = sampled_correct / total_questions
        correct_rates.append(sampled_rate)

    # 计算方差
    variance = sum((rate - correct_rate) ** 2 for rate in correct_rates) / (
        num_bootstrap - 1
    )
    return variance, correct_rates


# 计算 Bootstrap 方差
variance, bootstrap_rates = bootstrap_variance(standard_answers, student_answers)

print(f"正确率的方差 (Bootstrap 方法): {variance:.6f}")

# 可选：如果需要，可以绘制 Bootstrap 正确率的分布
try:
    import matplotlib.pyplot as plt

    plt.hist(bootstrap_rates, bins=30, edgecolor="k", alpha=0.7)
    plt.title("Bootstrap 正确率分布")
    plt.xlabel("正确率")
    plt.ylabel("频数")
    plt.show()
except ImportError:
    print("matplotlib 未安装，无法绘制分布图。")
