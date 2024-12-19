student = [
    {'id': '1001', 'name': '无语', 'english': 98, 'python': 100, 'c': 96},
    {'id': '1002', 'name': '琦琦', 'english': 100, 'python': 96, 'c': 97},
    {'id': '1003', 'name': '明日', 'english': 99, 'python': 97, 'c': 95},
    {'id': '1004', 'name': '田珍', 'english': 93, 'python': 99, 'c': 98}
    ]   # 保存学生成绩的列表
student.sort(key=lambda x :x['english']+x['python']+x['c'],reverse = True)# 按总成绩排序
for item in student:  # 遍历输出排序结果
    print(item)
