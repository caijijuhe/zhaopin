import os
import json

# 设置合并完成后的json文件名
output_file = 'admin_utf_8.json'

# 获取所有需要合并的json文件的路径的文件名
input_files = [ os.path.join('/Users/jasonleung/PycharmProjects/recruitwebsite_ner_project/dataset', f)
                for f in os.listdir('/Users/jasonleung/PycharmProjects/recruitwebsite_ner_project/dataset')
                if f.endswith('.json')]

# 读取所有的输入文件并合并
outputs = []
for file_path in input_files:
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                outputs.append(obj)
            except json.JSONDecodeError:
                pass


# 讲合并的json数据保存写入
with open(output_file, 'w', encoding='utf-8') as out_file:
    json.dump(outputs, out_file, ensure_ascii=True, indent = 4)


