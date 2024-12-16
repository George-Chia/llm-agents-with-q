import json
'''
def jsonl_to_json(jsonl_file, json_file):
    with open(jsonl_file, 'r') as infile:
        with open(json_file, 'w') as outfile:
            json_lines = infile.readlines()
            json_list = [json.loads(line) for line in json_lines]
            json.dump(json_list, outfile, indent=4)
'''

def jsonl_to_json(input_file, output_file):
 with open(input_file, 'r', encoding='utf-8') as infile, \
      open(output_file, 'w', encoding='utf-8') as outfile:
     json_list = []
     for line in infile:
         line = line.strip()
         if not line or line.startswith('#'):  # 跳过空行和注释
             continue
         try:
             json_list.append(json.loads(line))
         except json.JSONDecodeError as e:
             print(f"Error parsing line: {line} - {e}")
     json.dump(json_list, outfile, indent=4, ensure_ascii=False)


# 用法示例
jsonl_to_json('ToG_webqsp.jsonl', '../eval/ToG_webqsp.json')