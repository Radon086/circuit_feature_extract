import json

with open('./output/test_proto/meta.jsonl', 'r') as f:
    data = [json.loads(line) for line in f] 

for item in data:
    key = list(item.keys())[0]
    print(key)
    IC = 0
    for connection in item[key]['ic_connection']:
        print(f"Connection {IC}:")
        IC += 1
        for x in connection:
            print(x)
