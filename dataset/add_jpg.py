import json
f = open('D:\PyTorch\RTDETR/runs/val\exp1_baseline\predictions.json', 'r')
content = f.read()
a = json.loads(content)
index = 0
img_name_dict = []
image_name = []
for i in a:
     if i["image_id"] not in image_name:
         image_name.append(i["image_id"])

for i in a:
    for n in image_name:
        if i["image_id"] == n :
            i["image_id"] = image_name.index(n)


f.close()

b = json.dumps(a)
f2 = open('new_json.json', 'w')
f2.write(b)
f2.close()

