import json
f = open('E:\VisDrone\VisDrone2019-DET-val\coco_annotations/val.json', 'r')
content = f.read()
a = json.loads(content)
images = a["images"]
name_id_dict = {}
for i in images:
    name_id_dict[i["id"]] = i["file_name"]

for i in a["annotations"]:
    for key, value in name_id_dict.items():
        if i["image_id"] == key:
            i["image_id"] = value[0:-4]

f.close()

b = json.dumps(a)
f2 = open('new_val.json', 'w')
f2.write(b)
f2.close()
