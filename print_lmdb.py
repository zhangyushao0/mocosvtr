from mmengine import load


lmdb_file = "data/mjsynth_lmdb"
data = load(lmdb_file)


if 'data_list' in data:
    for item in data['data_list']:
        img_path = item.get('img_path', '')
        instances = item.get('instances', [])
        print("")