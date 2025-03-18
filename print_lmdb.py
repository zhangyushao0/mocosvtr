import lmdb
from PIL import Image
import io
import os

# 指定 LMDB 文件路径
lmdb_path = r"G:\download\Union14M-U\cc_lmdb"

# 打开 LMDB 环境
env = lmdb.open(lmdb_path, readonly=True, lock=False)

try:
    # 开始一个读事务
    with env.begin() as txn:
        cursor = txn.cursor()
        count = 0  # 用于限制保存前十个图像

        # 遍历键值对
        for key, value in cursor:
            if count >= 10:  # 只处理前十个
                break

            # 打印键（假设键是 UTF-8 编码的字符串）
            key_str = key.decode("utf-8")
            print(f"Key: {key_str}")

            # 将字节数据转换为图像
            image = Image.open(io.BytesIO(value))

            # 保存图像到当前目录，文件名基于键
            image_filename = f"{key_str}.png"  # 假设保存为 PNG 格式
            image.save(image_filename)
            print(f"Saved image as: {image_filename}")

            count += 1

except lmdb.Error as e:
    print(f"LMDB error: {e}")
except Exception as e:
    print(f"Error processing image: {e}")
finally:
    env.close()
