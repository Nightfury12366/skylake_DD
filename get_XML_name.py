import os

# os.walk方法获取当前路径下的root（所有路径）、dirs（所有子文件夹）、files（所有文件）

path = r"/home/skylake/skylake_files/pic_sky_project/anns_val"

fileout = open('XML_val_name.txt', 'wt')


def file_name():
    F = []
    count = 0
    for root, dirs, files in os.walk(path):
        # print root
        # print dirs
        for file in files:
            count += 1
            # print file.decode('gbk')    #文件名中有中文字符时转码
            if os.path.splitext(file)[1] == '.xml':
                t = os.path.splitext(file)[0]
                print(t)  # 打印所有xml格式的文件名
                fileout.write(t)
                fileout.write('\n')
                F.append(t)  # 将所有的文件名添加到L列表中
    fileout.close()
    print(count)
    return F  # 返回L列表


file_name()
