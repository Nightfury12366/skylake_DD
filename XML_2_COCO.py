# coding=utf-8
import xml.etree.ElementTree as ET
import os
import json


voc_clses = ['柚', '皋', '莲', '枣', '樱', '桐', '椿', '柊', '菖', '八百比丘尼', '药店老板', '瓜之介', '阿菊', '其他']  # 顺序很重要


categories = []
for iind, cat in enumerate(voc_clses):  # 生成索引
    cate = {}
    cate['supercategory'] = cat
    cate['name'] = cat
    cate['id'] = iind + 1
    categories.append(cate)


def getimages(xmlname, id):
    sig_xml_box = []
    tree = ET.parse(xmlname)  # 输入是指定的xml文件，包括路径！
    root = tree.getroot()
    images = {}
    for i in root:  # 遍历一级节点
        if i.tag == 'filename':  # 图片名
            file_name = i.text  # 0001.jpg
            # print('image name: ', file_name)
            images['file_name'] = file_name
        if i.tag == 'size':
            for j in i:
                if j.tag == 'width':  # 图片大小
                    width = j.text
                    images['width'] = width
                if j.tag == 'height':
                    height = j.text
                    images['height'] = height
        if i.tag == 'object':  # 目标
            for j in i:
                if j.tag == 'name':
                    cls_name = j.text
                cat_id = voc_clses.index(cls_name) + 1  # 找到id
                if j.tag == 'bndbox':
                    bbox = []
                    xmin = 0
                    ymin = 0
                    xmax = 0
                    ymax = 0
                    for r in j:
                        if r.tag == 'xmin':
                            xmin = eval(r.text)
                        if r.tag == 'ymin':
                            ymin = eval(r.text)
                        if r.tag == 'xmax':
                            xmax = eval(r.text)
                        if r.tag == 'ymax':
                            ymax = eval(r.text)
                    bbox.append(xmin)
                    bbox.append(ymin)
                    bbox.append(xmax - xmin)
                    bbox.append(ymax - ymin)
                    bbox.append(id)   # 保存当前box对应的image_id
                    bbox.append(cat_id)
                    # anno area
                    bbox.append((xmax - xmin) * (ymax - ymin) - 10.0)   # bbox的ares
                    # coco中的ares数值是 < w*h 的, 因为它其实是按segmentation的面积算的,所以我-10.0一下...
                    sig_xml_box.append(bbox)  # 一张图片里所有的bbox
                    # print('bbox', xmin, ymin, xmax - xmin, ymax - ymin, 'id', id, 'cls_id', cat_id)
    images['id'] = id  # 这张图片的id
    # print ('sig_img_box', sig_xml_box)
    return images, sig_xml_box


def txt2list(txtfile):
    f = open(txtfile)
    l = []
    for line in f:
        l.append(line[:-1])
    print(l)
    return l


# news2020xmls = '/home/skylake/skylake_files/news_2020_train'  # xml标注的路径地址
# # news2020xmls = '/home/skylake/skylake_files/news_2020_val'  # xml标注的路径地址

# news2020xmls = '/home/skylake/skylake_files/pic_sky_project/anns'  # xml标注的路径地址
news2020xmls = '/home/skylake/skylake_files/pic_sky_project/anns_val'  # xml标注的路径地址


train_txt = 'XML_train_name.txt'
val_txt = 'XML_val_name.txt'

# xml_names = txt2list(train_txt)  # 每次切换数据集这里都要改
xml_names = txt2list(val_txt)


xmls = []  # 所有XML文件路径名
bboxes = []
ann_js = {}

'''这一步到时候要改'''
for ind, xml_name in enumerate(xml_names):
    xmls.append(os.path.join(news2020xmls, xml_name + '.xml'))
'''这一步到时候要改'''

# json_name = '/home/skylake/skylake_project/Yet-Another-EfficientDet-Pytorch/datasets/news2020/annotations/instances_train2020.json'  # 一整个训练集，一个json文件
json_name = '/home/skylake/skylake_project/Yet-Another-EfficientDet-Pytorch/datasets/news2020/annotations/instances_val2020.json'  # 一整个训练集，一个json文件
images = []
for i_index, xml_file in enumerate(xmls):
    image, sig_xml_bbox = getimages(xml_file, i_index)
    images.append(image)  # 添加图片
    bboxes.extend(sig_xml_bbox)  # 列表扩张

ann_js['images'] = images
ann_js['categories'] = categories
annotations = []

for box_ind, box in enumerate(bboxes):
    anno = {}
    anno['image_id'] = box[-3]
    anno['category_id'] = box[-2]
    anno['bbox'] = box[:-3]
    anno['id'] = box_ind
    anno['area'] = box[-1]
    anno['iscrowd'] = 0
    annotations.append(anno)
ann_js['annotations'] = annotations
json.dump(ann_js, open(json_name, 'w'), indent=4)  # indent=4 更加美观显示
