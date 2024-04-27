import os

import laspy
import numpy as np
def sparsify_majordata(infile, outfile, scale):

    las = laspy.file.File(infile, mode='r')
    header = las.header  # 创建LAS文件头
    output = laspy.file.File(outfile, mode='w', header=header)
    num = 10000000
    step = scale
    count = 0
    temp = None

# 循环遍历列表中的每个数组
    # 如果 stacked_array 是 None，则直接设置为当前数组
    while True:
        p = las.points[count:count+num:step]
        if temp is None:
            temp = p
        # 否则，使用 np.vstack 将当前数组堆叠到 stacked_array 上
        else:
            temp = np.hstack((temp, p))
        count += num
        print(count)
        if count+num >= len(las.points):
            break
    output.set_points(temp)
    output.close()
    print(outfile)
def sparsification(infile, outfile, scale):

    if os.path.exists(outfile):
        return
    las = laspy.file.File(infile, mode='r')
    header = las.header  # 创建LAS文件头
    output = laspy.file.File(outfile, mode='w', header=header)
    output.points = las.points[::scale]
    output.close()
    print(outfile)

if __name__ == '__main__':
    '''
    sparsification(r"H:\data\original_las\Hongle330dawaji.las", r"H:\data\sparsed_rotated_las\Hongle330dawaji.las", 1)
    sparsification(r"H:\data\original_las\Hongle330diaoche.las", r"H:\data\sparsed_rotated_las\Hongle330diaoche.las", 10)
    sparsification(r"H:\data\original_las\Hongle330G29.las", r"H:\data\sparsed_rotated_las\Hongle330G29.las", 100)
    sparsification(r"H:\data\original_las\Hongle330G52jikeng.las", r"H:\data\sparsed_rotated_las\Hongle330G52jikeng.las", 100)
    sparsification(r"H:\data\original_las\Hongle330GA46.las", r"H:\data\sparsed_rotated_las\Hongle330GA46.las", 100)
    sparsification(r"H:\data\original_las\Hongle330GA49_1.las", r"H:\data\sparsed_rotated_las\Hongle330GA49_1.las", 100)
    sparsification(r"H:\data\original_las\Hongle330GA49_2.las", r"H:\data\sparsed_rotated_las\Hongle330GA49_2.las", 100)
    sparsification(r"H:\data\original_las\Hongle330GA49_3.las", r"H:\data\sparsed_rotated_las\Hongle330GA49_3.las", 100)
    sparsification(r"H:\data\original_las\Hongle330GA50.las", r"H:\data\sparsed_rotated_las\Hongle330GA50.las", 100)
    sparsification(r"H:\data\original_las\Hongle330suodao.las", r"H:\data\sparsed_rotated_las\Hongle330suodao.las", 10)
    sparsification(r"H:\data\original_las\Hongle330xiaowaji.las", r"H:\data\sparsed_rotated_las\Hongle330xiaowaji.las", 1)
    sparsification(r"H:\data\original_las\Qingyang750GA149_1.las", r"H:\data\sparsed_rotated_las\Qingyang750GA149_1.las", 100)
    sparsification(r"H:\data\original_las\Qingyang750GA149_2.las", r"H:\data\sparsed_rotated_las\Qingyang750GA149_2.las", 100)
    sparsification(r"H:\data\original_las\Qingyang750GB163.las", r"H:\data\sparsed_rotated_las\Qingyang750GB163.las", 100)
    sparsification(r"H:\data\original_las\Jinta750biandianzhan.las", r"H:\data\sparsed_rotated_las\Jinta750biandianzhan.las", 100)
    '''
    sparsify_majordata(r"H:\data\original_las\Lanlin750biandianzhan.las", r"H:\data\sparsed_rotated_las\Lanlin750biandianzhan.las", 100)