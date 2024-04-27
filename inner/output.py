import os
import open3d as o3d
import numpy as np


def sem_label(source, target, num=18):
    data = np.loadtxt(source)[::100, :]
    lxyz = []
    for i in range(num - 1):
        condition = data[:, 3] == i
        # print(condition)
        select = data[condition]
        if select.size > 0:
            print(i)
            # np.save(os.path.join(savepath, str(i) + ".npy"), select)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(select[:, :3])
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                nums = pcd.cluster_dbscan(eps=3, min_points=20, print_progress=False)
            maxnums = max(nums)
            # print(nums)
            for j in range(maxnums + 1):
                print(*(np.where(np.array(nums) == j)))
                pcd_single = pcd.select_by_index(*(np.where(np.array(nums) == j)))
                #print(pcd_single)
                #o3d.visualization.draw_geometries([pcd_single])
                min_bound = pcd_single.get_min_bound()
                max_bound = pcd_single.get_max_bound()
                lxyz.append(np.hstack((i,min_bound,max_bound)))
    print(lxyz)
    with open(target, 'w')as f:
        for _, l in enumerate(lxyz):
            f.writelines(' '.join(map(str, l))+"\n")



if __name__ == '__main__':
    sor = r'H:\data\output\txt'
    des = r'H:\data\output\box'
    for file in os.listdir(sor):
        inpath = os.path.join(sor, file)
        outpath = os.path.join(des, file.split('.')[0]+'.txt')
        print(file)
        sem_label(inpath, outpath)
    # source = r'C:\Users\yxy\Desktop\train_data\npy\test'
    # target = r'C:\Users\yxy\Desktop\train_data\output'
    # sem_label(source, target, num=18)
