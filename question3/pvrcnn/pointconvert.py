import numpy as np
import os
#pc = np.fromfile(str('/root/OpenPCDet-master/data/POINTCLOUD/Mesh08.bin'),dtype=np.float32,count=-1)

#print(pc)

def load_pc_kitti(file):
    scan = np.fromfile(file_dir+'/'+file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, :]  # get xyz

    f = open(file_dir+'/'+ file ,'w')
    for i in range(points.shape[0]):
        for j in range(4):
            strNum = str(points[i][j])
            f.write(strNum)
            f.write(' ')
        f.write('\n')
    f.close()

    print(points)
    return points

file_dir = '/root/OpenPCDet-master/data/POINTCLOUD'
files = os.listdir(file_dir)

for file in files:
    load_pc_kitti(file)



