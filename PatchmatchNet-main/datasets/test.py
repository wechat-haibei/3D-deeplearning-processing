from data_io import *
# t = read_image('./data/scan2/images/00000001.jpg',640)
# print(t)
import os
scan_folder = '.\data\scan2'
pair_file = os.path.join(scan_folder, "pair.txt")
pair_data = read_pair_file(pair_file)
# print(pair_data)
    # for each reference view and the corresponding source views
for ref_view, src_views in pair_data:
#     # load the reference image
    ref_img, original_h, original_w = read_image(
        os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)),640)
    # print(read_cam_file(
    #     os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))[0])
    ref_intrinsics, ref_extrinsics,_ = read_cam_file(
        os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
#     ref_intrinsics[0] *= img_wh[0]/original_w
#     ref_intrinsics[1] *= img_wh[1]/original_h