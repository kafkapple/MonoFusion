


from utils import get_preset_data, interpolate_extrinsics

pose_matrices, intrinsics_matrices = get_preset_data(size, seq='_dance')

interpolated_pose_matrices = []
for i in range(len(pose_matrices) - 1):
    ex1 = pose_matrices[i]
    ex2 = pose_matrices[i + 1]
    interpolated_pose = interpolate_extrinsics(ex1, ex2)
    interpolated_pose_matrices.append(interpolated_pose)




#### 
# rd every training frame (covers drop every seq)
####
    

####
# rd every interpolated frame 
####
    

####
# rd every novel temporal frame 
####