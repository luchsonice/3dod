import torch
import cv2
# might need to export PYTHONPATH=/work3/$username/3dod/

def init_dataset():
    ''' dataloader stuff.
     I'm not sure what the difference between the omni3d dataset and load omni3D json functions are. this is a 3rd alternative to this. The train script calls something similar to this.'''
    cfg, filter_settings = get_config_and_filter_settings()

    dataset_names = ['KITTI_train', 'KITTI_val', 'KITTI_test',]
    dataset_paths_to_json = ['datasets/Omni3D/'+dataset_name+'.json' for dataset_name in dataset_names]
    # for dataset_name in dataset_names:
    #     simple_register(dataset_name, filter_settings, filter_empty=True)

    # Get Image and annotations
    datasets = data.Omni3D(dataset_paths_to_json, filter_settings=filter_settings)
    data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)

    thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
    dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id

    infos = datasets.dataset['info']

    dataset_id_to_unknown_cats = {}
    possible_categories = set(i for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1))
    
    dataset_id_to_src = {}

    for info in infos:
        dataset_id = info['id']
        known_category_training_ids = set()
        
        if not dataset_id in dataset_id_to_src:
            dataset_id_to_src[dataset_id] = info['source']

        for id in info['known_category_ids']:
            if id in dataset_id_to_contiguous_id:
                known_category_training_ids.add(dataset_id_to_contiguous_id[id])
        
        # determine and store the unknown categories.
        unknown_categories = possible_categories - known_category_training_ids
        dataset_id_to_unknown_cats[dataset_id] = unknown_categories

    return datasets

if __name__ == '__main__':
    from detectron2.data.catalog import MetadataCatalog
    import numpy as np

    from cubercnn import data
    from priors import get_config_and_filter_settings

    from tqdm import tqdm
    from ProposalNetwork.utils.plane import Plane as Plane_cuda
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    datasets = init_dataset()

    normal_vecs = []
    use_nth = 5
    # this is using a non-standard coordinate system where +x right, +y up, +z into the screen
    for img_id, img_info in tqdm(datasets.imgs.items()):
        file_path = img_info['file_path']
        org_image_size = img_info['width'], img_info['height']

        dp_img = np.load(f'datasets/depth_maps/{img_info['id']}.npz')['depth']
        try:
            ground_img = np.load(f'datasets/ground_maps/{img_info['id']}.npz')['mask']
            ground_map = torch.as_tensor(ground_img, device=device)
        except FileNotFoundError:
            ground_map = None
        # the imagesize by default is (height, width)
        depth_map = torch.as_tensor(dp_img, device=device)
        K = torch.as_tensor(img_info['K'], device=device)
        z = depth_map[::use_nth,::use_nth]
        # i don't know if it makes sense to use the image shape as the 
        # this way it looks much more correct
        # https://github.com/DepthAnything/Depth-Anything-V2/blob/31dc97708961675ce6b3a8d8ffa729170a4aa273/metric_depth/depth_to_pointcloud.py#L100
        width, height = z.shape[1], z.shape[0]
        focal_length_x, focal_length_y = K[0,0] // use_nth, K[1,1] // use_nth

        u, v = torch.meshgrid(torch.arange(width, device=device), torch.arange(height,device=device), indexing='xy')
        cx, cy = width / 2, height / 2 # principal point of camera
        # https://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.create_point_cloud_from_depth_image.html
        x = (u - cx) * z / focal_length_x
        y = (v - cy) * z / focal_length_y
        if ground_map is not None:
            # select only the points in x,y,z that are part of the ground map
            ground = ground_map[::use_nth,::use_nth]
            zg = z[ground > 0]
            xg = x[ground > 0]
            yg = y[ground > 0]
        else:
            # the ground map also works to remove the padded 0's to the depth maps
            # so in the case the ground map is not available we must ensure to only select the valid part of the image
            mask = torch.ones(org_image_size[::-1], device=device)
            image_without_pad = mask[::use_nth,::use_nth]
            zg = z[image_without_pad > 0]
            xg = x[image_without_pad > 0]
            yg = y[image_without_pad > 0]

        # normalise the points
        points = torch.stack((xg, yg, zg), axis=-1)

        plane = Plane_cuda()
        # best_eq is the ground plane as a,b,c,d in the equation ax + by + cz + d = 0
        # if this errors out, run the filter ground script first
        best_eq, best_inliers = plane.fit_parallel(points, thresh=0.05, maxIteration=1000)
        normal_vec = best_eq[:-1]

        x_up = torch.tensor([1.0, 0.0, 0.0], device=device)
        y_up = torch.tensor([0.0, 1.0, 0.0], device=device)
        z_up = torch.tensor([0.0, 0.0, 1.0], device=device)
        # make sure normal vector is consistent with y-up
        if (normal_vec @ z_up).abs() > (normal_vec @ y_up).abs():
            # this means the plane has been found as the back wall
            # to rectify this we can turn the vector 90 degrees around the local x-axis
            # note that this assumes that the walls are perpendicular to the floor
            normal_vec = normal_vec[torch.tensor([0,2,1], device=device)] * torch.tensor([1, 1, -1], device=device)
        if (normal_vec @ x_up).abs() > (normal_vec @ y_up).abs():
            # this means the plane has been found as the side wall
            # to rectify this we can turn the vector 90 degrees around the local y-axis
            # note that this assumes that the walls are perpendicular to the floor
            normal_vec = normal_vec[torch.tensor([2,0,1], device=device)] * torch.tensor([-1, 1, 1], device=device)
        if normal_vec @ y_up < 0:
            normal_vec *= -1

        normal_vecs.append(normal_vec)
        print(img_id)

    nrml_vecs = torch.stack(normal_vecs)
    torch.save(nrml_vecs, 'datasets/normal_vectors.pth')