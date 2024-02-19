import os
import random
from cubercnn import util,vis,data
import numpy as np
    
def load_gt(dataset='SUNRGBD'):
    dataset_paths_to_json = ['/work3/s194235/3dod/datasets/Omni3D/'+dataset+'_test.json']

    # Get Image and annotations
    try:
        dataset = data.Omni3D(dataset_paths_to_json)
    except:
        print('Dataset does not exist or is not in the correct format!')
        exit()
    imgIds = dataset.getImgIds()
    imgs = dataset.loadImgs(imgIds)
    img = random.choice(imgs)
    annIds = dataset.getAnnIds(imgIds=img['id'])
    anns = dataset.loadAnns(annIds)

    # Extract necessary annotations
    R_cams = []
    center_cams = []
    dimensions_all = []
    cats = []
    for instance in range(len(anns)):
        R_cams.append(anns[instance]['R_cam'])
        center_cams.append(anns[instance]['center_cam'])
        dimensions_all.append(anns[instance]['dimensions'])
        cats.append(anns[instance]['category_name'])
    
    return img, R_cams, center_cams, dimensions_all, cats
    


def plot_scene(image_path, output_dir, center_cams, dimensions_all, Rs, K, cats):
    # Compute meshes
    meshes = []
    meshes_text = []
    for idx, (center_cam, dimensions, pose, cat) in enumerate(zip(
            center_cams, dimensions_all, Rs, cats
        )):
        bbox3D = center_cam + dimensions
        meshes_text.append('{}'.format(cat))
        color = [c/255.0 for c in util.get_color(idx)]
        box_mesh = util.mesh_cuboid(bbox3D, pose, color=color)
        meshes.append(box_mesh)
    
    image_name = util.file_parts(image_path)[1]
    print('File: {} with {} dets'.format(image_name, len(meshes)))

    # Plot
    image = util.imread('datasets/'+image_path)
    if len(meshes) > 0:
        im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(image, np.array(K), meshes, text=meshes_text, scale=image.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)

        if False:
            im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
            vis.imshow(im_concat)

        util.imwrite(im_drawn_rgb, os.path.join(output_dir, image_name+'_boxes.jpg'))
        util.imwrite(im_topdown, os.path.join(output_dir, image_name+'_novel.jpg'))
    else:
        print('No meshes')
        util.imwrite(image, os.path.join(output_dir, image_name+'_boxes.jpg'))



def show_data(dataset):
    # Load Image and Ground Truths
    image, Rs, center_cams, dimensions_all, cats = load_gt(dataset)

    # Create Output Directory
    output_dir = 'output/playground/' + dataset
    util.mkdir_if_missing(output_dir)
    
    plot_scene(image['file_path'], output_dir, center_cams, dimensions_all, Rs, image['K'], cats)



show_data('SUNRGBD')    #{SUNRGBD,ARKitScenes,KITTI,nuScenes,Objectron,Hypersim}