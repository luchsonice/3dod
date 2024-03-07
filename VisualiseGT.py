import os
import random
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cubercnn import data, util, vis
from cubercnn.config import get_cfg_defaults
from cubercnn.data.build import (build_detection_test_loader,
                                 build_detection_train_loader)
from cubercnn.data.dataset_mapper import DatasetMapper3D
from cubercnn.data.datasets import load_omni3d_json, simple_register
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.utils.logger import setup_logger


def load_gt(dataset='SUNRGBD', mode='test', single_im=True):

    # we can do this block of code to get the categories reduced number of categories in the sunrgbd dataset as there normally is 83 categories, however we only work with 38.
    config_file = 'configs/Base_Omni3D.yaml'
    cfg, filter_settings = get_config_and_filter_settings(config_file)

    if mode == 'test':
        dataset_paths_to_json = ['datasets/Omni3D/'+dataset+'_test.json']
    elif mode == 'train':
        dataset_paths_to_json = ['datasets/Omni3D/'+dataset+'_train.json']

    # Get Image and annotations
    try:
        dataset = data.Omni3D(dataset_paths_to_json, filter_settings=filter_settings)
    except:
        print('Dataset does not exist or is not in the correct format!')
        exit()
    imgIds = dataset.getImgIds()
    imgs = dataset.loadImgs(imgIds)
    if single_im:
        img = random.choice(imgs)
        # img = imgs[1]
        annIds = dataset.getAnnIds(imgIds=img['id'])
    else:
        # get all annotations
        img = imgs
        annIds = dataset.getAnnIds()

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
    


def plot_scene(image_path, output_dir, center_cams, dimensions_all, Rs, K, cats, filter_invalid):
    # TODO: currently this function does not filter out invalid annotations, but it should have the option to do so.
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



def show_data(dataset, filter_invalid=False):
    # Load Image and Ground Truths
    image, Rs, center_cams, dimensions_all, cats = load_gt(dataset)

    # Create Output Directory
    output_dir = 'output/playground/' + dataset
    util.mkdir_if_missing(output_dir)
    
    plot_scene(image['file_path'], output_dir, center_cams, dimensions_all, Rs, image['K'], cats, filter_invalid)


def category_distribution(dataset):
    '''Plot a histogram of the category distribution in the dataset.'''
    # Load Image and Ground Truths
    image, Rs, center_cams, dimensions_all, cats = load_gt(dataset, mode='train', single_im=False)
    image_t, Rs_t, center_cams_t, dimensions_all_t, cats_t = load_gt(dataset, mode='test', single_im=False)

    output_dir = 'output/figures/' + dataset
    util.mkdir_if_missing(output_dir)

    # histogram of categories
    cats_all = cats + cats_t
    cats_unique = list(set(cats_all))
    print('cats unique: ', len(cats_unique))
    # make dict with count of each category
    cats_count = {cat: cats_all.count(cat) for cat in cats_unique}
    cats_sorted = dict(sorted(cats_count.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(14,5))
    plt.bar(cats_sorted.keys(), cats_sorted.values())
    plt.xticks(rotation=60, size=9)

    plt.title('Category Distribution')
    plt.savefig(os.path.join(output_dir, 'category_distribution.png'),dpi=300, bbox_inches='tight')
    plt.close()

    return cats_sorted

def spatial_statistics(dataset):
    '''Compute spatial statistics of the dataset.
    wanted to reproduce fig. 7 from the omni3D paper
    however, we must standardise the images for it to work
    '''
    # Load Image and Ground 
    # this function filters out invalid images if there are no valid annotations in the image
    # annnotations in each image can also be marked as is_ignore => True
    image_root = 'datasets'
    cfg, filter_settings = get_config_and_filter_settings()
    dataset_names = ['SUNRGBD_train','SUNRGBD_test','SUNRGBD_val']
    output_dir = 'output/figures/' + dataset

    # this is almost the same as the simple_register function, but it also stores the model metadata
    # which is needed for the load_omni3d_json function 
    data.register_and_store_model_metadata(None, output_dir, filter_settings=filter_settings)

    data_dicts = []
    for dataset_name in dataset_names:
        json_file = 'datasets/Omni3D/'+dataset_name+'.json'
        data_dict = load_omni3d_json(json_file, image_root, dataset_name, filter_settings, filter_empty=True)
        data_dicts.extend(data_dict)
    

    # standardise the images to a fixed size
    # and map the annotations to the standardised images
    std_image_size = (480//4, 640//4)
    tot_outliers = 0
    img = np.zeros(std_image_size)
    for img_dict in data_dicts:
        original_width = img_dict['width']
        original_height = img_dict['height']
        
        # Calculate the scale factor for resizing
        scale_x = std_image_size[1] / original_width
        scale_y = std_image_size[0] / original_height

        # Update the image size in the annotation
        img_dict['width'] = std_image_size[1]
        img_dict['height'] = std_image_size[0]
        for anno in img_dict['annotations']:
            if not anno['ignore']:
                # Update the 2D box coordinates (boxes are XYWH)
                anno['bbox2D_tight'][0] *= scale_x
                anno['bbox2D_tight'][1] *= scale_y
                anno['bbox2D_tight'][2] *= scale_x
                anno['bbox2D_tight'][3] *= scale_y
                # get the centerpoint of the annotation as (x, y)
                x0, y0, x1, y1 = BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                x_m, y_m = int((x0+x1)/2), int((y0+y1)/2)
                if x_m >= std_image_size[1] or x_m < 0:
                    # print(f'x out of line {x_m}')
                    tot_outliers += 1
                elif y_m >= std_image_size[0] or y_m < 0:
                    # print(f'y out of line {y_m}')
                    tot_outliers += 1
                else:
                    img[y_m, x_m] += 1
            else:
                # Remove the annotation if it is marked as ignore
                img_dict['annotations'].remove(anno)


    print('num center points outside frame: ', tot_outliers)
    img = img/img.max()
    # this point is so large that all the points become invisible, so I remove it.
    img[0,0] = 0.00 
    img = img/img.max()
    plt.figure()
    plt.imshow(img, cmap='gray_r', vmin=0, vmax=1)
    plt.xticks([]); plt.yticks([])
    plt.title('Histogram of 2D box centre points')
    plt.savefig(os.path.join(output_dir, '2d_histogram.png'),dpi=300, bbox_inches='tight')
    plt.close()
    return

def AP_vs_no_of_classes(dataset, file='output/Baseline_sgd/log.txt'):
    '''Search the log file for the precision numbers corresponding to the last iteration
    then parse it in as a pd.DataFrame and plot the AP vs number of classes'''

    # search the file from the back until the line 
    # cubercnn.vis.logperf INFO: Performance for each of 38 categories on SUNRGBD_test:
    # is found

    target_line = "cubercnn.vis.logperf INFO: Performance for each of 38 categories on SUNRGBD_test:"
    df = search_file_backwards(file, target_line)
    if df is None:
        print('df not found')
        return

    cats = category_distribution(dataset)
    df.sort_values(by='category', inplace=True)
    cats = dict(sorted(cats.items()))
    merged_df = pd.merge(df.reset_index(), pd.DataFrame(cats.values(), columns=['cats']), left_index=True, right_index=True)
    merged_df = merged_df.sort_values(by='cats')
    merged_df = merged_df.drop('index',axis=1)
    merged_df = merged_df.reset_index(drop=True)
    
    
    fig, ax = plt.subplots(figsize=(12,8))
    scatter = ax.scatter(merged_df['cats'].values, merged_df['AP3D'].values, s=merged_df['AP2D'].values*2, alpha=0.5, label='AP3D (scaled by AP2D)')
    for i, txt in enumerate(merged_df['category']):
        ax.text(merged_df['cats'].values[i], merged_df['AP3D'].values[i], txt)
    
    correlation_coef = np.corrcoef(merged_df['cats'].values, merged_df['AP3D'].values)[0, 1]
    line_fit = np.polyfit(merged_df['cats'].values, merged_df['AP3D'].values, 1)

    # plot the line of best fit
    ax.plot(merged_df['cats'].values, np.poly1d(line_fit)(merged_df['cats'].values), linestyle='--', color='black',alpha=0.5, label=f'Linear fit (R={correlation_coef:.2f})')

    # Set labels and title
    ax.set_xlabel('No. of annotations')
    ax.set_ylabel('AP3D')
    ax.set_xscale('log')
    ax.set_title('AP3D vs No. of annotations')
    ax.legend()

    # Save the plot
    plt.savefig('output/figures/'+dataset+'/AP_vs_no_of_classes.png', dpi=300, bbox_inches='tight')
    plt.close()

    return

def AP3D_vs_AP2D(dataset, file='output/Baseline_sgd/log.txt'):
    '''Search the log file for the precision numbers corresponding to the last iteration
    then parse it in as a pd.DataFrame and plot the AP vs number of classes'''

    # search the file from the back until the line 
    # cubercnn.vis.logperf INFO: Performance for each of 38 categories on SUNRGBD_test:
    # is found

    target_line = "cubercnn.vis.logperf INFO: Performance for each of 38 categories on SUNRGBD_test:"
    df = search_file_backwards(file, target_line)
    if df is None:
        print('df not found')
        return

    cats = category_distribution(dataset)
    df.sort_values(by='category', inplace=True)
    cats = dict(sorted(cats.items()))
    merged_df = pd.merge(df.reset_index(), pd.DataFrame(cats.values(), columns=['cats']), left_index=True, right_index=True)
    merged_df = merged_df.sort_values(by='cats')
    merged_df = merged_df.drop('index',axis=1)
    merged_df = merged_df.reset_index(drop=True)
    print(merged_df)
    
    fig, ax = plt.subplots(figsize=(12,8))
    scatter = ax.scatter(merged_df['AP2D'].values, merged_df['AP3D'].values, alpha=0.5, label='')
    for i, txt in enumerate(merged_df['category']):
        ax.text(merged_df['AP2D'].values[i], merged_df['AP3D'].values[i], txt)
    # plot average line
    ax.plot((0, 70), (0, 70), linestyle='--', color='black', alpha=0.3, label=f'AP2D=AP3D')

    # Set labels and title
    ax.set_xlabel('AP2D')
    ax.set_ylabel('AP3D')
    ax.set_xlim(0, 75); ax.set_ylim(0, 75)
    ax.set_title('AP3D vs AP in 2D annotations')
    ax.legend()

    # Save the plot
    plt.savefig('output/figures/'+dataset+'/AP3D_vs_AP2D.png', dpi=300, bbox_inches='tight')
    plt.close()

    return


def search_file_backwards(file_path:str, target_line:str) -> pd.DataFrame:
    '''Search a file backwards for a target line and return the table of the performance of the model. The point of this is to parse the part of the log file that looks like this
    |  category  | AP2D    | AP3D      |  category   | AP2D     | AP3D     |   category   | AP2D      | AP3D       |
    |:----------:|:--------|:----------|:-----------:|:---------|:---------|:------------:|:----------|:-----------|
    |   chair    | 45.9374 | 53.4913   |    table    | 34.5982  | 39.7769  |   cabinet    | 16.3693   | 14.0878    |
    |    lamp    | 24.8081 | 7.67653   |    books    | 0.928978 | 0.599711 |     sofa     | 49.2354   | 57.9649    |
    
    ...
    To a pandas DataFrame that has 3 columns: category, AP2D, AP3D'''
    import re
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(reversed(lines)):
            is_found = re.search(f'.*{target_line}$', line)
            if is_found:
                table = lines[-i:-i+15]
                tab_as_str= ' '.join(table)
                # i know this is really ugly
                df = pd.read_csv( StringIO(tab_as_str.replace(' ', '')),  # Get rid of whitespaces
                    sep='|',).dropna(axis=1, how='all').drop(0)
                # https://stackoverflow.com/a/65884212
                df.columns = pd.MultiIndex.from_frame(df.columns.str.split('.', expand=True)
                                        .to_frame().fillna('0'))
                df = df.stack().reset_index(level=1, drop=True).reset_index().drop('index', axis=1)               
                df['AP3D'] = df['AP3D'].astype(float)
                df['AP2D'] = df['AP2D'].astype(float)

                return df
                
    return None


def get_config_and_filter_settings(config_file='configs/Base_Omni3D.yaml'):
    # we must load the config file to get the filter settings
    cfg = get_cfg()
    get_cfg_defaults(cfg)
    cfg.merge_from_file(config_file)
    # must setup logger to get info about filtered out annotations
    setup_logger(output=cfg.OUTPUT_DIR, name="cubercnn")
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    return cfg, filter_settings


def init_dataloader():
    ''' dataloader stuff.
    currently not used anywhere, because I'm not sure what the difference between the omni3d dataset and load omni3D json functions are. this is a 3rd alternative to this. The train script calls something similar to this.'''
    cfg, filter_settings = get_config_and_filter_settings()

    dataset_names = ['SUNRGBD_train','SUNRGBD_val']
    dataset_paths_to_json = ['datasets/Omni3D/'+dataset_name+'.json' for dataset_name in dataset_names]
    for dataset_name in dataset_names:
        simple_register(dataset_name, filter_settings, filter_empty=True)

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

    from detectron2 import data as d2data
    NoOPaug = d2data.transforms.NoOpTransform()

    # def NoOPaug(input):
        # return input
    # TODO: how to load in images without having them resized?
    # data_mapper = DatasetMapper3D(cfg, augmentations=[NoOPaug], is_train=True)
    data_mapper = DatasetMapper3D(cfg, is_train=True)
    # test loader does resize images, like the train loader does
    # this is the function that filters out the invalid annotations
    data_loader = build_detection_train_loader(cfg, mapper=data_mapper, dataset_id_to_src=dataset_id_to_src, num_workers=1)
    # data_loader = build_detection_test_loader(cfg, dataset_names[1], num_workers=1)

    # this is a detectron 2 thing that we just have to do
    data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats


    for item in data_loader:
        print(item)

if __name__ == '__main__':
    # show_data('SUNRGBD')  #{SUNRGBD,ARKitScenes,KITTI,nuScenes,Objectron,Hypersim}
    # _ = category_distribution('SUNRGBD')
    # AP_vs_no_of_classes('SUNRGBD')
    # spatial_statistics('SUNRGBD')
    AP3D_vs_AP2D('SUNRGBD')
    # init_dataloader()