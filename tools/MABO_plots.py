# Copyright (c) Meta Platforms, Inc. and affiliates
import warnings
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_512 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_384 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_224 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_11m_224 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_5m_224 in registry")

import os
import numpy as np
import torch
import pickle

from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def create_striped_patch(ax, x_start, x_end, color, alpha=0.3):
    ylim = ax.get_ylim()
    stripe_height = (ylim[1] - ylim[0]) / 20  # Adjust stripe height as needed
    vertices = []
    codes = []
    for y in np.arange(ylim[0], ylim[1], stripe_height * 2):
        vertices.extend([(x_start, y), (x_end, y + stripe_height), (x_end, y + stripe_height * 2), (x_start, y + stripe_height)])
        codes.extend([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
    path = Path(vertices, codes)
    patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha, hatch='/')
    ax.add_patch(patch)

color_palette = ['#008dff','#ff73bf','#c701ff','#4ecb8d','#ff9d3a','#f0c571','#384860','#d83034']

proposal_function = 'aspect'
print('loading ...')
with open('output/pkl_files/outputs_'+str(proposal_function)+'.pkl', 'rb') as file:
    outputs = pickle.load(file)
print('... done')

# Create output folder
if not os.path.exists('ProposalNetwork/output/MABO_'+str(proposal_function)):
    os.makedirs('ProposalNetwork/output/MABO_'+str(proposal_function)) # This is maybe unnecessary
    os.makedirs('ProposalNetwork/output/MABO_'+str(proposal_function)+'/vis/')

# mean over all the outputs
Iou2D             = np.concatenate([np.array(sublist) for sublist in (x[1] for x in outputs)])
score_seg         = np.concatenate([np.array(sublist) for sublist in (x[2] for x in outputs)])
score_dim         = np.concatenate([np.array(sublist) for sublist in (x[3] for x in outputs)])
score_combined    = np.concatenate([np.array(sublist) for sublist in (x[4] for x in outputs)])
score_random      = np.concatenate([np.array(sublist) for sublist in (x[5] for x in outputs)])
score_point_cloud = np.concatenate([np.array(sublist) for sublist in (x[6] for x in outputs)])
score_seg_mod     = np.concatenate([np.array(sublist) for sublist in (x[10] for x in outputs)])
score_corners     = np.concatenate([np.array(sublist) for sublist in (x[11] for x in outputs)])
stat_empty_boxes  = np.array([x[7] for x in outputs])
combinations      = np.mean(np.concatenate([np.array(sublist) for sublist in (x[12] for x in outputs)]),axis=0)
#logger.info('Percentage of cubes with no intersection:',np.mean(stat_empty_boxes))
print('Percentage of cubes with no intersection:',np.mean(stat_empty_boxes))
print('combination scores:',combinations)
print('best combination is C'+str(np.argmax(combinations)+1))

Iou2D = Iou2D.mean(axis=0)
score_seg = score_seg.mean(axis=0)
score_dim = score_dim.mean(axis=0)
score_combined = score_combined.mean(axis=0)
score_random = score_random.mean(axis=0)
score_point_cloud = score_point_cloud.mean(axis=0)
score_seg_mod = score_seg_mod.mean(axis=0)
score_corners = score_corners.mean(axis=0)
total_num_instances = np.sum([x[0].gt_boxes3D.shape[0] for x in outputs])

print('Avg IoU of chosen cube:', score_combined[0])
print('Best possible IoU:', score_combined[-1])

x_range = np.arange(1,1001)
plt.figure(figsize=(8,5))
plt.plot(x_range,score_combined, linestyle='-',c=color_palette[6], label='combined') 
plt.plot(x_range,score_dim, linestyle='-',c=color_palette[5],label='dim') 
plt.plot(x_range,score_seg, linestyle='-',c=color_palette[2],label='segment')
plt.plot(x_range,Iou2D, linestyle='-',c=color_palette[4],label='2D IoU') 
plt.plot(x_range,score_corners, linestyle='-',c=color_palette[7],label='corner dist')
plt.plot(x_range,score_random, linestyle='-',c='grey',label='random') 
plt.plot(x_range,score_point_cloud, linestyle='-',c=color_palette[3],label='point cloud')
plt.plot(x_range,score_seg_mod, linestyle='-',c=color_palette[0],label='mod segment')
plt.grid(True)
plt.xscale('log')
plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
plt.xlim(left=1, right=len(Iou2D))
plt.xlabel('Number of Proposals')
plt.ylabel('3D IoU')
plt.legend()
plt.title('Average Best Overlap vs Number of Proposals ({} images, {} instances)'.format(4815,total_num_instances))
f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'MABO_'+str(proposal_function)+'.png')
plt.savefig(f_name, dpi=300, bbox_inches='tight')
plt.close()
print('saved to ', f_name)

# Statistics
stats = torch.cat([x[8] for x in outputs],dim=0)
print('Percentage inside searched area:', ((stats >= 0) & (stats <= 1)).float().mean(dim=0) * 100)
num_bins = 40
titles = ['x','y','z']
plt.figure(figsize=(15, 5))
plt.suptitle("Distribution of Ground Truths in Normalised Searched Range", fontsize=20)
for i,title in enumerate(titles):
    ax = plt.subplot(1, 3, 1+i)
    plt.hist(stats[:,i].numpy(), bins=num_bins, color=color_palette[6],density=True, zorder=2)
    plt.axvline(x=0, color='#97a6c4',zorder=2)
    plt.axvline(x=1, color='#97a6c4',zorder=2)
    create_striped_patch(ax, 0, 1, '#97a6c4', alpha=0.8)
    plt.title(title)
f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_center_'+str(proposal_function)+'.png')
plt.savefig(f_name, dpi=300, bbox_inches='tight')
plt.close()
print('saved to ', f_name)

if proposal_function == 'xy' or 'z':
    num_bins = [600,300,200]
elif proposal_function == 'dim' or 'rotation' or 'full':
    num_bins = [300,700,70]
else:
    num_bins = [200, 200, 200]
plt.figure(figsize=(15, 5))
plt.suptitle("Distribution of Ground Truths in Normalised Searched Range", fontsize=20)
for i,title in enumerate(titles):
    ax = plt.subplot(1, 3, 1+i)
    plt.hist(stats[:,i].numpy(), bins=num_bins[i], color=color_palette[6],density=True, zorder=2)
    plt.axvline(x=0, color='#97a6c4', zorder=2)
    plt.axvline(x=1, color='#97a6c4', zorder=2)
    create_striped_patch(ax, 0, 1, '#97a6c4', alpha=0.8)
    plt.xlim([max(-2,min(stats[:,i])),min(2,max(stats[:,i]))])
    plt.title(title)
f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_center_zoom_'+str(proposal_function)+'.png')
plt.savefig(f_name, dpi=300, bbox_inches='tight')
plt.close()
print('saved to ', f_name)
num_bins = 40
titles = ['w','h','l']
plt.figure(figsize=(15, 5))
plt.suptitle("Distribution of Ground Truths in Normalised Searched Range", fontsize=20)
for i,title in enumerate(titles):
    ax = plt.subplot(1, 3, 1+i)
    plt.hist(stats[:,3+i].numpy(), bins=num_bins, color=color_palette[6],density=True, zorder=2)
    plt.axvline(x=0, color='#97a6c4', zorder=2)
    plt.axvline(x=1, color='#97a6c4', zorder=2)
    create_striped_patch(ax, 0, 1, '#97a6c4', alpha=0.8)
    plt.title(title)
f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_dim_'+str(proposal_function)+'.png')
plt.savefig(f_name, dpi=300, bbox_inches='tight')
plt.close()
print('saved to ', f_name)
titles = ['rx','ry','rz']
plt.figure(figsize=(15, 5))
plt.suptitle("Distribution of Ground Truths in Normalised Searched Range", fontsize=20)
for i,title in enumerate(titles):
    ax = plt.subplot(1, 3, 1+i)
    plt.hist(stats[:,6+i].numpy(), bins=num_bins, color=color_palette[6],density=True, zorder=2)
    plt.axvline(x=0, color='#97a6c4', zorder=2)
    plt.axvline(x=1, color='#97a6c4', zorder=2)
    create_striped_patch(ax, 0, 1, '#97a6c4', alpha=0.8)
    plt.title(title)
f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_rot_'+str(proposal_function)+'.png')
plt.savefig(f_name, dpi=300, bbox_inches='tight')
plt.close()
print('saved to ', f_name)

titles = ['x','y','z','w','h','l','rx','ry','rz']
stats_off = np.concatenate([np.array(sublist) for sublist in (x[9] for x in outputs)])
plt.figure(figsize=(15, 15))
for i,title in enumerate(titles):
    plt.subplot(3, 3, 1+i)
    plt.scatter(stats_off[:,1+i],stats_off[:,0], color=color_palette[6])
    plt.title(title)
f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_off_'+str(proposal_function)+'.png')
plt.savefig(f_name, dpi=300, bbox_inches='tight')
plt.close()
print('saved to ', f_name)

plt.figure(figsize=(15, 15))
for i,title in enumerate(titles):
    plt.subplot(3, 3, 1+i)
    plt.scatter(stats_off[:,1+i],stats_off[:,0], color=color_palette[6])
    plt.title(title)
    plt.xlim([0,2])
    plt.ylim([0,1])
f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_off_zoom_'+str(proposal_function)+'.png')
plt.savefig(f_name, dpi=300, bbox_inches='tight')
plt.close()
print('saved to ', f_name)