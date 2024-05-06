from VisualiseGT import search_file_backwards, category_distribution
import pandas as pd
import matplotlib.pyplot as plt

def AP3D_per_class(dataset, file='output/base_cubercnn.log'):
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

    # cats = category_distribution(dataset)
    # df.sort_values(by='category', inplace=True)
    # cats = dict(sorted(cats.items()))
    # merged_df = pd.merge(df.reset_index(), pd.DataFrame(cats.values(), columns=['cats']), left_index=True, right_index=True)
    df = df.sort_values(by='AP3D', ascending=False)
    df = df.reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(df['category'], df['AP3D'], color='blue', alpha=0.7, label=f'Cube R-CNN, AP3D: {df["AP3D"].mean():.1f}')
    # ax.bar(df['category'], df['AP3D'], color='blue', alpha=0.7, label='our method 1')
    # 
    # rotate x labels
    plt.xticks(rotation=45)
    ax.set_xlabel('Category')
    ax.set_ylabel('AP3D')
    ax.set_title('AP3D per category')
    plt.savefig('output/figures/cubercnn_acc_classes.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    AP3D_per_class('SUNRGBD', file='output/base_cubercnn.log')