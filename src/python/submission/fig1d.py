import pandas
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat


tmaze_keymap = {
    'L': 0,
    'R': 1,
    'f': 0,
    'j': 1
}
hmaze_keymap = {
    'LU': 0,
    'LD': 1,
    'RU': 2,
    'RD': 3,
    'q': 0,
    'z': 1,
    'p': 2,
    'm': 3
}

def fig1c(df):
    subject_ids = df.subject_id.unique()


    acc = []
    for i, s in enumerate(subject_ids):
        df_hmaze = df[(df.subject_id == s) & (df.task == 'HMaze')]
        result = compute_graded_acc(df_hmaze)
        acc.append(result)
        # print(f"subject {i}: {result}")

    acc = np.stack(acc)

    acc_mean = np.mean(acc, axis=0)
    acc_mean = np.flipud(acc_mean)
    fig, ax = plt.subplots(3,1)
    # 
    axi = ax[0]
    # heatmap show colorbar
    h=axi.imshow(acc_mean, cmap='jet',vmin=0, vmax=1)
    axi.set_xlabel('Horizontal Diff.')
    axi.set_ylabel('Vertical Diff.')

    axi = ax[1]
    axi.imshow(np.flipud(acc_mean.mean(0, keepdims=1).T), cmap='jet',vmin=0, vmax=1)
    axi.set_title('Horizontal marginal')

    axi = ax[2]
    axi.imshow(acc_mean.mean(1, keepdims=1), cmap='jet',vmin=0, vmax=1)
    fig.colorbar(h, ax=axi)
    axi.set_title('Horizontal marginal')

    plt.show()


def compute_graded_acc(df):
    arms = df['condition'].values
    arms = np.stack([eval(a) for a in arms])
    horizontal_diff = np.abs(arms[:, 1] - arms[:, 0])
    vertical_diff = np.zeros_like(horizontal_diff)
    for i in range(len(arms)):
        if df.iloc[i].target in ['LU', 'LD']:
            vertical_diff[i] = np.abs(arms[i, 2] - arms[i, 3])
        elif df.iloc[i].target in ['RU', 'RD']:
            vertical_diff[i] = np.abs(arms[i, 4] - arms[i, 5])
    horizontal_diff = (horizontal_diff / 30).astype(int)
    vertical_diff = (vertical_diff / 30).astype(int)
    acc = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            maze_id = (horizontal_diff == i) & (vertical_diff == j)
            target = df.iloc[maze_id].target.values
            response = df.iloc[maze_id].response.values
            for k in range(len(target)):
                target[k] = hmaze_keymap[target[k]]
                response[k] = hmaze_keymap[response[k]]

            acc[i,j] = np.mean(target == response)
    return acc



if __name__ == '__main__':
    proj_dir = Path(__file__).parents[1] 

    raw_data_name = proj_dir / 'human/t-maze-and-h-maze-raw.csv'
    save_data_name = proj_dir / 'human/t-h-maze.csv'

    if save_data_name.exists():
        df_transform = pandas.read_csv(save_data_name)
    else:
        df = pandas.read_csv(raw_data_name)
        subjects = df.run_id.unique()
        df_transform = pandas.DataFrame(columns=['subject_id', 'task', 'condition', 'target', 'response'])
        for s in subjects:
            df_subject = df[df.run_id == s]
            run_id = df_subject.run_id.values[0]
            prolific_id = df_subject.PROLIFIC_PID.values[0]

            idx_tmaze = np.where(df_subject.task == 'TMaze')[0]
            idx_tmaze = idx_tmaze[4:]
            df_s_tmaze = df_subject.iloc[idx_tmaze].loc[:, ['direction',
                                                            'key_pressed',
                                                            'horizontal_arms',
                                                            'task',
                                                            'run_id',
                                                            'PROLIFIC_PID']]

            df_s_tmaze.rename(columns={'direction': 'target',
                                       'key_pressed': 'response',
                                       'horizontal_arms': 'condition',
                                       'run_id': 'subject_id'},
                              inplace=True)

            idx_hmaze = np.where(df_subject.task == 't-shape-detection')[0]
            idx_hmaze = idx_hmaze[5:]
            df_s_hmaze = df_subject.iloc[idx_hmaze].loc[:, ['direction',
                                                            'key_pressed',
                                                            'horizontal_arms',
                                                            'task',
                                                            'run_id',
                                                            'PROLIFIC_PID']]

            df_s_hmaze.rename(columns={'direction': 'target',
                                       'key_pressed': 'response',
                                       'horizontal_arms': 'condition',
                                       'run_id': 'subject_id'},
                              inplace=True)
            df_s_hmaze.task = 'HMaze'
            if len(df_s_tmaze) < 20 or len(df_s_hmaze) < 20:
                print(run_id, prolific_id)
                print(f"t-maze: {len(df_s_tmaze)}, h-maze: {len(df_s_hmaze)}")
                continue
            df_transform = pandas.concat([df_transform, df_s_tmaze, df_s_hmaze], axis=0)

        df_transform['condition'] = df_transform['condition'].apply(eval).apply(
            lambda row: np.array(row).round(1).tolist())
        df_transform.to_csv(save_data_name)

    save_dir = proj_dir / 'save_data'

    fig1c(df_transform)
