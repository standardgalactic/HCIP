import pandas
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat


# Mappings from text labels to numeric indices for T-maze responses
tmaze_keymap = {
    'L': 0,  # Left target/response
    'R': 1,  # Right target/response 
    'f': 0,  # 'f' key maps to left
    'j': 1   # 'j' key maps to right
}

# Mappings from text labels to numeric indices for H-maze responses
hmaze_keymap = {
    'LU': 0,  # Left-Up target/response
    'LD': 1,  # Left-Down target/response
    'RU': 2,  # Right-Up target/response
    'RD': 3,  # Right-Down target/response
    'q': 0,   # 'q' key maps to Left-Up
    'z': 1,   # 'z' key maps to Left-Down
    'p': 2,   # 'p' key maps to Right-Up
    'm': 3    # 'm' key maps to Right-Down
}

def fig1c(df):
    """
    Generate accuracy heatmaps for H-maze task performance.
    
    Args:
        df: DataFrame containing trial data
    """
    # Get unique subject IDs
    subject_ids = df.subject_id.unique()

    # Calculate accuracy matrices for each subject
    acc = []
    for i, s in enumerate(subject_ids):
        # Get H-maze trials for this subject
        df_hmaze = df[(df.subject_id == s) & (df.task == 'HMaze')]
        result = compute_graded_acc(df_hmaze)
        acc.append(result)

    # Stack individual subject matrices
    acc = np.stack(acc)

    # Calculate mean accuracy across subjects and flip vertically
    acc_mean = np.mean(acc, axis=0)
    acc_mean = np.flipud(acc_mean)
    
    # Create figure with 3 subplots
    fig, ax = plt.subplots(3,1)
    
    # Plot 1: Full accuracy heatmap
    axi = ax[0]
    h = axi.imshow(acc_mean, cmap='jet',vmin=0, vmax=1)
    axi.set_xlabel('Horizontal Diff.')
    axi.set_ylabel('Vertical Diff.')

    # Plot 2: Horizontal marginal accuracy
    axi = ax[1]
    axi.imshow(np.flipud(acc_mean.mean(0, keepdims=1).T), cmap='jet',vmin=0, vmax=1)
    axi.set_title('Horizontal marginal')

    # Plot 3: Vertical marginal accuracy
    axi = ax[2]
    axi.imshow(acc_mean.mean(1, keepdims=1), cmap='jet',vmin=0, vmax=1)
    fig.colorbar(h, ax=axi)
    axi.set_title('Horizontal marginal')

    plt.show()


def compute_graded_acc(df):
    """
    Compute accuracy matrix based on arm length differences.
    
    Args:
        df: DataFrame containing trial data for one subject
        
    Returns:
        3x3 accuracy matrix indexed by horizontal and vertical differences
    """
    # Extract and parse arm length arrays
    arms = df['condition'].values
    arms = np.stack([eval(a) for a in arms])
    
    # Calculate horizontal differences between arms
    horizontal_diff = np.abs(arms[:, 1] - arms[:, 0])
    
    # Calculate vertical differences based on target location
    vertical_diff = np.zeros_like(horizontal_diff)
    for i in range(len(arms)):
        if df.iloc[i].target in ['LU', 'LD']:
            vertical_diff[i] = np.abs(arms[i, 2] - arms[i, 3])  # Left side vertical diff
        elif df.iloc[i].target in ['RU', 'RD']:
            vertical_diff[i] = np.abs(arms[i, 4] - arms[i, 5])  # Right side vertical diff
            
    # Convert differences to integer bins (divide by 30 pixels)
    horizontal_diff = (horizontal_diff / 30).astype(int)
    vertical_diff = (vertical_diff / 30).astype(int)
    
    # Initialize accuracy matrix
    acc = np.zeros((3,3))
    
    # Calculate accuracy for each combination of differences
    for i in range(3):
        for j in range(3):
            # Get trials with this combination of differences
            maze_id = (horizontal_diff == i) & (vertical_diff == j)
            target = df.iloc[maze_id].target.values
            response = df.iloc[maze_id].response.values
            
            # Convert text labels to numeric indices
            for k in range(len(target)):
                target[k] = hmaze_keymap[target[k]]
                response[k] = hmaze_keymap[response[k]]

            # Calculate accuracy as mean of correct responses
            acc[i,j] = np.mean(target == response)
    return acc


if __name__ == '__main__':
    # Set up file paths
    proj_dir = Path(__file__).parents[1] 
    raw_data_name = proj_dir / 'human/t-maze-and-h-maze-raw.csv'
    save_data_name = proj_dir / 'human/t-h-maze.csv'

    # Load or create processed data file
    if save_data_name.exists():
        df_transform = pandas.read_csv(save_data_name)
    else:
        # Load raw data and initialize output DataFrame
        df = pandas.read_csv(raw_data_name)
        subjects = df.run_id.unique()
        df_transform = pandas.DataFrame(columns=['subject_id', 'task', 'condition', 'target', 'response'])
        
        # Process data for each subject
        for s in subjects:
            df_subject = df[df.run_id == s]
            run_id = df_subject.run_id.values[0]
            prolific_id = df_subject.PROLIFIC_PID.values[0]

            # Extract T-maze trials (skip first 4 trials)
            idx_tmaze = np.where(df_subject.task == 'TMaze')[0]
            idx_tmaze = idx_tmaze[4:]
            df_s_tmaze = df_subject.iloc[idx_tmaze].loc[:, ['direction',
                                                            'key_pressed',
                                                            'horizontal_arms',
                                                            'task',
                                                            'run_id',
                                                            'PROLIFIC_PID']]

            # Rename T-maze columns
            df_s_tmaze.rename(columns={'direction': 'target',
                                       'key_pressed': 'response',
                                       'horizontal_arms': 'condition',
                                       'run_id': 'subject_id'},
                              inplace=True)

            # Extract H-maze trials (skip first 5 trials)
            idx_hmaze = np.where(df_subject.task == 't-shape-detection')[0]
            idx_hmaze = idx_hmaze[5:]
            df_s_hmaze = df_subject.iloc[idx_hmaze].loc[:, ['direction',
                                                            'key_pressed',
                                                            'horizontal_arms',
                                                            'task',
                                                            'run_id',
                                                            'PROLIFIC_PID']]

            # Rename H-maze columns
            df_s_hmaze.rename(columns={'direction': 'target',
                                       'key_pressed': 'response',
                                       'horizontal_arms': 'condition',
                                       'run_id': 'subject_id'},
                              inplace=True)
            df_s_hmaze.task = 'HMaze'
            
            # Skip subjects with too few trials
            if len(df_s_tmaze) < 20 or len(df_s_hmaze) < 20:
                print(run_id, prolific_id)
                print(f"t-maze: {len(df_s_tmaze)}, h-maze: {len(df_s_hmaze)}")
                continue
                
            # Combine data from both tasks
            df_transform = pandas.concat([df_transform, df_s_tmaze, df_s_hmaze], axis=0)

        # Convert arm length strings to rounded numeric arrays
        df_transform['condition'] = df_transform['condition'].apply(eval).apply(
            lambda row: np.array(row).round(1).tolist())
        df_transform.to_csv(save_data_name)

    save_dir = proj_dir / 'save_data'

    # Generate accuracy plots
    fig1c(df_transform)
