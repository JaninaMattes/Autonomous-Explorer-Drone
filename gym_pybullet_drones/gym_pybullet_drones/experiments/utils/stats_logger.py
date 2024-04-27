import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import os

# switch style
sns.set_theme()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
diverging_colors = sns.color_palette("RdBu", 10)

# setup plotting 
plt.rcParams["figure.figsize"] = [12.50, 9.50]
plt.rcParams["figure.autolayout"] = True


class StatsLogger:
    """ Log all statistical values over the runtime
    """
    def __init__(self) -> None:
        pass

    def write_csv(self):
        pass

    def plot_csv(self):
        pass

    # TODO: Q&A - clarify Stats Wrapper


class StatsPlotter:
    """Plotting collected network statistics as graphs."""
    def __init__(self, csv_folder_path, file_name_and_path) -> None:
        self.csv_folder_path = csv_folder_path
        self.file_name_and_path = file_name_and_path
    
    def get_files(self, path):
        """Get all files from a path"""
        all_files = glob.glob(os.path.join(path , "/*.csv"))
        files = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            files.append(df)
        return files

    def read_csv(self):
        """ Use file path to collect all csv files.
            Concats all files and returns a pandas dataframe.
        """
        # find all csv files in this folder
        if os.path.exists(self.csv_folder_path):
            csv_file = os.path.join(self.csv_folder_path, f"*.csv")
            all_files = glob.glob(csv_file)
            if len(all_files) > 0:
                try:
                    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
                    return df
                except:
                    print(f'No *csv file under {csv_file}')
                    return None
            else:
                return None
        else:
            return None

    def plot_box(self, dataframe, x, y, title='title', x_label='Timestep', y_label='Mean Episodic Time', wandb=None):
        """Create a box plot for time needed to converge per experiment."""     
        # get exp names
        df_exp = dataframe['experiment'].values.astype('str')
        df_exp_unique = list(dict.fromkeys(df_exp))
        # plot boxes
        ax = sns.barplot(data=dataframe, x=x, y=y)
        ax.set(title=title, xlabel=x_label, ylabel=y_label)
        # set legend
        plt.legend(labels=df_exp_unique, loc='center right')
        # plot the file to given destination
        ax.figure.savefig(self.file_name_and_path)
        plt.show()

        if wandb:
            wandb.log({'Mean Episodic Time': plt})

    def plot_seaborn_fill(self, 
                dataframe, 
                x, y, 
                y_min, y_max, 
                title='title', 
                x_label='Episode', 
                y_label='Mean Episodic Return', 
                color='blue', 
                smoothing=None, 
                wandb=None):

        # get values from df
        # add smoothing
        if smoothing:
            df_min = gaussian_filter1d(dataframe[y_min].to_numpy(), sigma=smoothing)
            df_max = gaussian_filter1d(dataframe[y_max].to_numpy(), sigma=smoothing)
            df_y = gaussian_filter1d(dataframe[y].to_numpy(), sigma=smoothing)
        else:
            df_min = dataframe[y_min].to_numpy()
            df_max = dataframe[y_max].to_numpy()
            df_y = dataframe[y].to_numpy()
            
        df_x = dataframe[x].to_numpy(dtype=int)
        # get exp names
        df_exp = dataframe['experiment'].values.astype('str')
        df_exp_unique = list(dict.fromkeys(df_exp))

        # draw mean line
        ax = sns.lineplot(x=df_x, y=df_y, lw=2)
        # fill std
        ax.fill_between(x=df_x, y1=df_min, y2=df_max, color=sns.xkcd_rgb[color], alpha=0.2)     
        # change x-y axis scale
        ax.set(title=title, xlabel=x_label, ylabel=y_label)
        # set legend
        plt.legend(labels=df_exp_unique, loc='upper right')

        # plot the file to given destination
        ax.figure.savefig(self.file_name_and_path)
        
        # show for 3 sec
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        if wandb:
            images = wandb.Image(plt)
            wandb.log({'Mean Episodic Return': images})


    def plot_seaborn(self, dataframe, x, y, hue='hue', title='title', x_label='Timestep', y_label='Mean Episodic Return', upper_bound=0, lower_bound=-1800, wandb=None):
        """ Create a lineplot with seaborn.
            Doc: https://seaborn.pydata.org/tutorial/introduction
        """
        g = sns.relplot(data=dataframe, x=x, y=y, hue=hue, kind='line')

        (g
            # draw line into log
            .map(plt.axhline, y=upper_bound, color=".7", dashes=(2, 1), zorder=0)
            .map(plt.axhline, y=lower_bound, color=".7", dashes=(2, 1), zorder=0)
            .set_axis_labels(x_label, y_label)
            .set_titles(title)
            .tight_layout(w_pad=0))

        # plot the file to given destination
        g.figure.savefig(self.file_name_and_path)
        plt.show()

        if wandb:
            wandb.log({'Mean Episodic Return': plt})
    
    def plot_matplot(self, x_values, y_values, y_lower, y_upper, wandb=None):
        """ Create a matplot lineplot with filling between ."""
        plt.fill_between(x_values, y_lower, y_upper, alpha=0.2) # standard deviation
        plt.plot(x_values, y_values) # plotted mean 
        plt.show()

        if wandb:
            wandb.log({'Mean Episodic Return': plt})
    
    def get_random_col(self):
        colour = (np.random.random(), np.random.random(), np.random.random())
        return colour

class CSVWriter:
    """Log the network outputs via pandas to a CSV file.
    """
    def __init__(self, filename: str):
        self.count = 1
        self.filename = filename

    def __call__(self, data: dict):
        df = pd.DataFrame(data)
        if not os.path.isfile(self.filename):
            df.to_csv(self.filename, header='column_names')
        else: # else it exists so append without writing the header
            df.to_csv(self.filename, mode='a', header=False)