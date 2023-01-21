import glob
import random
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
        csv_file = os.path.join(self.csv_folder_path, f"*.csv")
        all_files = glob.glob(csv_file)
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        return df

    def random_colour_generator(self, number_of_colors=16):
        colour = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
        return colour

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
        # show and close automatically
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        if wandb:
            wandb.log({'Mean Episodic Time': plt})

    def plot_seaborn_fill(self, dataframe, x, y, y_min, y_max, title='title', x_label='Timestep', y_label='Mean Episodic Return', upper_bound=0, lower_bound=-1800, xlim_up=3_000_000, ylim_low=-2000, ylim_up=200, color=None, smoothing=2, wandb=None):
            
        # get values from df
        # add smoothing
        df_min = gaussian_filter1d(dataframe[y_min].to_numpy(), sigma=smoothing)
        df_max = gaussian_filter1d(dataframe[y_max].to_numpy(), sigma=smoothing)
        df_x = dataframe[x].to_numpy(dtype=int)
        df_y = gaussian_filter1d(dataframe[y].to_numpy(), sigma=smoothing)
        # get exp names
        df_exp = dataframe['experiment'].values.astype('str')
        df_exp_unique = list(dict.fromkeys(df_exp))

        # random colour generator
        if not color:
            color = self.random_colour_generator()
        else:
            color = sns.xkcd_rgb[color]

        # draw mean line
        ax = sns.lineplot(x=df_x, y=df_y, lw=2)
        # fill std
        ax.fill_between(x=df_x, y1=df_min, y2=df_max, color=color, alpha=0.2)
        # draw upper and lower bounds
        ax.axhline(lower_bound, linewidth=1, color='red', label='lower bound')
        ax.axhline(upper_bound, linewidth=1, color='red', label='upper bound')        
        # change x-y axis scale
        ax.set(title=title, xlabel=x_label, ylabel=y_label, xlim=(0, xlim_up), ylim=(ylim_low, ylim_up))
        # set legend
        plt.legend(labels=df_exp_unique, loc='upper right')

        # plot the file to given destination
        ax.figure.savefig(self.file_name_and_path)
        # show and close automatically
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