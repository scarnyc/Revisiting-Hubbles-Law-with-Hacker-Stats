"""
*********************************************************
hacker_ols.py
Created by William Scardino on 8/15/20

Functions to validate Hubble's Law with Hacker Statistics
for OLS Simple Linear Regression & Hypothesis Testing
*********************************************************
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import probplot, ttest_ind
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.power import TTestIndPower

# set seaborn style
sns.set_style('white')


def load_hubble_data() -> pd.DataFrame:
    """
    Load Edwin Hubble dataset retreived from Source:

    A relation between distance and radial velocity among extra-galactic nebulae
    by Edwin Hubble

    PNAS March 15, 1929 15 (3) 168-173; https://doi.org/10.1073/pnas.15.3.168
    Communicated January 17, 1929

    column names =  Object Name, Distance [Mpc], Velocity [Km/second]
    Notes on units: 1 parsec = 3.26 light years, 1 Mpc = megaparsec = 10 parsees.

    Purpose:
        load Edwin Hubble data .csv file for hacker stats regression

    Returns:
        pandas DataFrame with Hubble's data
    """
    return pd.read_csv(
        r"hubble_data.csv",
        header=9
    )


def normalize_deviations(x: pd.Series, y: pd.Series) -> tuple:
    """
    Purpose:
        Compute normalized deviations of x and y for EDA.

    Args:
        x (pandas Series): explanatory variable to compute normalized deviations
        y (pandas Series): dependent variable to compute normalized deviations

    Returns:
        tuple (ie. deviations of x and y, normalized deviations of x and y)
    """
    # Compute the deviations by subtracting the mean offset
    dx = x - np.mean(x)
    dy = y - np.mean(y)

    # Normalize the data by dividing the deviations by the standard deviation
    zx = dx / np.std(x)
    zy = dy / np.std(y)

    return dx, dy, zx, zy


def plot_deviations(x: pd.Series, y: pd.Series, plt_title: str, xlabel: str, ylabel: str) -> None:
    """
    Purpose:
        Plot normalized deviations of x and y for EDA.

    Args:
        x (pd.Series): explanatory variable to compute normalized deviations
        y (pd.Series): dependent variable to compute normalized deviations
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        plt_title (str): title of Plot

    Returns:
        None
    """
    # Create figure and axis objects and call axis.plot() twice to plot normalized deviations
    # Create scatter connected by line subplots
    fig, axis = plt.subplots()
    axis.plot(x.index, x, linestyle="-", marker=None, color="blue")
    axis.scatter(x.index, x, edgecolors='black', color="blue", label=xlabel)
    axis.plot(y.index, y, linestyle="-", marker=None, color="red")
    axis.scatter(y.index, y, edgecolors='red', color="red", label=ylabel)

    # Add legend, title, x and y axis labels
    axis.legend(loc="best")
    plt.xlabel('Array Index')
    plt.ylabel('Deviations')
    plt.title(plt_title)

    # instantiate tight layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_regplot(x: str, y: str, data: pd.DataFrame, xlabel: str, ylabel: str, legend: str, plt_title: str) -> None:
    """
    Purpose:
        Create Regression Plot for x and y.

    Args:
        x (pandas Series): explanatory variable
        y (pandas Series): dependent variable
        data (pd.DataFrame): DataFrame that holds x and y variables
        xlabel(str): x-axis label
        ylabel(str): y-axis label
        legend(str): legend annotation
        plt_title (str): title of Regression Plot

    Returns:
        None
    """
    # Create Regression Plot with custom Scatter Plot colors and Line Plot color
    axis = sns.regplot(
        x=x,
        y=y,
        data=data,
        scatter_kws=dict(color='#00bfff', edgecolors='b'),
        line_kws=dict(color='red'),
        label=legend
    )

    # Add legend, title, x and y axis labels
    axis.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plt_title)

    # instantiate tight layout and show the plot
    plt.tight_layout()
    plt.show()


def pearson_r(x: pd.Series, y: pd.Series) -> np.float:
    """
    Purpose:
        Calculate Pearson Correlation Coefficient for x and y.

    Args:
        x (pandas Series): explanatory variable
        y (pandas Series): dependent variable

    Returns:
        np.float (ie. Pearson R)
    """
    cor = x.corr(y)
    print('Correlation (pearson r): {:.02f}'.format(cor))

    return cor


def least_squares_sm(x: str, formula: str, df: pd.DataFrame) -> RegressionResultsWrapper:
    """
    Purpose:
        Perform Ordinary Least Squares Linear Regression via statsmodels ols() api.

    Args:
        x (pandas Series): explanatory variable
        formula (str): formula for simple Linear Regression, ie. "y ~ x"
        df (pd.DataFrame): DataFrame that contains x and y variables

    Returns:
        RegressionResultsWrapper (ie. fitted model)
    """
    # Fit the model, based on the form of the formula
    model_fit = ols(formula=formula, data=df).fit()

    # Extract the model parameters and associated "errors" or uncertainties
    a0 = model_fit.params['Intercept']
    a1 = model_fit.params[x]
    e0 = model_fit.bse['Intercept']
    e1 = model_fit.bse[x]

    # Print the results
    print('For slope a1={:.02f}, the uncertainty in a1 is {:.02f}'.format(a1, e1))
    print()
    print('For intercept a0={:.02f}, the uncertainty in a0 is {:.02f}'.format(a0, e0))
    print()
    print(model_fit.summary())

    return model_fit


def rss_rmse(y_data: pd.Series, y_model: pd.Series, model_intercept: np.float, model_slope: np.float) -> tuple:
    """
    Purpose:
        Calculate RSS, residuals, MSE and RMSE for OLS model.

    Args:
        y_data (pd.Series): dependent variable
        y_model (pd.Series): model predictions
        model_intercept (np.float): model intercept
        model_slope (np.float): model slope

    Returns:
        tuple (ie. RSS, residuals, MSE, RMSE)
    """
    # Compute the residuals for the RSS, MSE and RMSE metrics
    residuals = y_model - y_data
    RSS = np.sum(np.square(residuals))
    MSE = RSS / len(residuals)
    RMSE = np.sqrt(MSE)

    print('RMSE = {:0.2f}, MSE = {:0.2f}, RSS = {:0.2f}'.format(RMSE, MSE, RSS))
    print("Model parameters: intercept={:0.2f}, slope={:0.2f} yield RSS={:0.2f}".format(model_intercept, model_slope, RSS))

    return RSS, residuals, MSE, RMSE


def plot_prob_residuals(residuals: pd.Series) -> None:
    """
    Purpose:
        Create a probability plot for the residuals

    Args:
        residuals (pd.Series): dependent variable

    Returns:
        None
    """
    # Create figure and axis objects and call axis.plot() twice to plot normalized deviations
    fig, ax = plt.subplots(figsize=(6, 4))
    # _, (__, ___, r) =
    probplot(residuals, plot=ax, fit=True)

    # Add title, and fit a tight layout
    plt.title('Probability Plot of Residuals \n')
    plt.tight_layout()

    # show the plot
    plt.show()


def least_squares_np(x: pd.Series, y: pd.Series) -> tuple:
    """
    Purpose:
        Perform Ordinary Least Squares Simple Linear Regression via numpy.

    Args:
        x (pd.Series): explanatory variable
        y (pd.Series): dependent variable

    Returns:
        tuple (ie. slope and intercept)
    """
    # prepare the means and deviations of the two variables
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = x - x_mean
    y_dev = y - y_mean

    # Complete least-squares formulae to find the optimal intercept, slope
    slope = np.sum(x_dev * y_dev) / np.sum(x_dev ** 2)
    intercept = y_mean - (slope * x_mean)

    return intercept, slope


def draw_bs_pairs_linreg(x: pd.Series, y: pd.Series, size=1) -> tuple:
    """
    Purpose:
        Perform pairs bootstrap for OLS.

    Args:
        x (pd.Series): explanatory variable
        y (pd.Series): dependent variable
        size (default=1): number of bootstrap samples to generate

    Returns:
        tuple (ie. bootstrap means x and y, bootstrap slopes and  intecepts)
    """
    # set random seed for reproducibility
    np.random.seed(42)

    # Set up array of indices to sample from: inds
    inds = np.arange(0, len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps, bs_means_x, bs_means_y
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    bs_means_x = np.empty(size)
    bs_means_y = np.empty(size)

    # Generate bootstrap replicates using random sampling with replacement
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds), replace=True)
        bs_inds.sort()
        bs_x, bs_y = x[bs_inds], y[bs_inds]

        # generate bootstrap model parameters via least_squares_np()
        bs_intercept_reps[i], bs_slope_reps[i] = least_squares_np(bs_x, bs_y)
        bs_means_x[i] = np.mean(bs_x)
        bs_means_y[i] = np.mean(bs_y)

    return bs_means_x, bs_means_y, bs_slope_reps, bs_intercept_reps


def conf_int95(np_array: np.ndarray) -> np.ndarray:
    """
    Purpose:
        Compute 95% confidence interval.

    Args:
        np_array (np.ndarray): array to compute confidence interval

    Returns:
        np.ndarray (ie. 95% confidence interval)
    """
    # Compute and print 95% CI for slope
    print(np.percentile(np_array, [2.5, 97.5]))
    print()

    return np.percentile(np_array, [2.5, 97.5])


def distr_mean_stde(distribution: np.ndarray) -> tuple:
    """
    Purpose:
        Compute the mean and standard deviation for a distribution.

    Args:
        distribution (np.ndarray): distribution

    Returns:
        tuple (ie. distribution mean and standard deviation)
    """
    # Compute and print the mean, stdev of the resample distribution of means
    distribution_mean = np.mean(distribution)
    standard_error = np.std(distribution)

    print('Bootstrap Distribution: center={:0.2f}, spread={:0.2f}'.format(distribution_mean, standard_error))
    print()

    return distribution_mean, standard_error


def plot_hist_expected_ci(data: np.ndarray, distribution_mean: np.float, ci: np.ndarray, plt_title: str,
                          data_label: str, bins=50, kde=True, draw_ci=True, mean_label="Expected Value",
                          shade=False) -> None:
    """
    Purpose:
        Compute the mean and standard deviation for a distribution.

    Args:
        data (np.ndarray): distribution to plot
        distribution_mean (np.float): distribution mean
        ci (np.ndarray): confidence interval
        plt_title (str): title of the chart
        data_label (str): title of the x-axis
        bins (default=50): number of bins
        kde (default=True): choose whether to show or hide kernel density estimate
        draw_ci (default=True): choose whether to show confidence intervals
        mean_label (default="Expected Value"): choose whether to display mean as "Expected Value", "Estimate"
            or "Effect Size" in legend
        shade (default=False): optionally shade a region gray for customization

    Returns:
        None
    """
    # Create figure and axis objects
    fig, axis = plt.subplots(figsize=(9.5, 5))

    # Create histogram via seaborn
    sns.distplot(data, bins=bins, norm_hist=True, kde=kde, hist_kws=dict(alpha=0.5, color='#00bfff', histtype='bar',
                                                                         ec='black', label=data_label),
                 color='red')
    plt.title(plt_title + ' \n')
    plt.xlabel(data_label)
    plt.ylabel('Probability Density')

    # Add custom axis vertical lines for expected value
    axis.axvline(distribution_mean, label=mean_label, color='black')

    # optionally shade a region gray for customization
    if shade:
        axis.axvspan(0, distribution_mean, alpha=0.5, color='grey', label='Distance from Zero')

    # choose whether to show vertical lines for confidence intervals
    if draw_ci:
        axis.axvline(ci[0], label=' 5th percentile', color='red')
        axis.axvline(ci[1], label='95th percentile', color='red')

    # plot with a tight layout and display the chart
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def rss_minima_viz(x: pd.Series, y: pd.Series, slope_ci95: np.ndarray, intercept_ci95: np.ndarray,
                   size: int) -> pd.DataFrame:
    """
    Purpose:
        Perform pairs bootstrap for linear regression and calculate the RSS minima

    Args:
        x (pd.Series): explanatory variable
        y (pd.Series): dependent variable
        slope_ci95 (np.ndarray): confidence interval for slope
        intercept_ci95 (np.ndarray): confidence interval for intercept
        size (int): number of bootstrap samples to generate

    Returns:
        pd.DataFrame (ie. contains best RSS value, with intercept and slope parameters)
    """
    # set random seed for reproducibility
    np.random.seed(42)

    # Set up array of indices to sample from: inds
    inds = np.arange(0, len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps,
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # instantiate empty list to hold RSS values
    rss_list = []

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds), replace=True)
        bs_inds.sort()
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_intercept_reps[i], bs_slope_reps[i] = least_squares_np(bs_x, bs_y)

        # computing rss for each sample
        y_model = bs_slope_reps[i] * bs_x + bs_intercept_reps[i]
        RSS, residuals, MSE, RMSE = rss_rmse(bs_y, y_model, bs_intercept_reps[i], bs_slope_reps[i])
        rss_list.append(RSS)
        print()

    # create pandas DataFrame with values from Boostrap Replacements
    df = pd.DataFrame({'Bootstrap Slopes': bs_slope_reps,
                       'Bootstrap Intercepts': bs_intercept_reps,
                       'RSS': rss_list})

    # where slopes fall within their 95% confidence intervals
    df = df[df['Bootstrap Slopes'].between(slope_ci95[0], slope_ci95[1])]

    # where intercepts fall within their 95% confidence intervals
    df = df[df['Bootstrap Intercepts'].between(intercept_ci95[0], intercept_ci95[1])]

    # Find the minimum RSS and the a1 value from whence it came
    best_rss = df['RSS'].min()
    print('RSS Minima: {}'.format(best_rss))
    print()
    print(df[df['RSS'] == best_rss])
    print()

    # Create figure and axis objects
    fig, axis = plt.subplots()

    # Plot the RSS against the slope values
    axis.scatter(df['Bootstrap Slopes'], df['RSS'], color="#00bfff", edgecolor='b')

    # Identify the RSS Minima on the plot
    axis.scatter(df[df['RSS'] == best_rss]['Bootstrap Slopes'], df[df['RSS'] == best_rss]['RSS'], color="red",
                 edgecolor='black')

    # Add x- and y-axis labels and chart title
    plt.xlabel("Sample Slopes")
    plt.ylabel("Sample RSS")
    plt.title('Minimum RSS = {:0.2f} \n came from Slope = {:0.2f} \n'.format(df[df['RSS'] == best_rss]['RSS'].values[0],
                                                                             df[df['RSS'] == best_rss]
                                                                             ['Bootstrap Slopes'].values[0]))
    # Plot with a tight layout and display the chart
    plt.tight_layout()
    plt.show()

    return df[df['RSS'] == best_rss]


def plot_linreg(x: pd.Series, y: pd.Series, predictions: pd.Series, plt_title: str, xlabel: str,
                ylabel: str) -> None:
    """
    Purpose:
        Plot the data points with OLS linear regression.

    Args:
        x (pd.Series): explanatory variable
        y (pd.Series): dependent variable
        predictions (pd.Series): model predictions
        plt_title (str): title of chart
        xlabel (str): title of x-axis
        ylabel (str): title of y-axis

    Returns:
        None
    """
    # Create figure and axis objects
    fig, axis = plt.subplots()

    # Plot scatter plot of x and y
    axis.scatter(x, y, edgecolors='b', color="#00bfff", label="measured")

    # Plot lineplot of x and model predictions
    axis.plot(x, predictions, linestyle="-", marker=None, color="red", label="modeled")

    # Add a legend, x- and y-axis labels, and title to the plot
    axis.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plt_title)

    # Plot with a tight layout and then show to display
    plt.tight_layout()
    plt.show()


# def plot_bs_regression(x, y, bs_slopes, bs_intercepts, slope_ci95, intercept_ci95, xlabel, ylabel):
#     # Plot the bootstrap lines
#     for i in range(10):
#         if (slope_ci95[1] >= bs_slopes[i] >= slope_ci95[0]) and (
#                 intercept_ci95[1] >= bs_intercepts[i] >= intercept_ci95[0]):
#             plt.plot(x, bs_intercepts[i] + (bs_slopes[i] * x), linewidth=0.5, alpha=0.2, color='red')
#
#     # Plot the data
#     plt.plot(x, y, marker='.', linestyle='none')
#
#     # Label axes, set the margins, and show the plot
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.margins(0.02)
#     plt.show()


# def resample_short_long(short_distances: pd.Series, long_distances: pd.Series, size: int) -> tuple:
#     # set random seed for reproducibility
#     np.random.seed(42)
#
#     # Then re-sample with replacement, taking 500 random draws from each population.
#     resample_short = np.random.choice(short_distances, size=size, replace=True)
#     resample_long = np.random.choice(long_distances, size=size, replace=True)
#
#     return resample_short, resample_long


def shuffle_and_split(short_distances: pd.Series, long_distances: pd.Series, size: int) -> tuple:
    """
    Purpose:
        Shuffle and and resample distances for Hypothesis Test.

    Args:
        short_distances (pd.Series): explanatory variable
        long_distances (pd.Series): dependent variable
        size (int): number of bootstrap samples to generate

    Returns:
        tuple (ie. 2 shuffled bootstrap samples)
    """
    # set random seed for reproducibility
    np.random.seed(42)

    # Shuffle the time-ordered distances, then slice the result into two populations.
    shuffle_bucket = np.concatenate((short_distances, long_distances))
    np.random.shuffle(shuffle_bucket)
    slice_index = len(shuffle_bucket) // 2
    shuffled_half1 = shuffle_bucket[0:slice_index]
    shuffled_half2 = shuffle_bucket[slice_index:]

    # Create new samples from each shuffled population, and compute the test statistic
    shuffle_resample1 = np.random.choice(shuffled_half1, size=size, replace=True)
    shuffle_resample2 = np.random.choice(shuffled_half2, size=size, replace=True)

    # test_statistic = shuffle_resample2 - shuffle_resample1

    return shuffle_resample1, shuffle_resample2


def test_statistic(resample1: np.ndarray, resample2: np.ndarray) -> tuple:
    """
    Purpose:
        Compute test statistic distribution, effect size and standard error for Hypothesis Test.

    Args:
        resample1 (np.ndarray): explanatory variable
        resample2 (np.ndarray): dependent variable

    Returns:
        tuple (ie. test statistic distribution, effect size and standard error)
    """
    # Difference the resamples to compute a test statistic distribution, then compute its mean and stdev
    test_statistic = resample2 - resample1
    effect_size = np.mean(test_statistic)
    standard_error = np.std(test_statistic)

    # Print the results
    print('Test Statistic: Effect Size={:0.2f}, Standard Error={:0.2f}'.format(effect_size, standard_error))

    return test_statistic, effect_size, standard_error


# def compute_p_value(test_statistic_unshuffled, test_statistic_shuffled):
#     # Compute the effect size for two population groups
#     effect_size = np.mean(test_statistic_unshuffled)
#
#     # Compute the p-value as the proportion of shuffled test stat values >= the effect size
#     condition = test_statistic_shuffled >= effect_size
#     p_value = len(test_statistic_shuffled[condition]) / len(test_statistic_shuffled)
#
#     # Print p-value and overplot the shuffled and unshuffled test statistic distributions
#     print("The p-value is = {}".format(p_value))
#
#     return p_value, effect_size


def t_test(sample1: np.ndarray, sample2: np.ndarray) -> tuple:
    """
    Purpose:
        Run a T-test to compare the differences of 2 means and generate the P-value.

    Args:
        sample1 (np.ndarray): 1st distribution to compare
        sample2 (np.ndarray): 2nd distribution to compare

    Returns:
        tuple (test statistic and p-value)
    """
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    tstat, pval = ttest_ind(sample1, sample2, equal_var=False)
    # tstat, pval = CompareMeans.from_data(sample1, sample2).ztest_ind(usevar='unequal')
    print('The P-value is = {0:0.8f}'.format(pval))
    print()

    return tstat, pval


def power_analysis(effect_size: np.float, alpha=0.05, power=0.95) -> float:
    """
    Purpose:
        Compute sample size needed for a hypothesis test

    Args:
        effect_size (np.float): effect size divided by the standard deviation
        alpha (default=0.05): alpha value, ie. opposite of significance level
        power(default=0.95): probability of statistically signficant results, ie. rejecting the null hypothesis

    Returns:
        float (ie. sample size)
    """
    # instantiate TTestIndPower object
    results = TTestIndPower()

    # calculated sample size
    sample_size = results.solve_power(effect_size=effect_size, alpha=alpha, power=power)
    print('Sample Size for Hypothesis test should be at least {:0.2f} observations'.format(sample_size))

    return sample_size
