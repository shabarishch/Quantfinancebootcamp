import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import pytz
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
import cvxpy as cp



def get_logrets_daily(symbol, start_date, end_date):
    eastern = pytz.timezone('US/Eastern')
    stockprices = pd.read_csv(f'StockPriceData/{symbol}.csv')
    stockprices['Date'] = pd.to_datetime(stockprices['Date'], utc=True)
    stockprices['Date'] = stockprices['Date'].dt.tz_convert('US/Eastern')
    stockprices['close_prev'] = stockprices['Close'].shift(1)
    stockprices['log_ret'] = np.log(stockprices['Close']/stockprices['close_prev'])
    stockprices = stockprices[(stockprices['Date']>=eastern.localize(start_date)) & (stockprices['Date']<=eastern.localize(end_date))]
    return stockprices['log_ret'].dropna()

def find_min_vol_wts(symbols_list, full_covs):
    selected_covs = full_covs.loc[symbols_list, symbols_list]
    n = len(symbols_list)
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, selected_covs))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return (problem.value, w.value)

def normal_test(data):
    mean = data.mean()
    std = data.std()
    normalised_data = (data - mean)/std
    res = scipy.stats.normaltest(normalised_data)
    print(f'p-value for the null hypothesis that the data comes from a standard normal dist. = {res.pvalue}')

def compare_with_normal_symbol_daily(symbol, start_date, end_date, n_bins, outliers_toexclude):
    data = get_logrets_daily(symbol, start_date, end_date)
    if outliers_toexclude:
        data = remove_outliers(data, outliers_toexclude)
    
    mean = data.mean()
    std = data.std()
    normalised_data = (data - mean)/std
    normal_samples = np.random.randn(len(data))

    # Plot the histograms and the PDF.
    plt.figure(figsize=(8, 6))

    # Plot the histogram of the given data.
    plt.hist(normalised_data, bins=n_bins, density=True, alpha=0.5, color='g', label='Log returns')

    # Plot the histogram of the standard normal sample.
    plt.hist(normal_samples, bins=n_bins, density=True, alpha=0.5, color='b', label='Standard Normal Sample')

    # Plot the PDF of the standard normal distribution.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, 0, 1)
    plt.plot(x, p, 'k', linewidth=2, label='Standard Normal PDF')

    # Add titles and labels.
    plt.title(f'Daily log returns of {symbol} from {start_date.month}/{start_date.day}/{start_date.year} to {end_date.month}/{end_date.day}/{end_date.year} compared with values from a standard normal distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    normal_test(data)
    plt.show()

def compare_with_normal_portfolio_daily(data, start_date, end_date, n_bins, outliers_toexclude):
    if outliers_toexclude:
        data = remove_outliers(data, outliers_toexclude)
    
    mean = data.mean()
    std = data.std()
    normalised_data = (data - mean)/std
    normal_samples = np.random.randn(len(data))

    # Plot the histograms and the PDF.
    plt.figure(figsize=(8, 6))

    # Plot the histogram of the given data.
    plt.hist(normalised_data, bins=n_bins, density=True, alpha=0.5, color='g', label='Log returns')

    # Plot the histogram of the standard normal sample.
    plt.hist(normal_samples, bins=n_bins, density=True, alpha=0.5, color='b', label='Standard Normal Sample')

    # Plot the PDF of the standard normal distribution.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, 0, 1)
    plt.plot(x, p, 'k', linewidth=2, label='Standard Normal PDF')

    # Add titles and labels.
    plt.title(f'Daily log returns of portfolio from {start_date.month}/{start_date.day}/{start_date.year} to {end_date.month}/{end_date.day}/{end_date.year} compared with values from a standard normal distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    normal_test(data)
    plt.show()

def remove_outliers(data, n_sigma):
    mean = data.mean()
    std = data.std()
    return data[(data<(mean+(n_sigma*std))) & (data>(mean-(n_sigma*std)))]

def get_daily_rets(symbols_list, start_date, end_date):
    eastern = pytz.timezone('US/Eastern')
    df = pd.DataFrame()
    for item in symbols_list:
        stockprices = pd.read_csv(f'StockPriceData/{item}.csv')
        stockprices['Date'] = pd.to_datetime(stockprices['Date'], utc=True)
        stockprices['Date'] = stockprices['Date'].dt.tz_convert('US/Eastern')
        stockprices['close_prev'] = stockprices['Close'].shift(1)
        stockprices = stockprices[(stockprices['Date']>=eastern.localize(start_date)) & (stockprices['Date']<=eastern.localize(end_date))]
        stockprices[f'{item}'] = stockprices['Close']/stockprices['close_prev'] - 1
        if df.empty:
            df = stockprices[['Date', f'{item}']]
        else:
            df = pd.merge(df, stockprices[['Date', f'{item}']], how='outer', on='Date')

    return df

def get_logrets_minute(symbol, start, end):
    stockprices = yf.download(symbol, interval='1m', start=start, end=end)
    stockprices['log_ret'] = np.log(stockprices['Close'][symbol]/(stockprices['Close'][symbol].shift(1)))
    return stockprices['log_ret'].dropna()

def compare_with_normal_symbol_minute(symbol, start_date, end_date, n_bins, outliers_toexclude):
    data = get_logrets_minute(symbol, start_date, end_date)
    if outliers_toexclude:
        data = remove_outliers(data, outliers_toexclude)
    
    mean = data.mean()
    std = data.std()
    normalised_data = (data - mean)/std
    normal_samples = np.random.randn(len(data))

    # Plot the histograms and the PDF.
    plt.figure(figsize=(8, 6))

    # Plot the histogram of the given data.
    plt.hist(normalised_data, bins=n_bins, density=True, alpha=0.5, color='g', label='Log returns')

    # Plot the histogram of the standard normal sample.
    plt.hist(normal_samples, bins=n_bins, density=True, alpha=0.5, color='b', label='Standard Normal Sample')

    # Plot the PDF of the standard normal distribution.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, 0, 1)
    plt.plot(x, p, 'k', linewidth=2, label='Standard Normal PDF')

    # Add titles and labels.
    plt.title(f'Per minute log returns of {symbol} from {start_date.month}/{start_date.day}/{start_date.year} to {end_date.month}/{end_date.day}/{end_date.year} compared with values from a standard normal distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    normal_test(data)
    plt.show()

def get_risk_return(symbols_list, weights, end_year):
    eastern = pytz.timezone('US/Eastern')
    avgyearlyreturn = 0
    for i in range(len(symbols_list)):
        stockprices = pd.read_csv(f'StockPriceData/{symbols_list[i]}.csv')
        stockprices['Date'] = pd.to_datetime(stockprices['Date'], utc=True)
        stockprices['Date'] = stockprices['Date'].dt.tz_convert('US/Eastern')
        stockprices = stockprices[stockprices['Date']>=eastern.localize(datetime.datetime(end_year-5, 5, 1, 0, 0, 0))]
        stockprices['close_prev'] = stockprices['Close'].shift(252)
        stockprices['growth'] = stockprices['Close']/stockprices['close_prev']
        avgyearlyreturn+= (weights[i])*stockprices[(stockprices['Date']>=eastern.localize(datetime.datetime(end_year-1, 5, 1, 0, 0, 0)))&(stockprices['Date']<eastern.localize(datetime.datetime(end_year, 5, 1, 0, 0, 0)))]['growth'].mean()

    start_date = datetime.datetime(end_year-1, 5, 1, 0, 0, 0)
    end_date = datetime.datetime(end_year, 5, 1, 0, 0, 0)

    # Get daily returns for our selected candidates and compute the covariances of daily returns
    div_categories_stock_data = get_daily_rets(symbols_list, start_date, end_date)
    div_categories_stock_data.drop(columns=['Date'], inplace=True)
    stock_covs = div_categories_stock_data.cov()

    print(f'Average yearly return = {(avgyearlyreturn-1)*100}%, Variance of daily returns = {np.dot(weights, stock_covs@weights)}')
    