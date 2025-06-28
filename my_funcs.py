import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import pytz
import matplotlib.pyplot as plt
import scipy
from arch import arch_model
from arch.univariate import GARCH
import math



def get_daily_log_returns(symbol, start_date, end_date):
    """Get the daily logarithmic returns for a stock.

    This function reads historical stock price data from downloaded CSV files in the folder StockPriceData,
    calculates the daily logarithmic returns over a specified date range,
    displays a plot of the closing prices for that period, and returns
    the calculated log returns.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'AAPL', 'GOOG').
            A CSV file named '{symbol}.csv' must exist in the
            'StockPriceData/' directory.
        start_date (datetime.datetime): The timezone-naive start date for
            the data range.
        end_date (datetime.datetime): The timezone-naive end date for
            the data range.

    Returns:
        pandas.Series: A Series containing the daily logarithmic returns
        for the specified period.

    Side Effects:
        Displays a matplotlib plot of the stock's closing prices against
        the dates for the given range.

    Example:
        >>> from datetime import datetime
        >>> # Assuming 'StockPriceData/AAPL.csv' exists.
        >>> start = datetime.datetime(2023, 1, 1)
        >>> end = datetime.datetime(2023, 12, 31)
        >>> log_returns = get_daily_log_returns('AAPL', start, end)
        >>> print(log_returns.head())
    """
    eastern = pytz.timezone('US/Eastern')
    stockprices = pd.read_csv(f'StockPriceData/{symbol}.csv')
    stockprices['Date'] = pd.to_datetime(stockprices['Date'], utc=True)
    stockprices['Date'] = stockprices['Date'].dt.tz_convert('US/Eastern')
    stockprices['close_prev'] = stockprices['Close'].shift(1)
    stockprices['log_ret'] = np.log(stockprices['Close']/stockprices['close_prev'])
    stockprices = stockprices[(stockprices['Date']>eastern.localize(start_date)) & (stockprices['Date']<eastern.localize(end_date))]
    plt.plot(stockprices['Date'], stockprices['Close'])
    plt.show()
    return stockprices['log_ret']


def get_garch_params(log_rets):
    """Fits a GARCH(1,1) model to a series of logarithmic returns.

    This function takes a time series of logarithmic returns, scales them
    by 100 (for optimization purposes), and fits a standard GARCH(1,1) model to estimate
    volatility parameters. The summary of the fitted model is printed to
    the console.

    Args:
        log_rets (pd.Series or np.array): A time series of logarithmic
            returns for a financial asset.

    Returns:
        pd.Series: A pandas Series containing the fitted parameters of the
        GARCH model (mu, omega, alpha[1], beta[1]).

    Side Effects:
        Prints the detailed summary of the GARCH model fitting results
        to standard output.

    Example:
        >>> # Generate some sample log returns
        >>> np.random.seed(0)
        >>> sample_returns = np.random.randn(1000) * 0.02
        >>> # Get the GARCH parameters
        >>> params = get_garch_params(sample_returns)
        >>> print(params)
        mu          -0.003952
        omega        0.000185
        alpha[1]     0.100000
        beta[1]      0.880000
        Name: params, dtype: float64
    """
    model = arch_model(100*log_rets, vol='GARCH', p=1, q=1)
    res = model.fit()
    print(res.summary)
    return res.params

def simulate_from_garch_params(sigma_start, omega, alpha, beta, n_paths, n_steps):
    """Simulates multiple paths from a GARCH(1,1) process.

    This function generates a specified number of time series paths using a
    GARCH(1,1) model with given parameters. For each path, it simulates a
    sequence of random errors (shocks) and their corresponding conditional
    variances.

    Args:
        sigma_start (float): The initial volatility (standard deviation, σ)
            used to set the starting variance (σ²) for the simulation.
        omega (float): The constant term (ω) in the GARCH variance equation.
        alpha (float): The ARCH parameter (α), governing the reaction to
            past squared errors.
        beta (float): The GARCH parameter (β), governing the persistence of
            volatility.
        n_paths (int): The number of separate simulation paths to generate.
        n_steps (int): The number of time steps (observations) in each path.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
        - errors (np.ndarray): A 2D array of shape (n_paths, n_steps)
          containing the simulated random errors (ε_t) for each path.
        - variances (np.ndarray): A 2D array of shape (n_paths, n_steps)
          containing the simulated conditional variances (σ²_t) for each path.
          
    Example:
        >>> # GARCH parameters
        >>> initial_vol = 0.02
        >>> omega_param = 0.00001
        >>> alpha_param = 0.05
        >>> beta_param = 0.94
        >>>
        >>> # Simulation settings
        >>> paths_to_sim = 10
        >>> steps_per_path = 252 # One trading year
        >>>
        >>> errors, variances = simulate_from_garch_params(
        ...     sigma_start=initial_vol,
        ...     omega=omega_param,
        ...     alpha=alpha_param,
        ...     beta=beta_param,
        ...     n_paths=paths_to_sim,
        ...     n_steps=steps_per_path
        ... )
        >>>
        >>> print(f"Shape of errors array: {errors.shape}")
        Shape of errors array: (10, 252)
        >>> print(f"Shape of variances array: {variances.shape}")
        Shape of variances array: (10, 252)
    """
    garch = GARCH(p=1, q=1)
    errors = np.zeros((n_paths, n_steps))
    variances = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        output = garch.simulate(
            parameters=[omega, alpha, beta],
            nobs=n_steps,
            initial_value=sigma_start**2,  # starting sigma² (variance)
            rng=np.random.standard_normal
        )
        errors[i]=output[0]
        variances[i]=output[1]
    return (errors, variances)


def stock_path_garch_sigma(S0, r, n_paths, t, garch_params):
    """Simulates stock price paths using outputs from a GARCH model.

    This function generates multiple stock price paths based on a geometric
    Brownian motion framework, where the volatility component is derived from
    the simulated errors of a pre-specified GARCH(1,1) model.

    Args:
        S0 (float): The initial stock price at time t=0.
        r (float): The annual risk-free interest rate (e.g., 0.05 for 5%).
        n_paths (int): The number of distinct stock price paths to simulate.
        t (float): The time horizon for the simulation in years (e.g., 1 for
            one year).
        garch_params (pd.Series): A pandas Series containing the fitted
            parameters of a GARCH(1,1) model. It must include 'mu', 'omega',
            'alpha[1]', and 'beta[1]'.

    Returns:
        np.ndarray: A 2D numpy array of shape (n_paths, n_steps) containing
        the simulated stock price paths. `n_steps` is calculated as
        `ceil(t * 252)`.
    """
    
    n_steps=math.ceil(t*252)
    
    simulated_sigma = simulate_from_garch_params(sigma_start=garch_params.omega, omega=garch_params.omega, alpha=garch_params.loc['alpha[1]'], beta=garch_params.loc['beta[1]'], n_paths=n_paths, n_steps=n_steps)
    sigma_adjusted = simulated_sigma[0]/100
        
    noise = np.random.normal(0,1,(n_paths,n_steps))
  
    increments = (garch_params.mu/100 + r*(1/252) - .5*sigma_adjusted**2) + sigma_adjusted*noise
    

    log_returns = np.cumsum(increments, axis = 1)
    
    paths = S0*np.exp(log_returns)

    
    return paths
    
def mc_sim_call_garch(S0, K, r, n_paths, t, garch_params):
    """Prices a European call option using a GARCH-based Monte Carlo simulation.

    This function estimates the value of a European call option by simulating
    a large number of possible stock price paths using a GARCH-driven model.
    It calculates the payoff of the option at expiration for each path,
    averages these payoffs, and discounts the result to its present value.

    Args:
        S0 (float): The initial stock price at time t=0.
        K (float): The strike price of the call option.
        r (float): The annual risk-free interest rate (e.g., 0.05 for 5%).
        n_paths (int): The number of simulation paths for the Monte Carlo
            estimation. A higher number leads to a more accurate price.
        t (float): The time to expiration of the option, in years.
        garch_params (pd.Series): A pandas Series containing the fitted
            parameters of a GARCH(1,1) model, required by the path
            simulation function.

    Returns:
        float: The estimated present value (price) of the European call option.
    """
    
    paths = stock_path_garch_sigma(S0, r, n_paths, t, garch_params)

    end_points = paths[:,-1]
    
    call_payouts = np.maximum(end_points - K, 0)

    call_sim_value = np.exp(-r*t)*np.mean(call_payouts)

    return call_sim_value

def mc_sim_call_garch_delta(S0, K, t, r, n_paths, garch_params):
    """Calculates the Delta of a European call option using a GARCH model.

    This function estimates the option's Delta, the rate of change of the
    option price with respect to the underlying asset's price. It uses a
    "pathwise" finite difference method within a Monte Carlo simulation
    framework where the asset's volatility is driven by a GARCH process.

    Args:
        S0 (float): The initial stock price at time t=0.
        K (float): The strike price of the call option.
        t (float): The time to expiration of the option, in years.
        r (float): The annual risk-free interest rate (e.g., 0.05 for 5%).
        n_paths (int): The number of simulation paths for the Monte Carlo
            estimation.
        garch_params (pd.Series): A pandas Series containing the fitted
            parameters of a GARCH(1,1) model. Must include 'omega',
            'alpha[1]', and 'beta[1]'.

    Returns:
        float: The estimated Delta of the European call option.

    Notes:
        - The function uses the "pathwise" method for calculating Delta. The same
          set of random numbers is used to simulate the paths for both the
          upwardly-bumped price (S0 + bump) and downwardly-bumped price
          (S0 - bump).
        - The drift term used in the path simulation is based on the risk-free
          rate `r`, for risk-neutral option pricing. It
          does not use the `mu` parameter from the GARCH fit.
        - The finite difference is calculated using a small "bump" equal to
          1% of the initial stock price S0.
    """

    n_steps = math.ceil(t*252)
    
    bump = .01*S0 
    simulated_sigma = simulate_from_garch_params(sigma_start=garch_params.omega, omega=garch_params.omega, alpha=garch_params.loc['alpha[1]'], beta=garch_params.loc['beta[1]'], n_paths=n_paths, n_steps=n_steps)
    sigma_adjusted = simulated_sigma[0]/100
        
    noise = np.random.normal(0,1,(n_paths,n_steps))
    
    increments = (r*(1/252) - .5*sigma_adjusted**2) + sigma_adjusted*noise
    
    log_returns = np.cumsum(increments, axis = 1)
    
    paths_up = (S0+bump)*np.exp(log_returns[:,-1])

    paths_down = (S0-bump)*np.exp(log_returns[:,-1])

    calls_up = np.mean(np.maximum(paths_up - K, 0)*np.exp(-r*t), axis=1)

    calls_down = np.mean(np.maximum(paths_down - K, 0)*np.exp(-r*t), axis=1)

    
    delta_sims = (calls_up - calls_down)/(2*bump.reshape(1,-1))

    return delta_sims[0]

def mc_sim_call_garch_delta_hedge(S0, K, garch_params, r, t, n_sims = 2500, n_hedges = 50, delta_sims = 250):
    """Simulates the P&L of a delta-hedged short call option portfolio.

    This function performs a Monte Carlo simulation to analyze the performance
    of a delta-hedging strategy for a European call option where the underlying
    asset follows a GARCH-driven process. It simulates multiple stock price
    paths and, for each path, periodically rebalances a hedging portfolio.
    The final result is a distribution of the hedging errors (Profit & Loss).

    Args:
        S0 (float): The initial stock price at time t=0.
        K (float): The strike price of the call option.
        garch_params (pd.Series): A pandas Series containing the fitted
            parameters of a GARCH(1,1) model.
        r (float): The annual risk-free interest rate.
        t (float): The time to expiration of the option, in years.
        n_sims (int, optional): The number of primary stock price paths to
            simulate for the hedging strategy. Defaults to 2500.
        n_hedges (int, optional): The number of discrete re-hedging intervals
            over the option's life. Defaults to 50.
        delta_sims (int, optional): The number of paths for the *nested* Monte
            Carlo simulation used to calculate Delta at each re-hedging step.
            Defaults to 250.

    Returns:
        np.ndarray: A 1D numpy array of shape (n_sims,) containing the final
        P&L (hedging error) for each of the simulated paths. A value near
        zero indicates a successful hedge for that path.

    """

    hedge_time_points = np.vectorize(math.ceil)(np.linspace(0, 1, n_hedges + 1)*t*252)

    if n_hedges !=0:
        dt = t/n_hedges

    paths = stock_path_garch_sigma(S0, r, n_sims, t, garch_params)
    paths = np.insert(paths, 0, S0, axis = 1)

    path_end_points = paths[:,-1]

    call_payouts = np.maximum(path_end_points - K,0)*np.exp(-r*t)

    stock_profits = []

    for i in range(0,n_hedges):
        stock_start = paths[:,hedge_time_points[i]]
        stock_end = paths[:,hedge_time_points[i+1]]
        tte = t-i*dt
        deltas = mc_sim_call_garch_delta(stock_start.reshape(-1,1), K, tte, r,delta_sims, garch_params)
        #print(deltas)
        stock_profit = (stock_end - stock_start*np.exp(r*dt))*deltas*np.exp(-(i+1)*dt*r)
        stock_profits.append(stock_profit)

    
    total_stock_profit = np.sum(stock_profits, axis = 0)

    profits_hedged = total_stock_profit-call_payouts
    
    #print(call_payouts)
    #print(total_stock_profit)
    return profits_hedged


