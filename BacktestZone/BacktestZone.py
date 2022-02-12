import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime as dt
import ffn
from Functions import *
import pandas_ta as ta
from PIL import Image
import performanceanalytics.statistics as pas
import performanceanalytics.table.table as pat
import investpy as inv

logo = Image.open('BacktestZone_logo.png')

st.set_page_config(
    page_title = 'A No-Code tool to Backtest Trading Strategies for Free',
    page_icon = logo,
    layout = 'wide'
)

st.sidebar.title('PirateHook')
st.sidebar.markdown('')
st.sidebar.markdown('')

backtest_timeframe = st.sidebar.expander('BACKTEST TIMEFRAME')

start_date = backtest_timeframe.date_input('Starting Date', value = dt(2017,1,1), min_value = dt(2015,1,1), max_value = dt(2019,1,1))
str_start_date = str(start_date)
start_date = str_start_date[-2:] + '/' + str_start_date[5:7] + '/' + str_start_date[:4]

end_date = backtest_timeframe.date_input('Ending Date', min_value = dt(2021,1,10))
str_end_date = str(end_date)
end_date = str_end_date[-2:] + '/' + str_end_date[5:7] + '/' + str_end_date[:4]

assets_list = ['Stocks', 'Cryptocurrencies', 'Forex', 'Funds', 'Commodities', 'ETFs', 'Indices', 'Bonds']
asset = st.sidebar.selectbox('ASSET', assets_list, key = 'assets_selectbox')
scripts = import_scripts(asset_type = asset)

symbol = st.sidebar.selectbox('SCRIPT', scripts)
if asset == 'Stocks':
    ticker = str(symbol).split('(')[1][:-1]
else:
    ticker = symbol

indicators = import_indicators()
indicator = st.sidebar.selectbox('INDICATOR', indicators)

df_start = dt(2013,1,1)
str_dfstart_date = str(df_start)[:-9]
df_start = str_dfstart_date[-2:] + '/' + str_dfstart_date[5:7] + '/' + str_dfstart_date[:4]

data = import_data(asset, ticker, df_start, end_date)

    
# 1. SUPERTREND
if indicator == 'SuperTrend':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_supertrend(st, data, start_date, end_date)
        
# 2. -DI, NEGATIVE DIRECTIONAL INDEX
if indicator == '-DI, Negative Directional Index':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_negative_directional_index(st, data, start_date, end_date)
    
# 3. NORMALIZED AVERAGE TRUE RANGE
if indicator == 'Normalized Average True Range (NATR)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_normalized_average_true_range(st, data, start_date, end_date)
    
# 4. AVERAGE DIRECTIONAL INDEX    
if indicator == 'Average Directional Index (ADX)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_average_directional_index(st, data, start_date, end_date)
    
# 5. STOCHASTIC OSCILLATOR FAST
if indicator == 'Stochastic Oscillator Fast (SOF)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_stochastic_oscillator_fast(st, data, start_date, end_date)
    
# 6. STOCHASTIC OSCILLATOR SLOW
if indicator == 'Stochastic Oscillator Slow (SOS)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_stochastic_oscillator_slow(st, data, start_date, end_date)

# 7. WEIGHTED MOVING AVERAGE
if indicator == 'Weighted Moving Average (WMA)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_weighted_moving_average(st, data, start_date, end_date)
        
# 8. MOMENTUM INDICATOR
if indicator == 'Momentum Indicator (MOM)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_momentum_indicator(st, data, start_date, end_date)
        
# 7. VORTEX INDICATOR
if indicator == 'Vortex Indicator (VI)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_vortex_indicator(st, data, start_date, end_date)
        
# 8. CHANDE MOMENTUM OSCILLATOR
if indicator == 'Chande Momentum Oscillator (CMO)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_chande_momentum_oscillator(st, data, start_date, end_date)
    
# 9. EXPONENTIAL MOVING AVERAGE
if indicator == 'Exponential Moving Average (EMA)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_exponential_moving_average(st, data, start_date, end_date)
        
# 10. TRIPLE EXPONENTIAL MOVING AVERAGE
if indicator == 'Triple Exponential Moving Average (TEMA)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_triple_exponential_moving_average(st, data, start_date, end_date)
    
# 11. DOUBLE EXPONENTIAL MOVING AVERAGE
if indicator == 'Double Exponential Moving Average (DEMA)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_double_exponential_moving_average(st, data, start_date, end_date)
    
# 12. SIMPLE MOVING AVERAGE
if indicator == 'Simple Moving Average (SMA)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_simple_moving_average(st, data, start_date, end_date)
    
# 13. TRIANGULAR MOVING AVERAGE
if indicator == 'Triangular Moving Average (TRIMA)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_triangular_moving_average(st, data, start_date, end_date)
    
# 14. CHANDE FORECAST OSCILLATOR
if indicator == 'Chande Forecast Oscillator (CFO)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_chande_forecast_oscillator(st, data, start_date, end_date)
    
# 15. CHOPPINESS INDEX
if indicator == 'Choppiness Index':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_choppiness_index(st, data, start_date, end_date)
    
# 16. AROON DOWN
if indicator == 'Aroon Down':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_aroon_down(st, data, start_date, end_date)
    
# 16. AVERAGE TRUE RANGE
if indicator == 'Average True Range (ATR)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_average_true_range(st, data, start_date, end_date)
    
# 17. WILLIAMS %R
if indicator == 'Williams %R':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_williamsr(st, data, start_date, end_date)
    
# 18. PARABOLIC SAR
if indicator == 'Parabolic SAR':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_parabolic_sar(st, data, start_date, end_date)
    
# 19. COPPOCK CURVE
if indicator == 'Coppock Curve':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_coppock_curve(st, data, start_date, end_date)
    
# 20. +DI, POSITIVE DIRECTIONAL INDEX
if indicator == '+DI, Positive Directional Index':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_positive_directional_index(st, data, start_date, end_date)
    
# 21. RELATIVE STRENGTH INDEX
if indicator == 'Relative Strength Index (RSI)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_rsi(st, data, start_date, end_date)
    
# 22. MACD Signal
if indicator == 'MACD Signal':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_macd_signal(st, data, start_date, end_date)
    
# 23. AROON OSCILLATOR
if indicator == 'Aroon Oscillator':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_aroon_oscillator(st, data, start_date, end_date)

# 24. STOCHASTIC RSI FASTK
if indicator == 'Stochastic RSI FastK':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_stochrsi_fastk(st, data, start_date, end_date)
    
# 25. STOCHASTIC RSI FASTD
if indicator == 'Stochastic RSI FastD':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_stochrsi_fastd(st, data, start_date, end_date)
    
# 26. ULTIMATE OSCILLATOR
if indicator == 'Ultimate Oscillator':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_ultimate_oscillator(st, data, start_date, end_date)

# 27. AROON UP
if indicator == 'Aroon Up':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_aroon_up(st, data, start_date, end_date)
    
# 28. BOLLINGER BANDS
if indicator == 'Bollinger Bands':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_bollinger_bands(st, data, start_date, end_date)
    
# 29. TRIX
if indicator == 'TRIX':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_trix(st, data, start_date, end_date)
    
# 30. COMMODITY CHANNEL INDEX
if indicator == 'Commodity Channel Index (CCI)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_cci(st, data, start_date, end_date)
    
# 31. MACD
if indicator == 'MACD':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_macd(st, data, start_date, end_date)
    
# 31. MACD HISTOGRAM
if indicator == 'MACD Histogram':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_macd_histogram(st, data, start_date, end_date)
    
# 32. MONEY FLOW INDEX
if indicator == 'Money Flow Index (MFI)':
    entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_mfi(st, data, start_date, end_date)
    
st.sidebar.markdown('')
cf_bt = st.sidebar.button('Save & Backtest')
if cf_bt == False:
    st.info('Hit the "Save & Backtest" button at the bottom left corner to view the results')
elif cf_bt == True:
    backtestdata = import_data(asset, ticker, start_date, end_date)
    str_start_date = str(backtestdata.index[0])[:10]
    str_end_date = str(backtestdata.index[-1])[:10]
    benchmark_start_date = str_start_date[-2:] + '/' + str_start_date[5:7] + '/' + str_start_date[:4]
    benchmark_end_date = str_end_date[-2:] + '/' + str_end_date[5:7] + '/' + str_end_date[:4]
    benchmark_data = import_data('Indices', 'S&P 500', benchmark_start_date, benchmark_end_date)
    
    bb_index_difference = len(backtestdata) - len(benchmark_data)
    if bb_index_difference > 0:
        benchmark_index_list = list(benchmark_data.index)
        backtestdata = backtestdata[backtestdata.index.isin(benchmark_index_list)]
    else:
        backtestdata_index_list = list(backtestdata.index)
        benchmark_data = benchmark_data[benchmark_data.index.isin(backtestdata_index_list)]
        
    benchmark = benchmark_data.Close.pct_change().dropna()
    
    entry_index_difference = len(entry_data1) - len(entry_data2)
    if entry_index_difference > 0:
        entry_data2_list = list(entry_data2.index)
        entry_data1 = entry_data1[entry_data1.index.isin(entry_data2_list)]
    else:
        entry_data1_list = list(entry_data1.index)
        entry_data2 = entry_data2[entry_data2.index.isin(entry_data1_list)]
        
    exit_index_difference = len(exit_data1) - len(exit_data2)
    if exit_index_difference > 0:
        exit_data2_list = list(exit_data2.index)
        exit_data1 = exit_data1[exit_data1.index.isin(exit_data2_list)]
    else:
        exit_data1_list = list(exit_data1.index)
        exit_data2 = exit_data2[exit_data2.index.isin(exit_data1_list)]
        
    if entry_comparator == '<, Crossing Down' and exit_comparator == '<, Crossing Down':
        buy_price, sell_price, strategy_signals = crossingdown_crossingdown(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == '<, Crossing Down' and exit_comparator == '>, Crossing Up':
        buy_price, sell_price, strategy_signals = crossingdown_crossingup(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2) 
    elif entry_comparator == '<, Crossing Down' and exit_comparator == '==, Equal To':
        buy_price, sell_price, strategy_signals = crossingdown_equalto(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == '>, Crossing Up' and exit_comparator == '<, Crossing Down':
        buy_price, sell_price, strategy_signals = crossingup_crossingdown(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == '>, Crossing Up' and exit_comparator == '>, Crossing Up':
        buy_price, sell_price, strategy_signals = crossingup_crossingup(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == '>, Crossing Up' and exit_comparator == '==, Equal To':
        buy_price, sell_price, strategy_signals = crossingup_equalto(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == '==, Equal To' and exit_comparator == '>, Crossing Up':
        buy_price, sell_price, strategy_signals = equalto_crossingup(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == '==, Equal To' and exit_comparator == '<, Crossing Down':
        buy_price, sell_price, strategy_signals = equalto_crossingdown(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    elif entry_comparator == '==, Equal To' and exit_comparator == '==, Equal To':
        buy_price, sell_price, strategy_signals = equalto_equalto(backtestdata, entry_data1, entry_data2, exit_data1, exit_data2)
    
    position = []
    for i in range(len(strategy_signals)):
        if strategy_signals[i] > 1:
            position.append(0)
        else:
            position.append(0)
        
    for i in range(len(backtestdata.Close)):
        if strategy_signals[i] == 1:
            position[i] = 1
        elif strategy_signals[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
    
    st.caption(f'BACKTEST  RESULTS  FROM  {str_start_date}  TO  {str_end_date}')
    
    st.markdown('')
    
    buy_hold = backtestdata.Close.pct_change().dropna()
    strategy = (position[1:] * buy_hold).dropna()
    
    sb_index_difference = len(strategy) - len(benchmark)
    if sb_index_difference > 0:
        benchmark_index_list = list(benchmark.index)
        strategy = strategy[strategy.index.isin(benchmark_index_list)]
    else:
        strategy_index_list = list(strategy.index)
        benchmark = benchmark[benchmark.index.isin(strategy_index_list)]
    
    strategy_index_list = list(strategy.index)
    benchmark = benchmark[benchmark.index.isin(strategy_index_list)]
    
    strategy_returns_per = np.exp(strategy.sum()) - 1
    bh_returns_per = np.exp(buy_hold.sum()) - 1
    
    n_days = len(backtestdata)
    annualized_returns = 252 / n_days * strategy_returns_per
    
    buy_signals = pd.Series(buy_price).dropna()
    sell_signals = pd.Series(sell_price).dropna()
    total_signals = len(buy_signals) + len(sell_signals)
    
    max_drawdown = pas.max_dd(strategy)
    
    profit = []
    losses = []
    for i in range(len(strategy)):
        if strategy[i] > 0:
            profit.append(strategy[i])
        elif strategy[i] < 0:
            losses.append(strategy[i])
        else:
            pass
        
    profit_factor = pd.Series(profit).sum() / (abs(pd.Series(losses)).sum())
    
    strat_percentage, bh_percentage, annr = st.columns(3)
    strat_percentage = strat_percentage.metric(label = 'Strategy Profit Percentage', value = f'{round(strategy_returns_per*100,2)}%')
    bh_percentage = bh_percentage.metric(label = 'Buy/Hold Profit Percentage', value = f'{round(bh_returns_per*100,2)}%')
    annr = annr.metric(label = 'Annualized Return', value = f'{round(annualized_returns*100,2)}%')
    
    nos, md, pf = st.columns(3)
    nos = nos.metric(label = 'Total No. of Signals', value = f'{total_signals}')
    md = md.metric(label = 'Max Drawdown', value = f'{round(max_drawdown,2)}%')
    pf = pf.metric(label = 'Profit Factor', value = f'{round(profit_factor,2)}')
    
    key_visuals = st.expander('KEY VISUALS')
    
    key_visuals.caption('Strategy Equity Curve')
    scr = pd.DataFrame(strategy.cumsum()).rename(columns = {'Close':'Returns'})
    scr.index = strategy.index
    key_visuals.area_chart(scr)
    
    key_visuals.markdown('')
    key_visuals.markdown('')
    
    key_visuals.caption('Maximum Drawdown')    
    strategy_drawdown = ffn.core.to_drawdown_series(scr.Returns)
    if len(strategy_drawdown) == 0:
        pass
    else:
        bh_drawdown = ffn.core.to_drawdown_series(buy_hold.cumsum())
        strategy_drawdown = strategy_drawdown[((strategy_drawdown != np.float64('NAN')) & (strategy_drawdown != np.float64('-Infinity')))]
        drawdown_index_difference = len(bh_drawdown) - len(strategy_drawdown)
        bh_drawdown = bh_drawdown[drawdown_index_difference:]
        strategy_drawdown.name, bh_drawdown.name = 'Strategy', 'Buy/Hold'
        frames = [strategy_drawdown, bh_drawdown]
        drawdown = pd.concat(frames, axis = 1)
        key_visuals.line_chart(drawdown)
    
    key_visuals.markdown('')
    key_visuals.markdown('')
    
    key_visuals.caption('Buy/Hold Returns Comparison')
    bhr = pd.DataFrame(buy_hold.cumsum()).rename(columns = {'Close':'Buy/Hold'})
    bhr.index = strategy.index
    scr = scr.rename(columns = {'Returns':'Strategy'})
    frames = [bhr, scr]
    bhr_compdf = pd.concat(frames, axis = 1)
    key_visuals.line_chart(bhr_compdf)
    
    key_visuals.markdown('')
    key_visuals.markdown('')
    
    key_visuals.caption('Benchmark Returns Comparison')
    benchmark_cs_returns_df = pd.DataFrame(benchmark.cumsum()).rename(columns = {'Close':'Benchmark'})
    frames = [benchmark_cs_returns_df, scr]
    benchmark_compdf = pd.concat(frames, axis = 1)
    key_visuals.line_chart(benchmark_compdf)
    
    drawdown_details = st.expander('DRAWDOWN DETAILS')
    
    dd_details = ffn.core.drawdown_details(strategy)
    drawdown_details.table(dd_details)
    
    ratios = st.expander('RATIOS')
    
    ratios.caption('Values Assumed:  Benchmark = S&P 500,  Risk-Free Rate = 0.01')
    
    ratios.markdown('')
    
    sharpe = pas.sharpe_ratio(strategy, 0.01)
    calmar = pas.calmar_ratio(strategy, 0.01)
    sortino = sortino_ratio(strategy, 255, 0.01)
    
    sharpe_ratio, calmar_ratio, sortino_ratio = ratios.columns(3)
    sharpe_ratio = sharpe_ratio.metric(label = 'Sharpe Ratio', value = f'{round(sharpe,3)}')
    calmar_ratio = calmar_ratio.metric(label = 'Calmar Ratio', value = f'{round(calmar,3)}')
    sortino_ratio = sortino_ratio.metric(label = 'Sortino Ratio', value = f'{round(sortino,3)}')

    treynor = pas.treynor_ratio(strategy, benchmark, 0.01)
    information = pas.information_ratio(strategy, benchmark)
    modigliani = pas.modigliani_ratio(strategy, benchmark, 0.01)
    
    treynor_ratio, information_ratio, modigliani_ratio = ratios.columns(3)
    treynor_ratio = treynor_ratio.metric(label = 'Treynor Ratio', value = f'{round(treynor,3)}')
    information_ratio = information_ratio.metric(label = 'Information Ratio', value = f'{round(information,3)}')
    modigliani_ratio = modigliani_ratio.metric(label = 'Modigliani Ratio', value = f'{round(modigliani,3)}')
    
    sterling = pas.sterling_ratio(strategy, 0.01, 5) 
    burke = pas.burke_ratio(strategy, 0.01, 5)
    cond_sharpe = pas.conditional_sharpe_ratio(strategy, 0.01, 0.05)

    sterling_ratio, burke_ratio, cond_sharpe_ratio = ratios.columns(3)
    sterling_ratio = sterling_ratio.metric(label = 'Sterling Ratio', value = f'{round(sterling,3)}')
    burke_ratio = burke_ratio.metric(label = 'Burke Ratio', value = f'{round(burke,3)}')
    cond_sharpe_ratio = cond_sharpe_ratio.metric(label = 'Conditional Sharpe Ratio', value = f'{round(cond_sharpe,3)}')
    
    general_statistics = st.expander('GENERAL STATISTICS')
    
    strategy_df = pd.DataFrame(strategy).rename(columns = {'Close':'Strategy'})
    buy_hold_df = pd.DataFrame(buy_hold).rename(columns = {'Close':'Buy/Hold'})
    benchmark_df = pd.DataFrame(benchmark).rename(columns = {'Close':'Benchmark'})
    
    frames = [strategy_df, buy_hold_df, benchmark_df]
    stats_df = pd.concat(frames, axis = 1)
    
    general_stats = pat.stats_table(stats_df, manager_col = 0, other_cols = [1,2])
    general_statistics.table(general_stats)
    
    advanced_statistics = st.expander('ADVANCED STATISTICS')
    
    advanced_stats = pat.create_downside_table(stats_df, [0,1,2])
    advanced_statistics.table(advanced_stats)
