#  파이널
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import financedatareader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import tempfile
import pandas as pd
import time
import talib
import numpy as np
import os
import shutil

# Execute실행시 주식종목+설명란 함수
def show_stock_description():
    global daily_data, weekly_data, monthly_data  
    stock_code = entry_label.get()

    if not stock_code:  # Check for validity
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", "주식 코드를 입력해주세요.")
        return

    try:
        daily_data = fdr.DataReader(stock_code, "2018-12-01")
        daily_data['Date'] = pd.to_datetime(daily_data.index) 
        weekly_data = daily_data.resample('W-Mon', on='Date').last()  # 주봉
        monthly_data = daily_data.resample('M', on='Date').last()  # 월봉
        
        # Read the excel file
        df = pd.read_excel('stocks_exp.xlsx')
        
        # Fetch data from the second and third columns
        column2_data = df.loc[df['Code'] == stock_code].iloc[0, 1]
        column3_data = df.loc[df['Code'] == stock_code].iloc[0, 2]
        
        # Update the text widget with the description
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", f"{column2_data}\n\n{column3_data}")
       
    except Exception as e:
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", f"오류: {e}")
        return

# 주식종목 다운로드      
def download_excel():
    source_file_path = "stocks_code.xlsx"  # 원본 파일 경로
    destination_folder = os.path.expanduser("~\\Downloads")  
    try:
        shutil.copy(source_file_path, destination_folder)
        messagebox.showinfo("성공", "파일을 성공적으로 다운로드하였습니다!")
    except Exception as e:
        messagebox.showerror("에러", f"다운로드 중 에러 발생: {str(e)}")

# 상단 entry창 기본입력표시
def on_entry_click(event):
    if entry_label.get() == '종목코드(숫자입력)':
        entry_label.delete(0, "end") # delete all the text in the entry
        entry_label.insert(0, '') # Insert blank for user input
        entry_label.config(fg='black')

def on_focusout(event):
    if entry_label.get() == '':
        entry_label.insert(0, '종목코드(숫자입력)')
        entry_label.config(fg='grey')

        
# 보조지표 처음에 숨기고 나중에 버튼누를경우 나오도록
bollinger_bands_visible = False
moving_averages_visible= False
parabolic_sar_visible = False 
rainbow_chart_visible=False
envelope_chart_visible=False
ichimoku_visible = False
macd_chart_visible = False
stochastic_visible = False
keltner_channels_visible = False
dmi_chart_visible = False
deviation_rate_visible = False
rsi_visible = False 
three_line_reversal_visible = False
ab_ratio_visible = False
price_oscillator_visible = False 
obv_visible = False
volumn_oscillator_visible = False
vr_chart_visible = False

# 이동평균선 함수
def calculate_moving_averages(data, periods=[5, 20, 60, 120]):
    mas = {}
    for period in periods:
        mas[period] = data['Close'].rolling(window=period).mean()
    return mas

# 볼린저밴드 함수
def calculate_bollinger_bands(data, window=20, num_of_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return rolling_mean, upper_band, lower_band

# 그물망(rainbow chart)함수
def rainbow_chart(data, fig, periods=[5,10,15,20,25,30,35,40,45,50,55,60]):
    for period in periods:
        ma = data['Close'].rolling(window=period).mean()
        fig.add_trace(go.Scatter(x=data.index, y=ma, mode='lines',name=f'그물{period}', line=dict(color='lightgray', width=0.5),showlegend=False), row=1, col=1)
    return ma

# Envelop bands 차트함수
def envelope_chart(data, period=20, envelope_percent=10):
    sma = data['Close'].rolling(window=period).mean()
    upper_band = sma * (1 + envelope_percent / 100)
    lower_band = sma * (1 - envelope_percent / 100)
    return sma, upper_band, lower_band


# 일목균형표(Ichimoku) 함수
def calculate_ichimoku(data):
    high_prices = data['High']
    low_prices = data['Low']

    # 전환선(Tenkan-sen)
    tenkan_sen = (high_prices.rolling(window=9).max() + low_prices.rolling(window=9).min()) / 2
    # 기준선(Kijun-sen)
    kijun_sen = (high_prices.rolling(window=26).max() + low_prices.rolling(window=26).min()) / 2
    # 선행스팬1(Senkou Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # 선행스팬2(Senkou Span B)
    senkou_span_b = ((high_prices.rolling(window=52).max() + low_prices.rolling(window=52).min()) / 2).shift(26)
    # 후행스팬(Chikou Span)
    chikou_span = data['Close'].shift(-26)
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span


# MACD 함수
def macd_chart(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    return macd_line, signal_line, macd_hist


# 스토캐스틱 함수
def stochastic(data, k_period=14, d_period=3):
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    # %K 값 계산
    k_value = 100 * (data['Close'] - low_min) / (high_max - low_min)
    # %D 값 계산
    d_value = k_value.rolling(window=d_period).mean()
    return k_value, d_value

#  Keltner Channels 함수
def keltner_channels(data, window=20, multiplier=4):
    rolling_mean = data['Close'].rolling(window=window).mean()
    # Average True Range (ATR) 계산
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=window).mean()
    upper_band = rolling_mean + (multiplier * atr)
    lower_band = rolling_mean - (multiplier * atr)
    return rolling_mean, upper_band, lower_band

#  DMI차트 함수
def dmi_chart(data, period=14):
    high_diff = data['High'].diff()
    low_diff = -data['Low'].diff()
    pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr = pd.concat([data['High'] - data['Low'], 
                    np.abs(data['High'] - data['Close'].shift(1)), 
                    np.abs(data['Low'] - data['Close'].shift(1))], 
                   axis=1).max(axis=1)

    tr_rolling = tr.rolling(period).sum()
    pos_dm_rolling = pd.Series(pos_dm).rolling(period).sum()
    neg_dm_rolling = pd.Series(neg_dm).rolling(period).sum()
    
    pos_di = 100 * pos_dm_rolling / tr_rolling
    neg_di = 100 * neg_dm_rolling / tr_rolling
    adx = 100 * (np.abs(pos_di - neg_di) / (pos_di + neg_di)).rolling(period).mean()
    return pos_di, neg_di, adx

# 이격도 차트 함수
def deviation_rate_chart(data, window=20):
    ma = data['Close'].rolling(window=window).mean()
    deviation_rate = (data['Close'] - ma) / ma * 100
    return deviation_rate

#  RSI(투자심리선) 함수
def rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

#  삼선전환도 함수
def three_line_reversal(data):
    ma1 = data['Close'].rolling(window=1).mean()
    ma2 = data['Close'].rolling(window=2).mean()
    ma3 = data['Close'].rolling(window=3).mean()
    three_line = (ma1 + ma2 + ma3) / 3
    return three_line

#  AB Ratio차트 함수
def calculate_ab_ratio(data):
    # AB Ratio 계산 로직. 예를 들어 A와 B의 데이터를 가져와서 AB Ratio를 계산한다고 가정하면:
    A = data['Close']
    B = data['Volume']  # 이것은 예시입니다. 실제 데이터에 따라 계산 방법을 수정하십시오.
    ab_ratio = A / B
    return ab_ratio

# Price Oscillator차트 함수
def price_oscillator(data, short_period=12, long_period=26):
    short_ma = data['Close'].rolling(window=short_period).mean()
    long_ma = data['Close'].rolling(window=long_period).mean()
    po = (short_ma - long_ma) / long_ma * 100  # Percent Oscillator
    return po


# OBV차트 함수
def calculate_obv(data):
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i-1]:  # If the closing price is above the prior close price 
              obv.append(obv[-1] + data['Volume'][i])
        elif data['Close'][i] < data['Close'][i-1]:  # If the closing price is below the prior close price 
              obv.append(obv[-1] - data['Volume'][i])
        else:  # If closing prices equals the prior close price
              obv.append(obv[-1])
    return obv


#  Volumn Oscillator차트 함수
def volume_oscillator(data, short_window=5, long_window=20):
    short_ma = data['Volume'].rolling(window=short_window).mean()
    long_ma = data['Volume'].rolling(window=long_window).mean()
    vo = (short_ma - long_ma) / long_ma * 100
    return vo


#  VR(Volumn Ratio)차트 함수
def vr_chart(data, window=24):
    delta = data['Close'].diff()
    volume_up = pd.Series(np.where(data['Close'].diff(1) > 0, data['Volume'], 0), index=data.index)
    volume_down = pd.Series(np.where(data['Close'].diff(1) < 0, data['Volume'], 0), index=data.index)
    
    vol_ratio = (volume_up.rolling(window=window).sum() + 
                volume_up.rolling(window=window).mean()) / \
               (volume_down.rolling(window=window).sum() + 
                volume_down.rolling(window=window).mean())
    return vol_ratio

# Reset 함수
def reset_chart_and_entry():
    global bollinger_bands_visible,moving_averages_visible,parabolic_sar_visible,rainbow_chart_visible,envelope_chart_visible,ichimoku_visible,macd_chart_visible,stochastic_visible,keltner_channels_visible,dmi_chart_visible,deviation_rate_visible,rsi_visible,three_line_reversal_visible,ab_ratio_visible,price_oscillator_visible,obv_visible,volumn_oscillator_visible,vr_chart_visible
    # Entry 위젯 초기화
    entry_label.delete(0, "end")
    entry_label.insert(0, '종목코드(숫자입력)')
    entry_label.config(fg='grey')
    
    # 하단 Text 위젯 초기화
    text_widget.delete("1.0", tk.END)
    
    # 보조 지표들 초기화
    bollinger_bands_visible = False
    moving_averages_visible= False
    parabolic_sar_visible = False 
    rainbow_chart_visible=False
    envelope_chart_visible=False
    ichimoku_visible = False
    macd_chart_visible = False
    stochastic_visible = False
    keltner_channels_visible = False
    dmi_chart_visible = False
    deviation_rate_visible = False
    rsi_visible = False 
    three_line_reversal_visible = False
    ab_ratio_visible = False
    price_oscillator_visible = False 
    obv_visible = False
    volumn_oscillator_visible = False
    vr_chart_visible = False
    display_chart("일봉")

    
# 캔들차트 함수
current_chart_type = "일봉"
def display_chart(chart_type):
    global daily_data, weekly_data, monthly_data, bollinger_bands_visible, moving_averages_visible, parabolic_sar_visible, current_chart_type
    current_chart_type = chart_type  # 현재 차트 타입 업데이트
    
    if chart_type == '일봉':
        data = daily_data
    elif chart_type == '주봉':
        data = weekly_data
    elif chart_type == '월봉':
        data = monthly_data
        
    high_values = np.array(data['High'].values, dtype=np.double)
    low_values = np.array(data['Low'].values, dtype=np.double)   
    
    # 주요차트 계산식
    middle_band, upper_band, lower_band = calculate_bollinger_bands(data)
    mas = calculate_moving_averages(data)
    sar = talib.SAR(high_values, low_values, acceleration=0.02, maximum=0.2)    
    
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.15])
   
    # 캔들스틱 차트 추가
    fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            increasing_line_color='red',
            decreasing_line_color='blue',
            increasing_fillcolor='red',
            decreasing_fillcolor='blue',
            showlegend=False), row=1, col=1)

    # 거래량 추가 (항상 하단에 위치)
    colors = ['red' if close >= open else 'blue' for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

     # 레이아웃 설정
    fig.update_layout(plot_bgcolor='white',paper_bgcolor='white', xaxis_showgrid=True,
        yaxis_showgrid=True, xaxis_gridcolor='lightgray',yaxis_gridcolor='lightgray')
    fig.update_layout(legend=dict(title=dict(text="이동평균")))
    fig.update_layout(legend=dict(x=0.1, y=1.1, tracegroupgap=0.3, xanchor='left', yanchor='top'))
    fig.update_layout(legend=dict(orientation='h'))
    fig.update_layout(yaxis_tickformat=',')
    
    fig.update_layout(xaxis=dict(type='category',
    rangeslider=dict(visible=True),
    rangeselector=dict(
        buttons=list([
            dict(count=1, label='1m', step='month', stepmode='backward'),
            dict(count=6, label='6m', step='month', stepmode='backward'),
            dict(count=1, label='YTD', step='year', stepmode='todate'),
            dict(count=1, label='1y', step='year', stepmode='backward'),
            dict(step='all')])),  
            tickformat='%Y-%m-%d'))
    
    # 요약형 차트 삭제
    fig.update_xaxes(rangeslider_thickness=0.005)
    
    
    # 볼린저 밴드를 차트에 추가
    if bollinger_bands_visible:        
        fig.add_trace(go.Scatter(x=data.index, y=middle_band, mode='lines', name='Middle Band', line=dict(color='#FF00FF', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=upper_band, mode='lines', name='Upper Band', line=dict(color='#800080', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=lower_band, mode='lines', name='Lower Band', line=dict(color='#800080', width=1.5)), row=1, col=1)
    
    # 이동평균곡선을 차트에 추가
    if moving_averages_visible:
        for period, ma in mas.items():
            fig.add_trace(go.Scatter(x=data.index, y=ma, mode='lines', name=f'{period}일 이동평균', line=dict(width=1.8)), row=1, col=1)
   
    # Parabolic Chart를 차트에 추가            
    if parabolic_sar_visible:
        fig.add_trace(go.Scatter(x=data.index, y=sar, mode='markers', name='Parabolic SAR', marker=dict(symbol='circle', size=5, color='black')), row=1, col=1)
     
     # 그물망차트를  차트에 추가
    if rainbow_chart_visible:
        rainbow_chart(data, fig) 
    
    # Envelope 를  차트에 추가
    if envelope_chart_visible:   
        sma, upper_env, lower_env = envelope_chart(data)
        fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='Envelope SMA', line=dict(width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=upper_env, mode='lines', name='Upper Envelope', line=dict(color='#ffa07a', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=lower_env, mode='lines', name='Lower Envelope', line=dict(color='#ffa07a', width=1.5)), row=1, col=1)

    # 일목균형표(ichimoku)를  차트에 추가       
    if ichimoku_visible:
        tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = calculate_ichimoku(data)
        fig.add_trace(go.Scatter(x=data.index, y=tenkan_sen, mode='lines', name='Tenkan-sen', line=dict(color='blue', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=kijun_sen, mode='lines', name='Kijun-sen', line=dict(color='red', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=senkou_span_a, mode='lines', name='Senkou Span A', fill='tonexty', line=dict(color='green', width=1.0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=senkou_span_b, mode='lines', name='Senkou Span B', fill='tonexty', line=dict(color='green', width=1.0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=chikou_span, mode='lines', name='Chikou Span', line=dict(color='magenta', width=1.0)), row=1, col=1)        
     
    # MACD를  차트에 추가(서브플랏에서 추가하기)    
    if macd_chart_visible:
        macd_line, signal_line, macd_hist = macd_chart(data)
        fig.add_trace(go.Scatter(x=data.index, y=macd_line, mode='lines', name='MACD Line', line=dict(color='blue', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=signal_line, mode='lines', name='Signal Line', line=dict(color='orange', width=1.5)), row=3, col=1)
        colors = ['red' if val >= 0 else 'blue' for val in macd_hist]
        fig.add_trace(go.Bar(x=data.index, y=macd_hist, marker_color=colors, name='MACD Histogram'), row=3, col=1)

     # 스토캐스틱 차트를 추가(서브플랏에서 추가하기)        
    if stochastic_visible:
        k_value, d_value = stochastic(data)
        fig.add_trace(go.Scatter(x=data.index, y=k_value, mode='lines', name='%K', line=dict(color='blue', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=d_value, mode='lines', name='%D', line=dict(color='orange', width=1.5)), row=3, col=1)

   # Keltner Channel차트를 캔들차트에 추가 
    if keltner_channels_visible:        
        middle_band, upper_band, lower_band = keltner_channels(data)
        fig.add_trace(go.Scatter(x=data.index, y=middle_band, mode='lines', name='Middle Keltner', line=dict(color='#006400', width=1.5)), row=1, col=1)  # DarkGreen
        fig.add_trace(go.Scatter(x=data.index, y=upper_band, mode='lines', name='Upper Keltner', line=dict(color='#32CD32', width=1.5)), row=1, col=1)  # LimeGreen
        fig.add_trace(go.Scatter(x=data.index, y=lower_band, mode='lines', name='Lower Keltner', line=dict(color='#32CD32', width=1.5)), row=1, col=1)  # LimeGreen

    # DMI차트를 서브플랏으로 추가 
    if dmi_chart_visible:
        pos_di, neg_di, adx = dmi_chart(data)
        fig.add_trace(go.Scatter(x=data.index, y=pos_di, mode='lines', name='+DI', line=dict(color='green', width=1.5)), row=3, col=1) # or whichever subplot you want
        fig.add_trace(go.Scatter(x=data.index, y=neg_di, mode='lines', name='-DI', line=dict(color='red', width=1.5)), row=3, col=1) # or whichever subplot you want
        fig.add_trace(go.Scatter(x=data.index, y=adx, mode='lines', name='ADX', line=dict(color='blue', width=1.5)), row=3, col=1)  # or whichever subplot you want
 
     # 이격도차트 서브플랏으로 추가
    if deviation_rate_visible:
        deviation_rate = deviation_rate_chart(data)
        fig.add_trace(go.Scatter(x=data.index, y=deviation_rate, mode='lines', name='이격도', line=dict(color='blue', width=1.5)), row=3, col=1)

    # RSI 서브플랏으로 추가
    if rsi_visible:
        rsi_values = rsi(data)
        fig.add_trace(go.Scatter(x=data.index, y=rsi_values, mode='lines', name='RSI (투자심리선)', line=dict(color='green', width=1.5)), row=3, col=1)      

    # 삼선전환도 추가
    if three_line_reversal_visible:
        three_line = three_line_reversal(data)
        fig.add_trace(go.Scatter(x=data.index, y=three_line, mode='lines', name='삼선전환도', line=dict(color='brown', width=1.5)), row=1, col=1)
   
    #  AB Ratio차트  추가
    if ab_ratio_visible:
        ab_ratio = calculate_ab_ratio(data)
        fig.add_trace(go.Scatter(x=data.index, y=ab_ratio, mode='lines', name='AB Ratio'), row=3, col=1)  # 서브플랏의 3번째 행에 추가

    # Price Oscillator를 서브플랏에 추가하기
    if price_oscillator_visible:
        po = price_oscillator(data)
        fig.add_trace(go.Scatter(x=data.index, y=po, mode='lines', name='Price Oscillator', line=dict(color='purple', width=1.5)), row=3, col=1) 

    # OBV를 서브플랏에 추가하기
    if obv_visible:
        obv_values = calculate_obv(data)
        fig.add_trace(go.Scatter(x=data.index, y=obv_values, mode='lines', name='OBV', line=dict(color='purple', width=1.5)), row=3, col=1)

    # volumn_oscillator서브플랏에 추가하기
    if volumn_oscillator_visible:
        vo = volume_oscillator(data)
        fig.add_trace(go.Scatter(x=data.index, y=vo, mode='lines', name='Volumn Oscillator', line=dict(width=1.8)), row=3, col=1)  # 3행에 추가

     # Volumn Ratio서브플랏에 추가하기
    if vr_chart_visible:
        vr_values=vr_chart(data)
        fig.add_trace(go.Scatter(x=data.index, y=vr_values, mode='lines', name='VR', line=dict(color='purple', width=1.5)), row=3, col=1)


     # 브라우저에서 차트열기
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    temp_file_name = temp_file.name
    
    fig.write_html(temp_file_name)
    webbrowser.open(temp_file_name)


root = tk.Tk()
root.title("Technical Analysis of KOSPI Stocks")
root.configure(bg="white")

# 양쪽에 대략 1cm 여백
main_frame = tk.Frame(root, bg="white")
main_frame.pack(padx=37.8, pady=37.8)  # 1cm의 여백

# 첫번째 줄: 제목
title_label = tk.Label(main_frame, text="Technical Analysis of KOSPI Stocks", font=("Arial", 24), bg="white")
title_label.grid(row=0, column=0, columnspan=6, pady=15)

# 두번째 줄: 주식코드 입력 & 버튼 추가
entry_label = tk.Entry(main_frame, width=18, font=("Arial", 16), fg='grey')
entry_label.insert(0, '종목코드(숫자입력)')
entry_label.bind('<FocusIn>', on_entry_click)
entry_label.bind('<FocusOut>', on_focusout)
entry_label.grid(row=1, column=0, columnspan=2, padx=20, pady=20)


execute_button = tk.Button(main_frame, text="Execute", height=2, width=10, command=show_stock_description)
execute_button.grid(row=1, column=2, padx=10, pady=10)

open_button = tk.Button(main_frame, text="주식종목 리스트 Down", command=download_excel, height=2, width=20)
open_button.grid(row=1, column=3, columnspan=2, padx=10, pady=10)

# 세번째 줄: 일봉, 주봉, 월봉, Reset버튼
# 콜백 함수 선택
def choose_callback(btn):
    if btn == "Reset":
        return reset_chart_and_entry
    else:
        return lambda: display_chart(btn)

buttons = {"일봉": "#add8e6", "주봉": "#90ee90", "월봉": "#ffb6c1", "Reset": 'lightgrey'}

for idx, (btn, color) in enumerate(buttons.items()):
    tk.Button(main_frame, text=btn, height=2, width=10, bg=color, command=choose_callback(btn)).grid(row=2, column=idx, padx=10, pady=10)
    
# 볼린저 밴드 토글 함수
def toggle_bollinger_bands():
    global bollinger_bands_visible, current_chart_type
    bollinger_bands_visible = not bollinger_bands_visible
    display_chart(current_chart_type)

# 이동평균곡선 토글 함수 추가    
def toggle_moving_averages():
    global moving_averages_visible, current_chart_type
    moving_averages_visible = not moving_averages_visible
    display_chart(current_chart_type) 

# Parabolic SAR 토글 함수 추가
def toggle_parabolic_sar():
    global parabolic_sar_visible, current_chart_type
    parabolic_sar_visible = not parabolic_sar_visible
    display_chart(current_chart_type)

# 그물망차트(Rainbow Chart) 토글 함수 추가   
def toggle_rainbow_chart():
    global rainbow_chart_visible, current_chart_type
    rainbow_chart_visible = not rainbow_chart_visible
    display_chart(current_chart_type)

# Envelope Bands Chart) 토글 함수 추가 
def toggle_envelope_chart():
    global envelope_chart_visible, current_chart_type
    envelope_chart_visible = not envelope_chart_visible
    display_chart(current_chart_type)

# 일목균형표(ichimoku) 토글 함수 추가 
def toggle_ichimoku():
    global ichimoku_visible, current_chart_type
    ichimoku_visible = not ichimoku_visible
    display_chart(current_chart_type)   
    
# MACD 토글 함수 추가    
def toggle_macd_chart():
    global macd_chart_visible, current_chart_type, stochastic_visible
    macd_chart_visible = not macd_chart_visible
    stochastic_visible = False
    dmi_chart_visible = False  
    deviation_rate_visible = False   
    rsi_visible = False    
    three_line_reversal_visible = False   
    ab_ratio_visible = False   
    price_oscillator_visible = False   
    obv_visible_visible = False  
    volumn_oscillator_visible = False  
    vr_chart_visible = False      
    display_chart(current_chart_type)   


# 스토캐스틱 토글 함수 추가   
def toggle_stochastic():
    global stochastic_visible, current_chart_type, macd_chart_visible
    stochastic_visible = not stochastic_visible
    macd_chart_visible = False
    dmi_chart_visible = False  
    deviation_rate_visible = False   
    rsi_visible = False    
    three_line_reversal_visible = False   
    ab_ratio_visible = False   
    price_oscillator_visible = False   
    obv_visible_visible = False  
    volumn_oscillator_visible = False  
    vr_chart_visible = False  
    display_chart(current_chart_type)  
    
# Keltner Channels 토글 함수
def toggle_keltner_channels():
    global keltner_channels_visible, current_chart_type
    keltner_channels_visible = not keltner_channels_visible
    display_chart(current_chart_type)    

# DMI토글 함수
def toggle_dmi_chart():
    global dmi_chart_visible, current_chart_type
    dmi_chart_visible = not dmi_chart_visible
    macd_chart_visible = False
    stochastic_visible = False
    deviation_rate_visible = False   
    rsi_visible = False    
    three_line_reversal_visible = False   
    ab_ratio_visible = False   
    price_oscillator_visible = False   
    obv_visible_visible = False  
    volumn_oscillator_visible = False  
    vr_chart_visible = False  
    display_chart(current_chart_type)   

# 이격도 토글 함수
def toggle_deviation_rate_chart():
    global deviation_rate_visible, current_chart_type
    deviation_rate_visible = not deviation_rate_visible
    macd_chart_visible = False
    stochastic_visible = False
    dmi_chart_visible = False  
    rsi_visible = False    
    three_line_reversal_visible = False   
    ab_ratio_visible = False   
    price_oscillator_visible = False   
    obv_visible_visible = False  
    volumn_oscillator_visible = False  
    vr_chart_visible = False  
    display_chart(current_chart_type)        

# RSI토글 함수    
def toggle_rsi_chart():
    global rsi_visible, current_chart_type
    rsi_visible = not rsi_visible    
    macd_chart_visible = False
    stochastic_visible = False
    dmi_chart_visible = False  
    deviation_rate_visible = False   
    three_line_reversal_visible = False   
    ab_ratio_visible = False   
    price_oscillator_visible = False   
    obv_visible_visible = False  
    volumn_oscillator_visible = False  
    vr_chart_visible = False  
    display_chart(current_chart_type)        

# 삼선전환도  토글함수  
def toggle_three_line_reversal():
    global three_line_reversal_visible, current_chart_type
    three_line_reversal_visible = not three_line_reversal_visible
    macd_chart_visible = False
    stochastic_visible = False
    dmi_chart_visible = False  
    deviation_rate_visible = False   
    rsi_visible = False    
    ab_ratio_visible = False   
    price_oscillator_visible = False   
    obv_visible_visible = False  
    volumn_oscillator_visible = False  
    vr_chart_visible = False  
    display_chart(current_chart_type)
    
#  AB Ratio차트 토글함수      
def toggle_ab_ratio():
    global ab_ratio_visible, current_chart_type
    ab_ratio_visible = not ab_ratio_visible
    macd_chart_visible = False
    stochastic_visible = False
    dmi_chart_visible = False  
    deviation_rate_visible = False   
    rsi_visible = False    
    three_line_reversal_visible = False   
    price_oscillator_visible = False   
    obv_visible_visible = False  
    volumn_oscillator_visible = False  
    vr_chart_visible = False  
    display_chart(current_chart_type)

# Price Oscillator 토글함수 
def toggle_price_oscillator():
    global price_oscillator_visible, current_chart_type
    price_oscillator_visible = not price_oscillator_visible
    macd_chart_visible = False
    stochastic_visible = False
    dmi_chart_visible = False  
    deviation_rate_visible = False   
    rsi_visible = False    
    three_line_reversal_visible = False   
    ab_ratio_visible = False   
    obv_visible_visible = False  
    volumn_oscillator_visible = False  
    vr_chart_visible = False   
    deviation_rate_visible =False  
    display_chart(current_chart_type)
    
# OBV 토글함수 
def toggle_obv():
    global obv_visible, current_chart_type
    obv_visible = not obv_visible
    macd_chart_visible = False
    stochastic_visible = False
    dmi_chart_visible = False  
    deviation_rate_visible = False   
    rsi_visible = False    
    three_line_reversal_visible = False   
    ab_ratio_visible = False   
    price_oscillator_visible = False   
    volumn_oscillator_visible = False  
    vr_chart_visible = False  
    display_chart(current_chart_type)
    
# Volumn Oscillator 토글 버튼 추가 함수
def toggle_volumn_oscillator():
    global volumn_oscillator_visible, current_chart_type
    volumn_oscillator_visible = not volumn_oscillator_visible
    macd_chart_visible = False
    stochastic_visible = False
    dmi_chart_visible = False  
    deviation_rate_visible = False   
    rsi_visible = False    
    three_line_reversal_visible = False   
    ab_ratio_visible = False   
    price_oscillator_visible = False   
    obv_visible_visible = False  
    vr_chart_visible = False  
    display_chart(current_chart_type)    
    
# VR 차트 토글 함수
def toggle_vr_chart():
    global vr_chart_visible, current_chart_type
    vr_chart_visible = not vr_chart_visible
    macd_chart_visible = False
    stochastic_visible = False
    dmi_chart_visible = False  
    deviation_rate_visible = False   
    rsi_visible = False    
    three_line_reversal_visible = False   
    ab_ratio_visible = False   
    price_oscillator_visible = False   
    obv_visible_visible = False  
    volumn_oscillator_visible = False  
    display_chart(current_chart_type)


    
# 지표 별 버튼 생성
indicators = {
    "추세지표": ["이동평균선","MACD", "스토캐스틱", "DMI", "Parabolic Sar", "그물망차트","일목균형표"],
    "변동성지표": ["볼린저벤트", "Envelope", "Keltner Channels"],
    "모멘텀지표": ["이격도", "RSI", "삼선전환도", "AB Ratio", "Price Oscillator",],
    "시장강도지표": ["OBV", "Volumn Oscillator", "VR(Volum Ratio)"]}

row_num = 3
for category, items in indicators.items():
    tk.Label(main_frame, text=category, font=("Arial", 16, "bold"), bg="white").grid(row=row_num, column=0, columnspan=4, pady=10, sticky="w")
    row_num += 1
    col_num = 0
    for item in items:
        tk.Button(main_frame, text=item, height=2, width=15).grid(row=row_num, column=col_num, padx=5, pady=5)
        col_num += 1
        if col_num == 4:
            col_num = 0
            row_num += 1
    row_num += 1

    
# 볼린저 밴드 버튼에 함수를 연결
bollinger_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == '볼린저벤트'][0]
bollinger_button.config(command=toggle_bollinger_bands)

# 이동평균곤선  버튼에 함수를 연결
moving_average_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == '이동평균선'][0]
moving_average_button.config(command=toggle_moving_averages)

# Parabolic Sar버튼에 함수를 연결
parabolic_sar_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'Parabolic Sar'][0]
parabolic_sar_button.config(command=toggle_parabolic_sar)

# 그물망차트(rainbow chart)버튼에 함수를 연결
rainbow_chart_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == '그물망차트'][0]
rainbow_chart_button.config(command=toggle_rainbow_chart)

# Envelope Chart버튼에 함수를 연결
envelope_chart_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'Envelope'][0]
envelope_chart_button.config(command=toggle_envelope_chart)


# 일목균형표 버튼에 함수연결
ichimoku_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == '일목균형표'][0]
ichimoku_button.config(command=toggle_ichimoku)

# MACD 버튼에 함수연결
macd_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'MACD'][0]
macd_button.config(command=toggle_macd_chart)

# 스토캐스틱 버튼에 함수연결
stochastic_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == '스토캐스틱'][0]
stochastic_button.config(command=toggle_stochastic)

# 캘트너 채널 버튼에 함수연결
keltner_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'Keltner Channels'][0]
keltner_button.config(command=toggle_keltner_channels)


# DMI 버튼에 함수연결
dmi_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'DMI'][0]
dmi_button.config(command=toggle_dmi_chart)

# 이격도 버튼에 함수연결
deviation_rate_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == '이격도'][0]
deviation_rate_button.config(command=toggle_deviation_rate_chart)

# RSI 버튼에 함수연결
rsi_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'RSI'][0]
rsi_button.config(command=toggle_rsi_chart)

# 삼선전환도 버튼에 함수 연결
three_line_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == '삼선전환도'][0]
three_line_button.config(command=toggle_three_line_reversal)

# AB Ratio 버튼에 함수 연결
ab_ratio_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'AB Ratio'][0]
ab_ratio_button.config(command=toggle_ab_ratio)

# Price Oscillator 버튼에 함수를 연결
price_oscillator_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'Price Oscillator'][0]
price_oscillator_button.config(command=toggle_price_oscillator)

# OBV버튼에 함수를 연결
obv_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'OBV'][0]
obv_button.config(command=toggle_obv)


# VolumB Ocillator버튼에 함수를 연결
vol_osci_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'Volumn Oscillator'][0]
vol_osci_button.config(command=toggle_volumn_oscillator)

# Volumn Ratio버튼에 함수를 연결
vr_button = [child for child in main_frame.winfo_children() if isinstance(child, tk.Button) and child['text'] == 'VR(Volum Ratio)'][0]
vr_button.config(command=toggle_vr_chart)


# 하단의 entry 창   
text_widget = tk.Text(main_frame, height=11, font=("Arial", 11,'bold'))
text_widget.grid(row=row_num, column=0, columnspan=6, padx=10, pady=10, ipadx=5, ipady=5)

root.mainloop()


