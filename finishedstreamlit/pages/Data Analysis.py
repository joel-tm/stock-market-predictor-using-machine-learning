import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.metrics import mean_squared_error

st.title("Data Analysis")
# Function to create dataset for LSTM
def create_dataset(dataset, target, look_back=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        data_X.append(a)
        data_Y.append(target[i + look_back])
    return np.array(data_X), np.array(data_Y)

    # Function to load and preprocess the data
def load_and_preprocess_data(df, look_back=3):
    # Split features and target
    features = df.drop(['Date', 'Open'], axis=1).values
    target = df['Open'].values

    # Scale the features and target
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler_features.fit_transform(features)
    target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))

    # Create the dataset for LSTM
    X, y = create_dataset(features_scaled, target_scaled, look_back)

    # Split into train and test sets
    train_size = int(len(X) * 0.95)
    test_size = len(X) - train_size
    train_X, test_X = X[0:train_size, :], X[train_size:len(X), :]
    train_Y, test_Y = y[0:train_size], y[train_size:len(y)]

    # Reshape the input to be 3D [samples, timesteps, features]
    train_X = np.reshape(train_X, (train_X.shape[0], look_back, train_X.shape[2]))
    test_X = np.reshape(test_X, (test_X.shape[0], look_back, test_X.shape[2]))

    return train_X, test_X, train_Y, test_Y, scaler_target

# Load HDFC data
hdfc_df = pd.read_csv('hdfc_merged.csv')
train_X_hdfc, test_X_hdfc, train_Y_hdfc, test_Y_hdfc, scaler_target_hdfc = load_and_preprocess_data(hdfc_df,look_back=7)

# Load Reliance data
reliance_df = pd.read_csv('reliance_merged.csv')
train_X_reliance, test_X_reliance, train_Y_reliance, test_Y_reliance, scaler_target_reliance = load_and_preprocess_data(reliance_df, look_back=5)

# Load TCS data
tcs_df = pd.read_csv('tcs_merged.csv')
train_X_tcs, test_X_tcs, train_Y_tcs, test_Y_tcs, scaler_target_tcs = load_and_preprocess_data(tcs_df)

# Load Wipro data
wipro_df = pd.read_csv('wipro_merged.csv')
train_X_wipro, test_X_wipro, train_Y_wipro, test_Y_wipro, scaler_target_wipro = load_and_preprocess_data(wipro_df)

# Load Infosys data
infosys_df = pd.read_csv('infosys_merged.csv')
train_X_infosys, test_X_infosys, train_Y_infosys, test_Y_infosys, scaler_target_infosys = load_and_preprocess_data(infosys_df, look_back=5)

# Streamlit UI
st.title('Stock Market Analysis')

# Dropdown to select company
company = st.selectbox('Select Company', ['HDFC', 'Reliance', 'TCS', 'Wipro', 'Infosys'])

if company == 'HDFC':
    st.subheader('HDFC Stock Analysis')
    model_hdfc = load_model('hdfclstm_model.h5')
    train_predict_hdfc = model_hdfc.predict(train_X_hdfc)
    train_predict_inv_hdfc = scaler_target_hdfc.inverse_transform(train_predict_hdfc)
    train_Y_inv_hdfc= scaler_target_hdfc.inverse_transform(np.reshape(train_Y_hdfc, (train_Y_hdfc.shape[0], 1)))
    test_predict_hdfc = model_hdfc.predict(test_X_hdfc)
    test_predict_inv_hdfc = scaler_target_hdfc.inverse_transform(test_predict_hdfc)
    test_Y_inv_hdfc = scaler_target_hdfc.inverse_transform(np.reshape(test_Y_hdfc, (test_Y_hdfc.shape[0], 1)))
    # Calculate RMSE and accuracy for each company
    trainScore_hdfc = np.sqrt(mean_squared_error(train_Y_inv_hdfc[:, 0], train_predict_inv_hdfc[:, 0]))
    testScore_hdfc = np.sqrt(mean_squared_error(test_Y_inv_hdfc[:, 0], test_predict_inv_hdfc[:, 0]))
    trainAccuracy_hdfc = 100 - (trainScore_hdfc / np.mean(train_Y_inv_hdfc) * 100)
    testAccuracy_hdfc = 100 - (testScore_hdfc / np.mean(test_Y_inv_hdfc) * 100)

        
    fig_hdfc = go.Figure()
    fig_hdfc.add_trace(go.Scatter(x=hdfc_df['Date'][-len(test_Y_hdfc):], y=hdfc_df['Open'][-len(test_Y_hdfc):], mode='lines', name='Actual Open Price', line=dict(color='green')))
    fig_hdfc.add_trace(go.Scatter(x=hdfc_df['Date'][-len(test_Y_hdfc):], y=test_predict_inv_hdfc.flatten(), mode='lines', name='Predicted Open Price', line=dict(color='red')))
    fig_hdfc.update_layout(title='Comparison between actual open price and predicted open price for HDFC', xaxis_title='Date', yaxis_title='Open Price')
    st.plotly_chart(fig_hdfc)
    testyinvflat_hdfc = test_Y_inv_hdfc.flatten()
    start_date_hdfc = '2023-06-13'
    df_filtered_hdfc = hdfc_df[hdfc_df['Date'] >= start_date_hdfc]
    dates_col_hdfc = df_filtered_hdfc['Date'].values
    df_result_hdfc = pd.DataFrame({'dates': dates_col_hdfc, '  Actual_Open_price': testyinvflat_hdfc, '   Predicted_Open_price': test_predict_inv_hdfc.flatten()})
    st.subheader('Actual vs Predicted Open Price for HDFC:')
    st.dataframe(df_result_hdfc)


    st.subheader('Model Evaluation Metrics')
    st.write('**HDFC:**')
    st.write(f'Training RMSE: {trainScore_hdfc}')
    st.write(f'Testing RMSE: {testScore_hdfc}')
    st.write(f'Training Accuracy: {trainAccuracy_hdfc:.2f}%')
    st.write(f'Testing Accuracy: {testAccuracy_hdfc:.2f}%')

    # Plotting matplotlib graph for HDFC
    dates_hdfc = hdfc_df['Date'].values
    sampleInterval_hdfc = 60
    sampledDates_hdfc = dates_hdfc[::sampleInterval_hdfc]

    plt.figure(figsize=(20, 10))
    plt.plot(dates_hdfc[:len(train_Y_hdfc)], train_Y_hdfc, label='Actual Train')
    plt.plot(dates_hdfc[:len(train_predict_hdfc)], train_predict_hdfc, label='Predicted Train')
    plt.plot(dates_hdfc[len(train_Y_hdfc):len(train_Y_hdfc) + len(test_Y_hdfc)], test_Y_hdfc, label='Actual Test')
    plt.plot(dates_hdfc[len(train_predict_hdfc):len(train_predict_hdfc) + len(test_predict_hdfc)], test_predict_hdfc, label='Predicted Test')

    plt.xticks(sampledDates_hdfc, rotation=90)

    plt.legend()
    plt.title('Comparison between actual and predicted open price for HDFC')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.tight_layout()

    # Convert matplotlib plot to PNG image for HDFC
    plt.savefig('hdfc_plot.png')

    # Display PNG image for HDFC
    st.subheader('Data Visualization for HDFC:')
    st.image('hdfc_plot.png', caption='Comparison between actual and predicted open price for HDFC')

elif company == 'Reliance':
    st.subheader('Reliance Stock Analysis')
    model_reliance = load_model('reliancelstm_model.h5')
    train_predict_reliance = model_reliance.predict(train_X_reliance)
    train_predict_inv_reliance = scaler_target_reliance.inverse_transform(train_predict_reliance)
    train_Y_inv_reliance = scaler_target_reliance.inverse_transform(np.reshape(train_Y_reliance, (train_Y_reliance.shape[0], 1)))
    test_predict_reliance = model_reliance.predict(test_X_reliance)
    test_predict_inv_reliance = scaler_target_reliance.inverse_transform(test_predict_reliance)
    test_Y_inv_reliance = scaler_target_reliance.inverse_transform(np.reshape(test_Y_reliance, (test_Y_reliance.shape[0], 1)))
    fig_reliance = go.Figure()
    fig_reliance.add_trace(go.Scatter(x=reliance_df['Date'][-len(test_Y_reliance):], y=reliance_df['Open'][-len(test_Y_reliance):], mode='lines', name='Actual Open Price', line=dict(color='green')))
    fig_reliance.add_trace(go.Scatter(x=reliance_df['Date'][-len(test_Y_reliance):], y=test_predict_inv_reliance.flatten(), mode='lines', name='Predicted Open Price', line=dict(color='red')))
    fig_reliance.update_layout(title='Comparison between actual open price and predicted open price for Reliance', xaxis_title='Date', yaxis_title='Open Price')
    st.plotly_chart(fig_reliance)
    testyinvflat_reliance = test_Y_inv_reliance.flatten()
    start_date_reliance = '2023-06-13'
    df_filtered_reliance = reliance_df[reliance_df['Date'] >= start_date_reliance]
    dates_col_reliance = df_filtered_reliance['Date'].values
    df_result_reliance = pd.DataFrame({'dates': dates_col_reliance, '  Actual_Open_price': testyinvflat_reliance, '   Predicted_Open_price': test_predict_inv_reliance.flatten()})
    st.subheader('Actual vs Predicted Open Price for Reliance:')
    st.dataframe(df_result_reliance)

    # Calculate RMSE and accuracy for each company
    trainScore_reliance = np.sqrt(mean_squared_error(train_Y_inv_reliance[:, 0], train_predict_inv_reliance[:, 0]))
    testScore_reliance = np.sqrt(mean_squared_error(test_Y_inv_reliance[:, 0], test_predict_inv_reliance[:, 0]))
    trainAccuracy_reliance = 100 - (trainScore_reliance / np.mean(train_Y_inv_reliance) * 100)
    testAccuracy_reliance = 100 - (testScore_reliance / np.mean(test_Y_inv_reliance) * 100)
        
    st.subheader('Model Evaluation Metrics')
    st.write('**Reliance:**')
    st.write(f'Training RMSE: {trainScore_reliance}')
    st.write(f'Testing RMSE: {testScore_reliance}')
    st.write(f'Training Accuracy: {trainAccuracy_reliance:.2f}%')
    st.write(f'Testing Accuracy: {testAccuracy_reliance:.2f}%')

    # Plotting matplotlib graph for Reliance
    dates_reliance = reliance_df['Date'].values
    sampleInterval_reliance = 60
    sampledDates_reliance = dates_reliance[::sampleInterval_reliance]

    plt.figure(figsize=(20, 10))
    plt.plot(dates_reliance[:len(train_Y_reliance)], train_Y_reliance, label='Actual Train')
    plt.plot(dates_reliance[:len(train_predict_reliance)], train_predict_reliance, label='Predicted Train')
    plt.plot(dates_reliance[len(train_Y_reliance):len(train_Y_reliance) + len(test_Y_reliance)], test_Y_reliance, label='Actual Test')
    plt.plot(dates_reliance[len(train_predict_reliance):len(train_predict_reliance) + len(test_predict_reliance)], test_predict_reliance, label='Predicted Test')

    plt.xticks(sampledDates_reliance, rotation=90)

    plt.legend()
    plt.title('Comparison between actual and predicted open price for Reliance')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.tight_layout()

    # Convert matplotlib plot to PNG image for Reliance
    plt.savefig('reliance_plot.png')

    # Display PNG image for Reliance
    st.subheader('Data Visualization for Reliance:')
    st.image('reliance_plot.png', caption='Comparison between actual and predicted open price for Reliance')

elif company == 'TCS':
    st.subheader('TCS Stock Analysis')
    model_tcs = load_model('tcslstm_model.h5')
    train_predict_tcs = model_tcs.predict(train_X_tcs)
    train_predict_inv_tcs = scaler_target_tcs.inverse_transform(train_predict_tcs)
    train_Y_inv_tcs = scaler_target_tcs.inverse_transform(np.reshape(train_Y_tcs, (train_Y_tcs.shape[0], 1)))
    test_predict_tcs = model_tcs.predict(test_X_tcs)
    test_predict_inv_tcs = scaler_target_tcs.inverse_transform(test_predict_tcs)
    test_Y_inv_tcs = scaler_target_tcs.inverse_transform(np.reshape(test_Y_tcs, (test_Y_tcs.shape[0], 1)))
    fig_tcs = go.Figure()
    fig_tcs.add_trace(go.Scatter(x=tcs_df['Date'][-len(test_Y_tcs):], y=tcs_df['Open'][-len(test_Y_tcs):], mode='lines', name='Actual Open Price', line=dict(color='green')))
    fig_tcs.add_trace(go.Scatter(x=tcs_df['Date'][-len(test_Y_tcs):], y=test_predict_inv_tcs.flatten(), mode='lines', name='Predicted Open Price', line=dict(color='red')))
    fig_tcs.update_layout(title='Comparison between actual open price and predicted open price for TCS', xaxis_title='Date', yaxis_title='Open Price')
    st.plotly_chart(fig_tcs)
    testyinvflat_tcs = test_Y_inv_tcs.flatten()
    start_date_tcs = '2023-06-13'
    df_filtered_tcs = tcs_df[tcs_df['Date'] >= start_date_tcs]
    dates_col_tcs = df_filtered_tcs['Date'].values
    df_result_tcs = pd.DataFrame({'dates': dates_col_tcs, '  Actual_Open_price': testyinvflat_tcs, '   Predicted_Open_price': test_predict_inv_tcs.flatten()})
    st.subheader('Actual vs Predicted Open Price for TCS:')
    st.dataframe(df_result_tcs)

    # For TCS
    trainScore_tcs = np.sqrt(mean_squared_error(train_Y_inv_tcs[:, 0], train_predict_inv_tcs[:, 0]))
    testScore_tcs = np.sqrt(mean_squared_error(test_Y_inv_tcs[:, 0], test_predict_inv_tcs[:, 0]))
    trainAccuracy_tcs = 100 - (trainScore_tcs / np.mean(train_Y_inv_tcs) * 100)
    testAccuracy_tcs = 100 - (testScore_tcs / np.mean(test_Y_inv_tcs) * 100)
    
    st.subheader('Model Evaluation Metrics')
    st.write('**TCS:**')
    st.write(f'Training RMSE: {trainScore_tcs}')
    st.write(f'Testing RMSE: {testScore_tcs}')
    st.write(f'Training Accuracy: {trainAccuracy_tcs:.2f}%')
    st.write(f'Testing Accuracy: {testAccuracy_tcs:.2f}%')
    # Plotting matplotlib graph for TCS
    dates_tcs = tcs_df['Date'].values
    sampleInterval_tcs = 60
    sampledDates_tcs = dates_tcs[::sampleInterval_tcs]

    plt.figure(figsize=(20, 10))
    plt.plot(dates_tcs[:len(train_Y_tcs)], train_Y_tcs, label='Actual Train')
    plt.plot(dates_tcs[:len(train_predict_tcs)], train_predict_tcs, label='Predicted Train')
    plt.plot(dates_tcs[len(train_Y_tcs):len(train_Y_tcs) + len(test_Y_tcs)], test_Y_tcs, label='Actual Test')
    plt.plot(dates_tcs[len(train_predict_tcs):len(train_predict_tcs) + len(test_predict_tcs)], test_predict_tcs, label='Predicted Test')

    plt.xticks(sampledDates_tcs, rotation=90)

    plt.legend()
    plt.title('Comparison between actual and predicted open price for TCS')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.tight_layout()

    # Convert matplotlib plot to PNG image for TCS
    plt.savefig('tcs_plot.png')

    # Display PNG image for TCS
    st.subheader('Data Visualization for TCS:')
    st.image('tcs_plot.png', caption='Comparison between actual and predicted open price for TCS')
    
elif company == 'Wipro':
    st.subheader('Wipro Stock Analysis')
    model_wipro = load_model('wiprolstm_model.h5')
    train_predict_wipro = model_wipro.predict(train_X_wipro)
    train_predict_inv_wipro = scaler_target_wipro.inverse_transform(train_predict_wipro)
    train_Y_inv_wipro = scaler_target_wipro.inverse_transform(np.reshape(train_Y_wipro, (train_Y_wipro.shape[0], 1)))
    test_predict_wipro = model_wipro.predict(test_X_wipro)
    test_predict_inv_wipro = scaler_target_wipro.inverse_transform(test_predict_wipro)
    test_Y_inv_wipro = scaler_target_wipro.inverse_transform(np.reshape(test_Y_wipro, (test_Y_wipro.shape[0], 1)))
    fig_wipro = go.Figure()
    fig_wipro.add_trace(go.Scatter(x=wipro_df['Date'][-len(test_Y_wipro):], y=wipro_df['Open'][-len(test_Y_wipro):], mode='lines', name='Actual Open Price', line=dict(color='green')))
    fig_wipro.add_trace(go.Scatter(x=wipro_df['Date'][-len(test_Y_wipro):], y=test_predict_inv_wipro.flatten(), mode='lines', name='Predicted Open Price', line=dict(color='red')))
    fig_wipro.update_layout(title='Comparison between actual open price and predicted open price for Wipro', xaxis_title='Date', yaxis_title='Open Price')
    st.plotly_chart(fig_wipro)
    testyinvflat_wipro = test_Y_inv_wipro.flatten()
    start_date_wipro = '2023-06-13'
    df_filtered_wipro = wipro_df[wipro_df['Date'] >= start_date_wipro]
    dates_col_wipro = df_filtered_wipro['Date'].values
    df_result_wipro = pd.DataFrame({'dates': dates_col_wipro, '  Actual_Open_price': testyinvflat_wipro, '   Predicted_Open_price': test_predict_inv_wipro.flatten()})
    st.subheader('Actual vs Predicted Open Price for Wipro:')
    st.dataframe(df_result_wipro)

    # For Wipro
    trainScore_wipro = np.sqrt(mean_squared_error(train_Y_inv_wipro[:, 0], train_predict_inv_wipro[:, 0]))
    testScore_wipro = np.sqrt(mean_squared_error(test_Y_inv_wipro[:, 0], test_predict_inv_wipro[:, 0]))
    trainAccuracy_wipro = 100 - (trainScore_wipro / np.mean(train_Y_inv_wipro) * 100)
    testAccuracy_wipro = 100 - (testScore_wipro / np.mean(test_Y_inv_wipro) * 100)
    
    st.subheader('Model Evaluation Metrics')
    # For Wipro
    st.write('**Wipro:**')
    st.write(f'Training RMSE: {trainScore_wipro}')
    st.write(f'Testing RMSE: {testScore_wipro}')
    st.write(f'Training Accuracy: {trainAccuracy_wipro:.2f}%')
    st.write(f'Testing Accuracy: {testAccuracy_wipro:.2f}%')

    # Plotting matplotlib graph for Wipro
    dates_wipro = wipro_df['Date'].values
    sampleInterval_wipro = 60
    sampledDates_wipro = dates_wipro[::sampleInterval_wipro]

    plt.figure(figsize=(20, 10))
    plt.plot(dates_wipro[:len(train_Y_wipro)], train_Y_wipro, label='Actual Train')
    plt.plot(dates_wipro[:len(train_predict_wipro)], train_predict_wipro, label='Predicted Train')
    plt.plot(dates_wipro[len(train_Y_wipro):len(train_Y_wipro) + len(test_Y_wipro)], test_Y_wipro, label='Actual Test')
    plt.plot(dates_wipro[len(train_predict_wipro):len(train_predict_wipro) + len(test_predict_wipro)], test_predict_wipro, label='Predicted Test')

    plt.xticks(sampledDates_wipro, rotation=90)

    plt.legend()
    plt.title('Comparison between actual and predicted open price for Wipro')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.tight_layout()

    # Convert matplotlib plot to PNG image for Wipro
    plt.savefig('wipro_plot.png')

    # Display PNG image for Wipro
    st.subheader('Data Visualization for Wipro:')
    st.image('wipro_plot.png', caption='Comparison between actual and predicted open price for Wipro')
    
elif company == 'Infosys':
    st.subheader('Infosys Stock Analysis')
    model_infosys = load_model('infosyslstm_model.h5')
    train_predict_infosys = model_infosys.predict(train_X_infosys)
    train_predict_inv_infosys = scaler_target_infosys.inverse_transform(train_predict_infosys)
    train_Y_inv_infosys = scaler_target_infosys.inverse_transform(np.reshape(train_Y_infosys, (train_Y_infosys.shape[0], 1)))
    test_predict_infosys = model_infosys.predict(test_X_infosys)
    test_predict_inv_infosys = scaler_target_infosys.inverse_transform(test_predict_infosys)
    test_Y_inv_infosys = scaler_target_infosys.inverse_transform(np.reshape(test_Y_infosys, (test_Y_infosys.shape[0], 1)))
    fig_infosys = go.Figure()
    fig_infosys.add_trace(go.Scatter(x=infosys_df['Date'][-len(test_Y_infosys):], y=infosys_df['Open'][-len(test_Y_infosys):], mode='lines', name='Actual Open Price', line=dict(color='green')))
    fig_infosys.add_trace(go.Scatter(x=infosys_df['Date'][-len(test_Y_infosys):], y=test_predict_inv_infosys.flatten(), mode='lines', name='Predicted Open Price', line=dict(color='red')))
    fig_infosys.update_layout(title='Comparison between actual open price and predicted open price for Infosys', xaxis_title='Date', yaxis_title='Open Price')
    st.plotly_chart(fig_infosys)
    testyinvflat_infosys = test_Y_inv_infosys.flatten()
    start_date_infosys = '2023-06-13'
    df_filtered_infosys = infosys_df[infosys_df['Date'] >= start_date_infosys]
    dates_col_infosys = df_filtered_infosys['Date'].values
    df_result_infosys = pd.DataFrame({'dates': dates_col_infosys, '  Actual_Open_price': testyinvflat_infosys, '   Predicted_Open_price': test_predict_inv_infosys.flatten()})
    st.subheader('Actual vs Predicted Open Price for Infosys:')
    st.dataframe(df_result_infosys)

    trainScore_infosys = np.sqrt(mean_squared_error(train_Y_inv_infosys[:, 0], train_predict_inv_infosys[:, 0]))
    testScore_infosys = np.sqrt(mean_squared_error(test_Y_inv_infosys[:, 0], test_predict_inv_infosys[:, 0]))
    trainAccuracy_infosys = 100 - (trainScore_infosys / np.mean(train_Y_inv_infosys) * 100)
    testAccuracy_infosys = 100 - (testScore_infosys / np.mean(test_Y_inv_infosys) * 100)
    
    # For Infosys
    st.subheader('Model Evaluation Metrics')    
    st.write('**Infosys:**')
    st.write(f'Training RMSE: {trainScore_infosys}')
    st.write(f'Testing RMSE: {testScore_infosys}')
    st.write(f'Training Accuracy: {trainAccuracy_infosys:.2f}%')
    st.write(f'Testing Accuracy: {testAccuracy_infosys:.2f}%')
    # Plotting matplotlib graph for Infosys
    dates_infosys = infosys_df['Date'].values
    sampleInterval_infosys = 60
    sampledDates_infosys = dates_infosys[::sampleInterval_infosys]

    plt.figure(figsize=(20, 10))
    plt.plot(dates_infosys[:len(train_Y_infosys)], train_Y_infosys, label='Actual Train')
    plt.plot(dates_infosys[:len(train_predict_infosys)], train_predict_infosys, label='Predicted Train')
    plt.plot(dates_infosys[len(train_Y_infosys):len(train_Y_infosys) + len(test_Y_infosys)], test_Y_infosys, label='Actual Test')
    plt.plot(dates_infosys[len(train_predict_infosys):len(train_predict_infosys) + len(test_predict_infosys)], test_predict_infosys, label='Predicted Test')

    plt.xticks(sampledDates_infosys, rotation=90)

    plt.legend()
    plt.title('Comparison between actual and predicted open price for Infosys')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.tight_layout()

    # Convert matplotlib plot to PNG image for Infosys
    plt.savefig('infosys_plot.png')

    # Display PNG image for Infosys
    st.subheader('Data Visualization for Infosys:')
    st.image('infosys_plot.png', caption='Comparison between actual and predicted open price for Infosys')