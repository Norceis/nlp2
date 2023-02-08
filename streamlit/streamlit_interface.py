import time

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selection = option_menu(None, ['Data exploration',
                                   'Data analysis from local model',
                                   'Analysis of pre-generated data'],
                            icons=['layers-half', 'hypnotize', 'journal'], default_index=0)

if selection == 'Data exploration':
    pass

elif selection == 'Data analysis from local model':
    # if model exists then:
    pass

elif selection == 'Analysis of pre-generated data':
    which_metrics = st.multiselect(
        'Metrics for which model do you want to see?',
        ['Days + name in 5 epochs',
         'Days + name in 20 epochs',
         'Days + name + part of description in 5 epochs',
         'Days + name + part of description in 20 epochs'])

    which_text = st.multiselect(
        'Predictions for which prompts do you want to see?',
        ['iphone 11 64, 128, 256gb',
         'iphone 11 pro 64, 256, 512gb',
         'iphone 11 pro max 64, 256, 512gb'])

    col1_1, col2_1, col3_1 = st.columns(3)
    if col2_1.button('Engage visualizations'):
        with st.spinner('Loading data in progress'):
            days_name_5_metrics = pd.read_csv('metrics/days_name_5_metrics.csv', index_col=0)
            days_name_20_metrics = pd.read_csv('metrics/days_name_20_metrics.csv', index_col=0)
            days_name_desc_5_ml100_metrics = pd.read_csv('metrics/days_name_desc_5_ml100_metrics.csv', index_col=0)
            days_name_desc_20_ml100_metrics = pd.read_csv('metrics/days_name_desc_20_ml100_metrics.csv', index_col=0)
            predictions = pd.read_csv('predictions/days_name_5_predictions.csv', index_col=0)
            time.sleep(1)
        st.success('Loading data done!')

        if 'Days + name in 5 epochs' in which_metrics:
            fig = px.line(days_name_5_metrics[['MSE', 'MAE', 'RMSE']]).update_layout(xaxis_title='Epoch',
                                                                                     yaxis_title='Metric value',
                                                                                     title='Metrics for model trained on '
                                                                                           'days + name string for 5 epochs',
                                                                                     title_x=1)
            st.plotly_chart(fig)

        if 'Days + name in 20 epochs' in which_metrics:
            fig = px.line(days_name_20_metrics[['MSE', 'MAE', 'RMSE']]).update_layout(xaxis_title='Epoch',
                                                                                      yaxis_title='Metric value',
                                                                                      title='Metrics for model trained on days + name string for 20 epochs',
                                                                                      title_x=0.5)
            st.plotly_chart(fig)

        if 'Days + name + part of description in 5 epochs' in which_metrics:
            fig = px.line(days_name_desc_5_ml100_metrics[['MSE', 'MAE', 'RMSE']]).update_layout(xaxis_title='Epoch',
                                                                                                yaxis_title='Metric value',
                                                                                                title='Metrics for model trained on '
                                                                                                      'days + name + description '
                                                                                                      'string for 5 epochs',
                                                                                                title_x=0.5)
            st.plotly_chart(fig)

        if 'Days + name + part of description in 20 epochs' in which_metrics:
            fig = px.line(days_name_desc_20_ml100_metrics[['MSE', 'MAE', 'RMSE']]).update_layout(xaxis_title='Epoch',
                                                                                                 yaxis_title='Metric value',
                                                                                                 title='Metrics for model trained on days + name + description string for 20 epochs',
                                                                                                 title_x=0.5)
            st.plotly_chart(fig)

        st.markdown('''<div style="text-align: justify;">  To summarize metrics values changing over time - it is 
        seen that the model is trying to learn, but not a lot of advance is made in this direction. In 2 models 
        during 20 epochs the model did not decrease MSE, MAE or RMSE, though it is possible that my method of inverse 
        scaling it after getting it out of the model may be faulty (the value of RMSE should be root of MSE, 
        but it is not). Quick mean square error calculation at the end of 04_auto_model.ipynb, done on test set vs 
        model prediction outputted MSE ~400, which was more believable than the values above. These error 
        calculations done by the model itself needs further investigation. .</div>''', unsafe_allow_html=True)


        if 'iphone 11 64, 128, 256gb' in which_text:
            fig = px.scatter(predictions[['iphone 11 64gb',
                                          'iphone 11 128gb',
                                          'iphone 11 256gb']], trendline='lowess',
                             title='Changes in price in relation to days passed since 01.01.2021').update_layout(
                xaxis_title='Days passed', yaxis_title='Price', title_x=0.5)
            fig.add_vline(55, line_color='white', annotation_text='Last day in dataset')
            st.plotly_chart(fig)

        if 'iphone 11 pro 64, 256, 512gb' in which_text:
            fig = px.scatter(predictions[['iphone 11 pro 64gb',
                                            'iphone 11 pro 256gb',
                                            'iphone 11 pro 512gb']], trendline='lowess',
                             title='Changes in price in relation to days passed since 01.01.2021').update_layout(
                xaxis_title='Days passed', yaxis_title='Price', title_x=0.5)
            fig.add_vline(55, line_color='white', annotation_text='Last day in dataset')
            st.plotly_chart(fig)

        if 'iphone 11 pro max 64, 256, 512gb' in which_text:
            fig = px.scatter(predictions[['iphone 11 pro max 64gb',
                                            'iphone 11 pro max 256gb',
                                            'iphone 11 pro max 512gb']], trendline='lowess',
                             title='Changes in price in relation to days passed since 01.01.2021').update_layout(
                xaxis_title='Days passed', yaxis_title='Price', title_x=0.5)
            fig.add_vline(55, line_color='white', annotation_text='Last day in dataset')
            st.plotly_chart(fig)

        st.markdown('''<div style="text-align: justify;"> It is wonderful to be able to say that the model seems to have learned to differentiate iPhone \
                    11 models based on price (even including different memory variants). It is clearly visible that \
                    predictions of prices for "Max" prompts are of greater value than "Pro" prompts and those \
                    are above in price in regards to usual "iphone 11" prompts. The prices are also higher for \
                    models with more memory for each iPhone 11 type.</div>''', unsafe_allow_html=True)
        f''
        st.markdown('''<div style="text-align: justify;"> 
                    On the other hand the model does not seem to grasp time series relevance - after crossing the \
                    barrier of 55 days (up to which samples are present in dataset) the model always flattens out price values, \
                    no matter what was the trend before the barrier. It is also worth mentioning that there is a \
                    difference in price prediction standard deviation in phrases like "64" vs "64gb" - the \
                    latter, which is probably more common in the dataset, outputs predictions with considerably less \
                    standard deviation (example with the "64" phrase is in **04_auto_model.ipynb** notebook).</div>''', unsafe_allow_html=True)

