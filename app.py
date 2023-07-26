from flask import Flask,request,render_template
import numpy as np
import os
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

## route for home page



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        pass
    else:
        
        data=CustomData(
            date=request.form.get('date'),
            month=request.form.get('month'),
            year=request.form.get('year'),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('index.html',results=results[0])

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)
