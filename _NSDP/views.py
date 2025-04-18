from django.shortcuts import render
from django.shortcuts import render
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import urllib, base64
from django.conf import settings


# Load dataset (assuming it's placed in the 'static' folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'dataset.csv')
df = pd.read_csv(csv_path)

df = df[:-3]
# Clean industry names
df['Industry Group'] = df['Industry Group'].str.strip()
df = df.drop(columns=['Sr.No.'])

# Extract unique industries
industries = df['Industry Group'].unique()

# View function to select industry and show analysis
def industry_insight(request):
    selected_industry = request.GET.get('industry')
    chart = None
    prediction = None
    insights = None

    if selected_industry:
        data = df[df['Industry Group'] == selected_industry].drop(columns=['Industry Group']).T
        data.columns = ['NSDP']
        data.index = data.index.str.extract(r'(\d{4})')[0].astype(int)

        # Regression
        X = data.index.values.reshape(-1, 1)
        y = data['NSDP'].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        future_year = 2025
        future_value = model.predict([[future_year]])[0]

        # Plot
        plt.figure(figsize=(8,4))
        plt.plot(X, y, 'bo-', label='Actual')
        plt.plot(X, y_pred, 'r--', label='Predicted')
        plt.title(f"NSDP Trend: {selected_industry}")
        plt.xlabel("Year")
        plt.ylabel("NSDP")
        plt.legend()
        plt.grid(True)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        chart = base64.b64encode(image_png).decode('utf-8')

        # Generate basic insight
        slope = model.coef_[0]
        r2 = model.score(X, y)
        insights = {
            'slope': round(slope, 2),
            'r2': round(r2, 2),
            'future_year': future_year,
            'future_prediction': round(future_value, 2)
        }

    return render(request, '_NSDP/industries.html', {
        'industries': industries,
        'selected_industry': selected_industry,
        'chart': chart,
        'insights': insights
    })

# Create your views here.
