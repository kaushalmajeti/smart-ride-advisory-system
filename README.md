# Smart Ride

Smart Ride is an AI-powered ride comparison and recommendation web app. It compares ride prices, predicts surge, profiles customers, and recommends the best Ola/Uber ride for each user.

## Features

- Public landing page explaining the project and ML pipeline
- Login/demo flow
- Ride fare comparison for Ola Auto, Ola Mini, Uber Go, and Uber Sedan
- Route distance and fare estimation
- Customer segmentation with K-Means style personas
- Full ML mode with XGBoost, Random Forest, Gradient Boosting, Stacking, LSTM, K-Means, surge regression, and SHAP-style explanation
- Model performance dashboard
- DynamoDB ride history with local fallback

## Run Locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start fast demo mode:

```powershell
python app.py
```

Start full ML mode:

```powershell
$env:SMART_RIDE_FULL_ML="1"
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

## Project Structure

- `app.py` - Flask backend and API routes
- `recommender.py` - ML recommendation engine
- `home.html`, `home.css` - public homepage
- `index.html`, `script.js`, `style.css` - main app UI
- `login.html`, `auth.js` - login/demo UI
- `charts/` - generated ML charts for dashboard/presentation
- `generate_charts.py` - chart generation script

## Note

The frontend currently contains browser-side Mapbox and Cognito configuration. For a public production repository, replace those values with environment-driven configuration or restrict the keys in the provider dashboards.
