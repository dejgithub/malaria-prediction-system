# 🦟 Malaria Prediction System

A **production-ready** Malaria Disease Prediction System using a **Hybrid Deep Learning Model (RNN + LSTM + GRU)** with a modern web interface and cloud deployment architecture.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Malaria Prediction System                │
├──────────────────────┬──────────────────────────────────────┤
│    Frontend (Vercel) │    Backend (Render)                  │
│    ┌──────────────┐  │    ┌──────────────────────────────┐  │
│    │  Next.js 15  │  │    │  FastAPI + TensorFlow/Keras  │  │
│    │  TypeScript  │◄─┼──►│  Python 3.11                 │  │
│    │  Tailwind CSS│  │    │  Uvicorn/Gunicorn            │  │
│    │  Chart.js    │  │    │  Pandas, NumPy, Scikit-learn │  │
│    └──────────────┘  │    └──────────────────────────────┘  │
└──────────────────────┴──────────────────────────────────────┘
```

## Features

- **Hybrid Deep Learning Model**: RNN → LSTM → GRU architecture
- **Data Preprocessing**: Missing values, outliers, scaling, sequence generation
- **Multi-Period Forecasting**: 1, 3, 6, and 12-month predictions
- **Risk Classification**: Low, Moderate, High, Critical risk levels
- **Automated Insights**: AI-generated analysis and recommendations
- **Interactive Dashboard**: Real-time charts and visualizations
- **Report Generation**: Standalone HTML and CSV reports
- **API-First Design**: Fully documented REST API (Swagger/OpenAPI)
- **Cloud Deployable**: Ready for Vercel (frontend) and Render (backend)

## Model Architecture

```
Input Layer (Sequence Length: 12)
    ↓
Simple RNN (64 units) + Dropout (0.2)
    ↓
LSTM (128 units) + Dropout (0.2)
    ↓
GRU (64 units)
    ↓
Dense (32 units, ReLU)
    ↓
Output Layer (1 unit, Linear)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **Max Epochs**: 100
- **Early Stopping**: Patience=15
- **ReduceLROnPlateau**: Patience=7, factor=0.5

## Project Structure

```
malaria-prediction-system/
├── frontend/                 # Next.js 15 frontend
│   ├── src/
│   │   ├── app/             # Next.js app directory
│   │   ├── components/      # React components
│   │   │   ├── Charts.tsx   # Chart.js visualizations
│   │   │   ├── Dashboard.tsx# Main dashboard
│   │   │   └── Sidebar.tsx  # Navigation sidebar
│   │   └── lib/
│   │       └── api.ts       # API client
│   ├── package.json
│   ├── tsconfig.json
│   └── vercel.json
├── backend/                  # FastAPI backend
│   ├── app.py               # Main API server
│   ├── config.py            # Configuration
│   ├── preprocessing.py     # Data preprocessing pipeline
│   ├── model_builder.py     # Hybrid model definition
│   ├── forecasting.py       # Forecast engine
│   ├── visualization.py     # Plot generation
│   ├── report_generator.py  # HTML/CSV report generation
│   └── insights.py          # Insights engine
├── model/                    # Trained models (auto-generated)
├── reports/                  # Generated reports (auto-generated)
├── datasets/                 # Dataset storage
│   └── sample_malaria_data.csv
├── docs/                     # Documentation
├── requirements.txt
├── Dockerfile
├── render.yaml
├── .env.example
└── .github/workflows/ci-cd.yml
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- npm or yarn

### Backend Setup

```bash
cd malaria-prediction-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m uvicorn backend.app:app --reload --port 8000
```

The API will be available at `http://localhost:8000` with Swagger docs at `http://localhost:8000/docs`.

### Frontend Setup

```bash
cd malaria-prediction-system/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

The dashboard will be available at `http://localhost:3000`.

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
PORT=8000
CORS_ORIGINS=*
API_RATE_LIMIT=100
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System information |
| GET | `/health` | Health check |
| POST | `/upload-dataset` | Upload and preprocess dataset |
| POST | `/train-model` | Train hybrid model |
| POST | `/predict` | Make prediction |
| GET | `/forecast` | Generate forecasts |
| GET | `/metrics` | Get evaluation metrics |
| GET | `/insights` | Get AI-generated insights |
| GET | `/reports` | Download reports (HTML/CSV) |
| GET | `/dataset-info` | Dataset information |
| GET | `/model-info` | Model architecture info |
| GET | `/visualizations/{name}` | Get plot images |

## Deployment

### Render (Backend)

1. Push the repository to GitHub
2. Create a new **Web Service** on Render
3. Connect your GitHub repository
4. Use the following settings:
   - **Runtime**: Python 3.11
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn backend.app:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`
5. Set environment variables in Render dashboard
6. Deploy

The `render.yaml` file is also provided for Infrastructure-as-Code deployment.

### Vercel (Frontend)

1. Push the frontend directory to GitHub
2. Import the project in Vercel
3. Set environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-render-service.onrender.com
   ```
4. Deploy

### Docker

```bash
docker build -t malaria-prediction-system .
docker run -p 8000:8000 malaria-prediction-system
```

## Dataset Format

The system expects a CSV or Excel file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Monthly timestamp |
| Malaria Cases | numeric | Target variable |
| Malaria Deaths | numeric | Monthly deaths |
| Temperature | numeric | Average temperature |
| Rainfall | numeric | Monthly rainfall |
| Humidity | numeric | Average humidity |
| Population | numeric | Population count |
| Health Facility Coverage | numeric | Coverage percentage |
| Environmental Factors | numeric | Environmental index |

## Evaluation Metrics

The model calculates:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)
- **Accuracy** (Classification accuracy)
- **Precision**, **Recall**, **F1 Score**

## Visualization

Chart.js is used for interactive visualizations:
- Line Charts (trends, loss curves)
- Bar Charts (metrics, forecast comparison)
- Pie Charts (risk distribution)
- Scatter Plots (actual vs predicted)

## Security

- Input validation on all endpoints
- File type and size validation
- API rate limiting
- CORS protection
- Environment variable management
- Error logging

## License

MIT License - see LICENSE file for details.

## Citation

If you use this system in academic research, please cite:

```bibtex
@software{malaria_prediction_system,
  author = {Malaria Prediction System},
  title = {Malaria Disease Prediction System using Hybrid RNN-LSTM-GRU},
  year = {2025},
  url = {https://github.com/yourusername/malaria-prediction-system}
}
```
