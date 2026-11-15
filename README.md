# Renewable Energy Data Engineering Pipeline and Forecasting

An end-to-end data engineering and analytics project focused on collecting, processing, analysing, and forecasting renewable energy data.

## Project Overview

This project develops a data engineering pipeline for renewable energy forecasting. It integrates data ingestion, real-time streaming, distributed data processing, exploratory data analysis, machine learning, and dashboard visualisation.

The objective is to transform raw renewable energy and weather-related data into meaningful insights that can support energy analysis and forecasting.

## Project Architecture

The project is organised into six phases:

### Phase 1 – Raw Data Collection & Data Ingestion
- Collect raw renewable energy and related datasets.
- Organise and prepare datasets for further processing.
- Perform initial data validation and ingestion.

### Phase 2 – Real-Time Data Pipeline
- Build a real-time data ingestion workflow.
- Use Apache Kafka for streaming data.
- Use Apache NiFi for data flow management.
- Organise incoming data for downstream processing.

### Phase 3 – Spark Preprocessing
- Process datasets using Apache Spark.
- Perform data cleaning and preprocessing.
- Handle missing values and prepare structured datasets for analysis.

### Phase 4 – Data Analysis
- Perform exploratory data analysis (EDA).
- Analyse renewable energy trends and patterns.
- Generate visualisations and statistical insights.

### Phase 5 – Machine Learning
- Develop machine-learning models for renewable energy forecasting.
- Train and evaluate models using prepared datasets.
- Analyse model performance using evaluation metrics.

### Phase 6 – Dashboard Deployment
- Develop an interactive dashboard using Streamlit.
- Present analytical results and forecasting insights.
- Provide visual access to the processed data and project results.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Streaming | Apache Kafka |
| Data Flow Management | Apache NiFi |
| Distributed Processing | Apache Spark |
| Workflow Orchestration | Apache Airflow |
| Database | PostgreSQL |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## Project Structure

```text
Renewable-Energy-Data-Engineering-Pipeline-and-Forecasting/
│
├── Phase_1/
│   └── Raw Data Collection and Ingestion
│
├── Phase_2/
│   └── Real-Time Data Pipeline
│
├── Phase_3/
│   └── Spark Preprocessing
│
├── Phase_4/
│   └── Data Analysis and EDA
│
├── Phase_5/
│   └── Machine Learning and Forecasting
│
├── Phase_6/
│   └── Dashboard Deployment
│
├── Renewable_energy.pptx
├── .gitignore
└── README.md
```

*Note: Folder names above are illustrative. Adjust them to match the actual names in the repository.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/manideepgalipellids/Renewable-Energy-Data-Engineering-Pipeline-and-Forecasting.git
```

### 2. Navigate to the project directory

```bash
cd Renewable-Energy-Data-Engineering-Pipeline-and-Forecasting
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 4. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Install the required libraries and configure the supporting services, such as Kafka, NiFi, Spark, Airflow, and PostgreSQL, according to the phase you want to run.

## Results and Deliverables

The repository includes project implementation files and supporting materials, including:

- Data
