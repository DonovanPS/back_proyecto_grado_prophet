from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
import numpy as np

# Cargar las variables de entorno
load_dotenv()

# Configuración del cliente S3
s3_client = boto3.client(
    's3',
    region_name=os.getenv('AWS_BUCKET_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_KEY_ID')
)

bucket_name = os.getenv('AWS_BUCKET_NAME')

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    folder_name: str
    description: str
    file_name: str
    periods: int  # Número de meses a predecir


class CorrelationRequest(BaseModel):
    folder_name: str
    description: str
    file_name: str
    top_n: int = 5  # Número de medicamentos a retornar, por defecto 5


def get_excel_file_from_s3(folder_name: str, file_name: str) -> pd.DataFrame:
    """Descargar y leer un archivo Excel desde S3."""
    try:
        object_key = f"{folder_name}/{file_name}"
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_content = response['Body'].read()
        df = pd.read_excel(BytesIO(file_content))
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def clean_and_convert_columns(df):
    numeric_columns = df.columns
    for col in numeric_columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('.', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(how='all', inplace=True)
    return df


def get_top_correlated_medications(medication_name, corr_matrix, top_n=5):
    try:
        medication_correlations = corr_matrix[medication_name]
        medication_correlations = medication_correlations.drop(labels=[medication_name])
        sorted_correlations = medication_correlations.abs().sort_values(ascending=False)
        top_medications = sorted_correlations.head(top_n)
        result = []
        for med in top_medications.index:
            corr_value = medication_correlations[med]
            result.append({
                "medication": med,
                "correlation": corr_value
            })
        return result
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"El medicamento '{medication_name}' no se encontró en la matriz de correlación.")


def mape_metric(y_true, y_pred):
    """Calcula el MAPE evitando división por cero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    y_true_nz = y_true[non_zero]
    y_pred_nz = y_pred[non_zero]
    if len(y_true_nz) == 0:
        return np.nan
    return np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100


def rolling_forecast_cv_xgb(X, y, initial, horizon=1, step=1):
    """
    Realiza validación cruzada de tipo rolling forecast para XGBoost.
    Para cada corte se entrena el modelo con los datos hasta ese punto y se pronostica 'horizon' periodo(s) adelante.
    """
    rmse_list = []
    mae_list = []
    mape_list = []
    for i in range(initial, len(y) - horizon + 1, step):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[i:i + horizon]
        y_test = y.iloc[i:i + horizon]
        try:
            model_cv = XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model_cv.fit(X_train, y_train)
        except Exception as e:
            continue  # Si falla en algún corte, se omite
        y_pred_cv = model_cv.predict(X_test)
        rmse_cv = np.sqrt(mean_squared_error(y_test, y_pred_cv))
        mae_cv = mean_absolute_error(y_test, y_pred_cv)
        mape_cv = mape_metric(y_test, y_pred_cv)
        rmse_list.append(rmse_cv)
        mae_list.append(mae_cv)
        mape_list.append(mape_cv)
    if len(rmse_list) == 0:
        return np.nan, np.nan, np.nan
    return np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)


@app.post("/predict")
def predict(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description
    periods = request.periods

    # Obtener y transformar los datos
    df = get_excel_file_from_s3(folder_name, file_name)
    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")
    df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")
    df_filtered = df_melted[df_melted["DESCRIPCION"] == description].copy()
    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")
    df_filtered = df_filtered[["Fecha", "Valor"]].rename(columns={"Fecha": "ds", "Valor": "y"})

    # Crear variables temporales para XGBoost
    df_filtered['month'] = df_filtered['ds'].dt.month
    df_filtered['year'] = df_filtered['ds'].dt.year

    # Entrenamiento del modelo
    X = df_filtered[['month', 'year']]
    y = df_filtered['y']
    # Se hace un split para obtener métricas en test (usado solo para la predicción aquí)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Predicción para los próximos 'periods' meses
    last_date = df_filtered['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['month'] = future_df['ds'].dt.month
    future_df['year'] = future_df['ds'].dt.year
    future_df['y_pred'] = model.predict(future_df[['month', 'year']])

    historical_data = df_filtered.to_dict(orient="records")
    predictions = future_df[['ds', 'y_pred']].rename(columns={'y_pred': 'yhat'}).to_dict(orient="records")

    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "metrics": {
            "rmse": rmse,
            "mae": mae
        },
        "model": "XGBoost"
    }


import math


def scale_if_leading_zero(x):
    """
    Si x es menor que 1, multiplica x por 10^(ceil(-log10(x)) + 1) para que la parte entera tenga dos dígitos.
    Luego se formatea con dos decimales usando coma como separador.
    Si x es mayor o igual que 1, se retorna el valor sin modificar (sólo se reemplaza el punto decimal por coma).
    """
    if x < 1:
        factor = 10 ** (math.ceil(-math.log10(x)) + 0)
        scaled = x * factor
        return scaled

    else:
        return x


# --- Dentro del endpoint /evaluate-model para XGBoost ---

@app.post("/evaluate-model")
def evaluate_model(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description

    # Preparar y transformar los datos
    df = get_excel_file_from_s3(folder_name, file_name)
    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")
    df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")
    df_filtered = df_melted[df_melted["DESCRIPCION"] == description].copy()
    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")
    df_filtered = df_filtered[["Fecha", "Valor"]].rename(columns={"Fecha": "ds", "Valor": "y"})

    # Crear variables para el modelo
    df_filtered['month'] = df_filtered['ds'].dt.month
    df_filtered['year'] = df_filtered['ds'].dt.year
    X = df_filtered[['month', 'year']]
    y = df_filtered['y']

    # 1. Métricas en el conjunto completo (training)
    full_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    full_model.fit(X, y)
    y_full_pred = full_model.predict(X)
    rmse_train = np.sqrt(mean_squared_error(y, y_full_pred))
    mae_train = mean_absolute_error(y, y_full_pred)
    mape_train = mape_metric(y, y_full_pred)

    # Aplicar escalado y formato SOLO si el número es menor a 1
    rmse_train_fmt = scale_if_leading_zero(rmse_train)
    mae_train_fmt = scale_if_leading_zero(mae_train)
    mape_train_fmt = scale_if_leading_zero(mape_train)

    # 2. Validación cruzada con rolling forecast (pronóstico de 1 periodo)
    initial = int(len(y) * 0.7) if len(y) > 10 else 1
    cv_rmse, cv_mae, cv_mape = rolling_forecast_cv_xgb(X, y, initial, horizon=1, step=1)
    cv_rmse_fmt = scale_if_leading_zero(cv_rmse)
    cv_mae_fmt = scale_if_leading_zero(cv_mae)
    cv_mape_fmt = scale_if_leading_zero(cv_mape)

    # 3. Modelo naive: predicción = último valor observado
    naive_pred = y.shift(1).dropna()
    actual_naive = y.iloc[1:]
    rmse_naive = np.sqrt(mean_squared_error(actual_naive, naive_pred))
    mae_naive = mean_absolute_error(actual_naive, naive_pred)
    mape_naive = mape_metric(actual_naive, naive_pred)
    rmse_naive_fmt = scale_if_leading_zero(rmse_naive)
    mae_naive_fmt = scale_if_leading_zero(mae_naive)
    mape_naive_fmt = scale_if_leading_zero(mape_naive)

    return {
        "model_name": "XGBoost",
        "training_metrics": {
            "rmse": rmse_train_fmt,
            "mae": mae_train_fmt,
            "mape": mape_train_fmt
        },
        "cross_validation_metrics": {
            "rmse": cv_rmse_fmt,
            "mae": cv_mae_fmt,
            "mape": cv_mape_fmt
        },
        "naive_model_metrics": {
            "rmse": rmse_naive_fmt,
            "mae": mae_naive_fmt,
            "mape": mape_naive_fmt
        },
        "comparison_xgboost_vs_naive": {
            "rmse_xgboost": rmse_train_fmt,
            "rmse_naive": rmse_naive_fmt,
            "mae_xgboost": mae_train_fmt,
            "mae_naive": mae_naive_fmt,
            "mape_xgboost": mape_train_fmt,
            "mape_naive": mape_naive_fmt
        }
    }


@app.post("/top_correlated")
def top_correlated(request: CorrelationRequest):
    folder_name = request.folder_name
    description = request.description
    file_name = request.file_name
    top_n = request.top_n

    df = get_excel_file_from_s3(folder_name, file_name)
    if 'DESCRIPCION' not in df.columns:
        raise HTTPException(status_code=400, detail="La columna 'DESCRIPCION' no se encontró en los datos.")
    df.set_index('DESCRIPCION', inplace=True)
    df = clean_and_convert_columns(df)
    if description not in df.index:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")
    df_transposed = df.transpose()
    corr_matrix = df_transposed.corr()
    if description not in corr_matrix.columns:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en la matriz de correlación.")
    top_medications = get_top_correlated_medications(description, corr_matrix, top_n)
    return {
        "description": description,
        "top_correlated_medications": top_medications
    }
