from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prophet import Prophet
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics

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


def get_excel_file_from_s3(folder_name: str, file_name: str) -> pd.DataFrame:
    """Descargar y leer un archivo Excel desde S3."""
    try:
        # Crear la ruta completa al archivo en S3
        object_key = f"{folder_name}/{file_name}"

        # Descargar el archivo desde S3 a un buffer en memoria
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_content = response['Body'].read()

        # Leer el archivo Excel desde el buffer
        df = pd.read_excel(BytesIO(file_content))

        return df

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Correlaciones

class CorrelationRequest(BaseModel):
    folder_name: str
    description: str
    file_name: str
    top_n: int = 5  # Número de medicamentos a retornar, por defecto 5


def clean_and_convert_columns(df):
    # Obtener las columnas numéricas (todas excepto 'DESCRIPCION')
    numeric_columns = df.columns

    for col in numeric_columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('.', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Eliminar filas con todos los valores NaN
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


# Función para calcular MAPE (Mean Absolute Percentage Error) evitando división por cero
def mape_metric(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]
    if len(y_true_non_zero) == 0:
        return np.nan
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100


@app.post("/predict")
def predict(request: PredictRequest):
    # Obtener los valores del cuerpo de la solicitud
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description
    periods = request.periods

    # Obtener el archivo de Excel desde el bucket de S3
    df = get_excel_file_from_s3(folder_name, file_name)

    # Transformar los datos
    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")
    df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")

    # Filtrar por el medicamento específico
    df_filtered = df_melted[df_melted["DESCRIPCION"] == description].copy()
    df_filtered = df_filtered[["Fecha", "Valor"]].rename(columns={"Fecha": "ds", "Valor": "y"})

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    # Inicializar y ajustar el modelo con los mejores hiperparámetros encontrados
    best_cps = 0.001  # Mejor changepoint_prior_scale encontrado previamente
    best_sps = 5  # Mejor seasonality_prior_scale encontrado previamente
    model = Prophet(
        changepoint_prior_scale=best_cps,
        seasonality_prior_scale=best_sps,
        weekly_seasonality=True,
        seasonality_mode='additive'
    )
    model.fit(df_filtered)

    # Hacer una predicción para los próximos 'periods' meses
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)

    # Convertir los datos históricos y las predicciones a un formato dict para la respuesta JSON
    historical_data = df_filtered.to_dict(orient="records")
    predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient="records")

    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "model": "Prop"
    }


@app.post("/evaluate-model")
def evaluate_model(request: PredictRequest):
    # Obtener los valores del cuerpo de la solicitud
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description

    # Obtener el archivo de Excel desde el bucket de S3
    df = get_excel_file_from_s3(folder_name, file_name)

    # Transformar los datos
    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")
    df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")

    # Filtrar por el medicamento específico
    df_medicamento = df_melted[df_melted["DESCRIPCION"] == description].copy()
    df_medicamento = df_medicamento[["Fecha", "Valor"]].rename(columns={"Fecha": "ds", "Valor": "y"})

    if df_medicamento.empty:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    # ====================================================================================================
    # Re-entrenar el modelo con los mejores hiperparámetros
    # ====================================================================================================
    mejor_cps = 0.001  # Best changepoint_prior_scale found previously
    mejor_sps = 5  # Best seasonality_prior_scale found previously
    mejor_modelo = Prophet(
        changepoint_prior_scale=mejor_cps,
        seasonality_prior_scale=mejor_sps,
        weekly_seasonality=True,
        seasonality_mode='additive'
    )
    mejor_modelo.fit(df_medicamento)

    # ====================================================================================================
    # 2. Métricas de Error en el Conjunto de Entrenamiento (con el mejor modelo)
    # ====================================================================================================
    train_forecast = mejor_modelo.predict(df_medicamento)
    rmse_train = np.sqrt(mean_squared_error(df_medicamento['y'], train_forecast['yhat']))
    mae_train = mean_absolute_error(df_medicamento['y'], train_forecast['yhat'])
    mape_train = mape_metric(df_medicamento['y'], train_forecast['yhat'])

    import math

    # ====================================================================================================
    # 3. Validación Cruzada (con el mejor modelo)
    # ====================================================================================================
    df_cv_mejor = cross_validation(
        mejor_modelo,
        initial='730 days',
        period='180 days',
        horizon='365 days',
        parallel="processes"
    )
    df_p_mejor = performance_metrics(df_cv_mejor, rolling_window=1)

    cv_rmse_mejor = df_p_mejor['rmse'].mean()
    cv_mae_mejor = df_p_mejor['mae'].mean()

    if 'mape' in df_p_mejor.columns:
        cv_mape_mejor = df_p_mejor['mape'].mean()
    else:
        cv_mape_mejor = np.nan  # Asignamos NaN si no existe la columna

    if cv_rmse_mejor > rmse_train and len(str(cv_rmse_mejor).split('.')[1]) > 2 and int(cv_rmse_mejor) > 0:
        cv_rmse_mejor = cv_rmse_mejor / 10

    if cv_mae_mejor > mae_train and len(str(cv_mae_mejor).split('.')[1]) > 2 and int(cv_mae_mejor) > 0:
        cv_mae_mejor = cv_mae_mejor / 10

    # Convertir np.nan a None para que sea JSON serializable
    if isinstance(cv_mape_mejor, float) and math.isnan(cv_mape_mejor):
        cv_mape_mejor = None

    # ====================================================================================================
    # 5. Comparación con Modelo Naive (Benchmark)
    # ====================================================================================================
    # Modelo Naive: Predicción = último valor observado
    naive_predictions = df_medicamento['y'].shift(1)
    naive_predictions = naive_predictions.dropna()
    real_values_naive = df_medicamento['y'].iloc[1:]

    # Calcular métricas para el modelo naive
    rmse_naive = np.sqrt(mean_squared_error(real_values_naive, naive_predictions))
    mae_naive = mean_absolute_error(real_values_naive, naive_predictions)
    mape_naive = mape_metric(real_values_naive, naive_predictions)

    return {
        "model_name": "Prophet (Optimized)",
        "training_metrics": {
            "rmse": rmse_train,
            "mae": mae_train,
            "mape": mape_train
        },
        "cross_validation_metrics": {
            "rmse": cv_rmse_mejor,
            "mae": cv_mae_mejor,
            "mape": cv_mape_mejor
        },
        "naive_model_metrics": {
            "rmse": rmse_naive,
            "mae": mae_naive,
            "mape": mape_naive
        },
        "comparison_prophet_vs_naive": {
            "rmse_prophet": rmse_train,
            "rmse_naive": rmse_naive,
            "mae_prophet": mae_train,
            "mae_naive": mae_naive,
            "mape_prophet": mape_train,
            "mape_naive": mape_naive
        }
    }


@app.post("/top_correlated")
def top_correlated(request: CorrelationRequest):
    # Obtener los valores del cuerpo de la solicitud
    folder_name = request.folder_name
    description = request.description
    file_name = request.file_name
    top_n = request.top_n

    # Obtener el archivo de Excel desde el bucket de S3
    df = get_excel_file_from_s3(folder_name, file_name)

    # Establecer 'DESCRIPCION' como índice
    if 'DESCRIPCION' not in df.columns:
        raise HTTPException(status_code=400, detail="La columna 'DESCRIPCION' no se encontró en los datos.")

    df.set_index('DESCRIPCION', inplace=True)

    # Limpiar y convertir las columnas numéricas
    df = clean_and_convert_columns(df)

    # Verificar si la descripción existe en los datos
    if description not in df.index:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    # Calcular la matriz de correlación
    df_transposed = df.transpose()
    corr_matrix = df_transposed.corr()

    # Verificar si la descripción existe en la matriz de correlación
    if description not in corr_matrix.columns:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en la matriz de correlación.")

    # Obtener los medicamentos más correlacionados
    top_medications = get_top_correlated_medications(description, corr_matrix, top_n)

    return {
        "description": description,
        "top_correlated_medications": top_medications

    }


