from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from io import BytesIO
import boto3
import os
from dotenv import load_dotenv

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
        raise HTTPException(status_code=404, detail=f"El medicamento '{medication_name}' no se encontró en la matriz de correlación.")



import numpy as np

@app.post("/predict")
def predict(request: PredictRequest):
    folder_name = request.folder_name
    file_name = request.file_name
    description = request.description
    periods = request.periods

    # Obtener el archivo de Excel desde S3
    df = get_excel_file_from_s3(folder_name, file_name)

    # Transformar los datos
    df_melted = df.melt(id_vars=["DESCRIPCION"], var_name="Fecha", value_name="Valor")
    df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], format="%m-%Y")

    # Filtrar por el medicamento específico
    df_filtered = df_melted[df_melted["DESCRIPCION"] == description].copy()
    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="Descripción no encontrada en los datos.")

    df_filtered = df_filtered[["Fecha", "Valor"]].rename(columns={"Fecha": "ds", "Valor": "y"})

    # Crear características para XGBoost
    df_filtered['month'] = df_filtered['ds'].dt.month
    df_filtered['year'] = df_filtered['ds'].dt.year

    # Dividir datos en conjunto de entrenamiento y prueba
    X = df_filtered[['month', 'year']]
    y = df_filtered['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo XGBoost
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular RMSE manualmente
    rmse = np.sqrt(((y_test - y_pred) ** 2).mean())  # Fórmula manual de RMSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Predecir los próximos 'periods' meses
    last_date = df_filtered['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['month'] = future_df['ds'].dt.month
    future_df['year'] = future_df['ds'].dt.year

    # Hacer predicciones para los meses futuros
    future_df['y_pred'] = model.predict(future_df[['month', 'year']])

    # Convertir datos históricos y predicciones a formato dict para la respuesta
    historical_data = df_filtered.to_dict(orient="records")
    predictions = future_df[['ds', 'y_pred']].rename(columns={'y_pred': 'yhat'}).to_dict(orient="records")

    return {
        "historical_data": historical_data,
        "predictions": predictions,
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        },
        "model": "XGBoost"
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