from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prophet import Prophet
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

    # Inicializar y ajustar el modelo con los datos de entrenamiento
    model = Prophet()
    model.fit(df_filtered)

    # Hacer una predicción para los próximos 'periods' meses
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)

    # Convertir los datos históricos y las predicciones a un formato dict para la respuesta JSON
    historical_data = df_filtered.to_dict(orient="records")
    predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient="records")

    return {
        "historical_data": historical_data,
        "predictions": predictions
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