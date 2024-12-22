import fastapi
from challenge.model import DelayModel 
import pandas as pd
from pydantic import BaseModel,field_validator

app = fastapi.FastAPI()
delay_model = DelayModel()

valid_opera = ['American Airlines','Air Canada' ,'Air France' ,'Aeromexico',
 'Aerolineas Argentinas', 'Austral' ,'Avianca', 'Alitalia' ,'British Airways',
 'Copa Air', 'Delta Air', 'Gol Trans', 'Iberia', 'K.L.M.', 'Qantas Airways',
 'United Airlines', 'Grupo LATAM', 'Sky Airline', 'Latin American Wings',
 'Plus Ultra Lineas Aereas', 'JetSmart SPA', 'Oceanair Linhas Aereas',
 'Lacsa']

required_columns = ['OPERA', 'TIPOVUELO', 'MES']

class FlightInfo(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class Flight(BaseModel):
    flights: list[FlightInfo]

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }
@app.post("/predict",status_code=200)
async def post_predict(request : Flight) -> dict:
    """
    Handle POST requests for predicting delays of flights.

    The function processes the incoming flight data, performs feature preprocessing,
    and returns the prediction of delays using the trained model.

    Steps:
        - Extract flight data from the request.
        - Convert the data into a DataFrame for preprocessing.
        - Preprocess the data by generating features for prediction.
        - Use the trained model to predict delays based on the processed features.
        - Return the prediction results.

    Args:
        request (Flight): Incoming flight data in the expected format.

    Returns:
        dict: A dictionary containing the prediction results for the delays.

    Raises:
        fastapi.HTTPException: If an error occurs during processing or validation.
    """

    try:
        data = pd.DataFrame([dict(flight) for flight in request.flights])
        validate_data(data)

        features = delay_model.preprocess(data)
        predictions = delay_model.predict(features)
        return {"predict": predictions}

    
    except fastapi.HTTPException as e:
        raise e
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")
    

def validate_data(df: pd.DataFrame):
    """
    Validates the DataFrame to ensure the data in OPERA, TIPOVUELO, and MES columns is correct.

    Args:
        df (pd.DataFrame): The input DataFrame to validate.

    Raises:
        HTTPException: If any validation fails.
    """
    required_columns = ['OPERA', 'TIPOVUELO', 'MES']
    for column in required_columns:
        if column not in df.columns:
            raise fastapi.HTTPException(status_code=400, detail=f"Missing required column: {column}")

    invalid_opera = df[~df['OPERA'].isin(valid_opera)]
    if not invalid_opera.empty:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"Invalid values in 'OPERA': {invalid_opera['OPERA'].unique().tolist()}"
        )

    invalid_tipovuelo = df[~df['TIPOVUELO'].isin(['N', 'I'])]
    if not invalid_tipovuelo.empty:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"Invalid values in 'TIPOVUELO': {invalid_tipovuelo['TIPOVUELO'].unique().tolist()}"
        )

    invalid_mes = df[(df['MES'] < 1) | (df['MES'] > 12)]
    if not invalid_mes.empty:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"Invalid values in 'MES': {invalid_mes['MES'].unique().tolist()}"
        )