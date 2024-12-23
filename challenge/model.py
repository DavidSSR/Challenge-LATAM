import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Tuple, Union, List
import numpy as np
import pickle

top_features = [
    "OPERA_Latin American Wings", 
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
       ]

class DelayModel:
    """
    DelayModel loads a predictive model and sets top features for prediction.
    """
    def __init__(
        self
    ):
        """
        Loads the model from disk if available, otherwise sets it to None. 
        Initializes top features and weight_handler for model balancing.

        Attributes:
            - self._model: The loaded model for prediction, or None if loading fails.
            - self.top_features: List of top features for preprocessing and prediction.
            - self.weight_handler: Handler for balancing the model through weights, initialized as None.
        """
        self.weight_handler = None
        self.top_features = top_features
        try:
            self._model = pickle.load(open("data/model.sav", 'rb'))
        except:
            self._model = None 
    

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        The function goes through the following steps:
        - Create a new column called 'period_day' based on the scheduled date and time of the flight.
            * This column indicates the period of the day in which the flight is scheduled to depart (mañana, tarde, noche).
        - Create a new column called 'high_season' based on the scheduled date and time of the flight.
            * This column contains 1 if the flight is in high season (15-Dec - 31-Dec, 1-Jan - 3-Mar, 15-Jul - 31-Jul, 11-Sep - 30-Sep), 0 otherwise.
        - Create a new column called 'min_diff' based on the difference between the scheduled and actual times.
            * This column contains the difference in minutes between the scheduled and actual times.
        - Create a new column called 'delay' based on the 'min_diff' column.
            * This column contains 1 if the delay is greater than 15 minutes and 0 otherwise.
        - Create dummy variables for the following columns: 'OPERA', 'TIPOVUELO', 'MES'.
            * Dummy variables are created for the top 10 values of each column.
        - Save the selected features and target columns for training or prediction.
        - Return the processed features and target if a target column is provided, otherwise just the features.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        if 'Fecha-I' in data.columns:
           data['period_day'] = data['Fecha-I'].apply(self.get_period_day) 
           data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
           data['min_diff'] = data.apply(self.get_min_diff, axis = 1)
           threshold_in_minutes = 15
           data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        features = pd.concat([
         pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
         pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
         pd.get_dummies(data['MES'], prefix = 'MES')], 
         axis = 1
         )
    
        

        if target_column is not None:
           target = data[target_column]
           self.get_balance_data(features, target)

        if not set(top_features).issubset(features.columns):
            data_predict = []
            for _, row in features.iterrows():
                row_data = [0] * len(top_features)
                for col in features.columns.tolist():
                    if col in top_features:
                       row_data[top_features.index(col)] = row[col]
                data_predict.append(row_data)
            features = pd.DataFrame(data_predict, columns=top_features)
            
        
        return (features[top_features], target.to_frame()) if target_column else features[top_features]


    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.
            The function performs the following steps:
            - Split the preprocessed data into training and validation sets (33% for validation).
            - Initialize a Logistic Regression model with class weights to handle imbalanced data.
            - Train the model using the training features and target.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train2, _, y_train2, _ = train_test_split(features, target, test_size = 0.33, random_state = 42)
        self._model = LogisticRegression(class_weight= self.weight_handler)
        self._model.fit(x_train2,y_train2)
        pickle.dump(self._model, open('data/model.sav','wb'))

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.
        The function uses the trained model to predict delays for new flights based on the provided features.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        model = self._model
        predictions = model.predict(features)
        return predictions.tolist()
    


    def get_balance_data(self, features: pd.DataFrame, target: pd.Series):
        """
        Calculate and store class weights for balancing the model.

        Args:
            features (pd.DataFrame): The feature data for splitting.
            target (pd.Series): The target labels for class distribution.

        Returns:
            None: The method updates the weight_handler attribute with the calculated class weights.
        """    
        _, _, y_train, _ = train_test_split(features, target, test_size = 0.33, random_state = 42)
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        self.weight_handler = {1: n_y0/len(y_train), 0: n_y1/len(y_train)}
    
    @staticmethod
    def get_period_day(date):
        """
        Determine the part of the day based on the given date and time.

        Args:
            date (str): A string representing the date and time in the format '%Y-%m-%d %H:%M:%S'.
    
        Returns:
            str: The part of the day ('mañana', 'tarde', or 'noche').
        """
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()
    
        if(date_time > morning_min and date_time < morning_max):
            return 'mañana'
        elif(date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif(
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
        ):
            return 'noche'
        
    @staticmethod
    def is_high_season(fecha):
        """
        Check if the given date is within a high season period.

        Args:
            fecha (str): The date in '%Y-%m-%d %H:%M:%S' format.
    
        Returns:
            int: 1 if the date is within a high season period, 0 otherwise.
        """
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
    
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0
    @staticmethod  
    def get_min_diff(data):
        """
        Calculate the difference in minutes between two datetime values: 'Fecha-O' and 'Fecha-I'.

        Args:
           data (dict): A dictionary containing 'Fecha-O' and 'Fecha-I' as keys with datetime values in '%Y-%m-%d %H:%M:%S' format.
    
        Returns:
           float: The difference in minutes between 'Fecha-O' and 'Fecha-I'.
        """
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
    
