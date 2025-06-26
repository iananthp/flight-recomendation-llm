# Enhanced Flight Recommendation System with Proper TimeLLM Fine-tuning
import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core ML and DL imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv(dotenv_path="api_llm.env")

# Configuration
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Fix authentication
if not HF_TOKEN:
    print("❌ Hugging Face token not found!\nPlease run: huggingface-cli login\nOr set HF_TOKEN in your environment variables")
else:
    print("✅ Hugging Face token loaded successfully.")

# Enhanced TimeLLM Configuration
class EnhancedTimeLLMConfig:
    """Enhanced configuration for TimeLLM model with proper fine-tuning"""
    def __init__(self):
        self.seq_len = 96          # Input sequence length
        self.pred_len = 24         # Prediction length
        self.d_model = 768         # Match GPT-2 dimension
        self.n_heads = 12          # GPT-2 heads
        self.d_ff = 3072          # GPT-2 feed-forward dimension
        self.dropout = 0.1
        self.patch_len = 16
        self.stride = 8
        self.llm_model = "gpt2"    # Base LLM
        self.llm_dim = 768         # GPT-2 hidden dimension
        self.num_flight_features = 10  # Number of flight-specific features
        self.enable_gradient_checkpointing = True
        self.learning_rate = 5e-5
        self.weight_decay = 0.01

class FlightTimeSeriesDataset(Dataset):
    """Custom dataset for flight time series data"""
    
    def __init__(self, flight_data: List[Dict], tokenizer, config, mode='train'):
        self.flight_data = flight_data
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        
    def __len__(self):
        return len(self.flight_data)
    
    def __getitem__(self, idx):
        flight = self.flight_data[idx]
        
        # Create numerical features
        numerical_features = self._extract_numerical_features(flight)
        
        # Create text prompt
        text_prompt = self._create_text_prompt(flight)
        
        # Tokenize text
        tokens = self.tokenizer(
            text_prompt, 
            return_tensors="pt", 
            padding='max_length',
            truncation=True,
            max_length=128
        )
        
        # Create time series sequence
        time_series = self._create_time_series(flight)
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'numerical_features': torch.FloatTensor(numerical_features),
            'time_series': torch.FloatTensor(time_series),
            'labels': torch.FloatTensor(self._create_labels(flight))
        }
    
    def _extract_numerical_features(self, flight: Dict) -> np.ndarray:
        """Extract numerical features from flight data"""
        features = [
            flight.get('duration_minutes', 0) / 1000.0,  # Normalize
            flight.get('price', 0) / 1000.0,
            flight.get('stops', 0),
            flight.get('departure_hour', 0) / 24.0,
            flight.get('arrival_hour', 0) / 24.0,
            hash(flight.get('airline', '')) % 100 / 100.0,  # Airline encoding
            hash(flight.get('aircraft_type', '')) % 50 / 50.0,  # Aircraft encoding
            np.random.random(),  # Weather factor placeholder
            np.random.random(),  # Historical delay factor
            np.random.random()   # Route popularity
        ]
        return np.array(features[:self.config.num_flight_features])
    
    def _create_text_prompt(self, flight: Dict) -> str:
        """Create descriptive text prompt for the flight"""
        prompt = f"Flight {flight.get('flight_number', 'Unknown')} from {flight.get('departure_airport', 'Unknown')} to {flight.get('arrival_airport', 'Unknown')}. "
        prompt += f"Airline: {flight.get('airline', 'Unknown')}. "
        prompt += f"Duration: {flight.get('duration_minutes', 0)} minutes. "
        prompt += f"Price: ${flight.get('price', 0)}. "
        prompt += f"Departure time: {flight.get('departure_time', 'Unknown')}. "
        prompt += "Predict flight delay and recommend suitability."
        return prompt
    
    def _create_time_series(self, flight: Dict) -> np.ndarray:
        """Create time series data for the flight"""
        # Generate synthetic time series based on flight characteristics
        base_delay = flight.get('price', 300) / 100  # Price-based delay tendency
        
        time_series = []
        for i in range(self.config.seq_len):
            # Add seasonal patterns, noise, and flight-specific factors
            seasonal = 5 * np.sin(2 * np.pi * i / 24)  # Daily pattern
            trend = 0.1 * i  # Slight upward trend
            noise = np.random.normal(0, 2)
            flight_factor = base_delay + flight.get('stops', 0) * 5
            
            value = max(0, seasonal + trend + noise + flight_factor)
            time_series.append(value)
        
        return np.array(time_series)
    
    def _create_labels(self, flight: Dict) -> np.ndarray:
        """Create labels for prediction"""
        # Future delay predictions (next 24 time steps)
        base_delay = flight.get('price', 300) / 100
        labels = []
        
        for i in range(self.config.pred_len):
            # Predict future delays with some randomness
            future_delay = base_delay + np.random.normal(0, 5)
            labels.append(max(0, future_delay))
        
        return np.array(labels)

class PatchEmbedding(nn.Module):
    """Simple patch embedding for time series data"""
    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len]
        patches = []
        for i in range(0, x.size(1) - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len]
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # [batch, num_patches, patch_len]
        patches = self.proj(patches)           # [batch, num_patches, d_model]
        return self.dropout(patches)

class EnhancedTimeLLMModel(nn.Module):
    """Enhanced TimeLLM model with proper fine-tuning capabilities"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(
            config.d_model, config.patch_len, config.stride, config.dropout
        )
        
        # Reprogramming layer (attention mechanism)
        self.reprogramming_layer = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout
        )
        
        # Load and configure LLM
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load LLM model (trainable, not frozen)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        if config.enable_gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
        
        # Time series processing layers
        self.time_series_encoder = nn.Sequential(
            nn.Linear(config.seq_len, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Numerical features encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(config.num_flight_features, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.d_model)
        )
        
        # Multi-modal fusion layer
        self.fusion_layer = nn.MultiheadAttention(
            config.d_model, 
            config.n_heads, 
            dropout=config.dropout
        )
        
        # Output projection layers
        self.delay_predictor = nn.Sequential(
            nn.Linear(config.llm_dim, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.pred_len)
        )
        
        self.recommendation_scorer = nn.Sequential(
            nn.Linear(config.llm_dim, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )
        
        # Loss functions
        self.delay_loss = nn.MSELoss()
        self.recommendation_loss = nn.BCELoss()
        
    def forward(self, input_ids, attention_mask, numerical_features, time_series, labels=None):
        batch_size = input_ids.size(0)
        
        # Process text through LLM (now trainable)
        llm_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get last hidden state
        text_embeddings = llm_outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        text_repr = text_embeddings.mean(dim=1)  # [batch, hidden_dim]
        
        # Process time series
        time_series_repr = self.time_series_encoder(time_series)  # [batch, d_model]
        
        # Process numerical features
        numerical_repr = self.numerical_encoder(numerical_features)  # [batch, d_model]
        
        # Multi-modal fusion
        # Stack representations for attention
        multi_modal_input = torch.stack([
            time_series_repr, 
            numerical_repr, 
            text_repr[:, :self.config.d_model]  # Project if needed
        ], dim=1)  # [batch, 3, d_model]
        
        fused_repr, _ = self.fusion_layer(
            multi_modal_input, 
            multi_modal_input, 
            multi_modal_input
        )
        
        # Use mean of fused representations
        final_repr = fused_repr.mean(dim=1)  # [batch, d_model]
        
        # Predictions
        delay_pred = self.delay_predictor(final_repr)  # [batch, pred_len]
        recommendation_score = self.recommendation_scorer(final_repr)  # [batch, 1]
        
        outputs = {
            'delay_prediction': delay_pred,
            'recommendation_score': recommendation_score
        }
        
        # Calculate loss if labels provided
        if labels is not None:
            delay_loss = self.delay_loss(delay_pred, labels)
            
            # Create recommendation labels (flights with low predicted delay are good)
            rec_labels = (labels.mean(dim=1) < 15).float().unsqueeze(1)  # < 15 min avg delay
            rec_loss = self.recommendation_loss(recommendation_score, rec_labels)
            
            total_loss = delay_loss + 0.5 * rec_loss
            outputs['loss'] = total_loss
            outputs['delay_loss'] = delay_loss
            outputs['recommendation_loss'] = rec_loss
        
        return outputs

class FlightRecommendationTrainer:
    """Custom trainer for flight recommendation system"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_flights: List[Dict], val_flights: List[Dict], epochs: int = 5):
        """Train the enhanced TimeLLM model"""
        print("Preparing datasets...")
        
        # Create datasets
        train_dataset = FlightTimeSeriesDataset(train_flights, self.tokenizer, self.config, 'train')
        val_dataset = FlightTimeSeriesDataset(val_flights, self.tokenizer, self.config, 'val')
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=8,  # Smaller batch size for memory efficiency
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=8, 
            shuffle=False,
            num_workers=2
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs * len(train_loader)
        )
        
        print(f"Training enhanced TimeLLM for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            delay_loss_sum = 0
            rec_loss_sum = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in train_pbar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    numerical_features=batch['numerical_features'],
                    time_series=batch['time_series'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                delay_loss_sum += outputs['delay_loss'].item()
                rec_loss_sum += outputs['recommendation_loss'].item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Delay': f"{outputs['delay_loss'].item():.4f}",
                    'Rec': f"{outputs['recommendation_loss'].item():.4f}"
                })
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_delay_loss = 0
            val_rec_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        numerical_features=batch['numerical_features'],
                        time_series=batch['time_series'],
                        labels=batch['labels']
                    )
                    
                    val_loss += outputs['loss'].item()
                    val_delay_loss += outputs['delay_loss'].item()
                    val_rec_loss += outputs['recommendation_loss'].item()
            
            # Print epoch results
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Train Delay Loss: {delay_loss_sum/len(train_loader):.4f}")
            print(f"  Val Delay Loss: {val_delay_loss/len(val_loader):.4f}")
            print(f"  Train Rec Loss: {rec_loss_sum/len(train_loader):.4f}")
            print(f"  Val Rec Loss: {val_rec_loss/len(val_loader):.4f}")
        
        print("Training completed!")
    
    def predict(self, flight_data: Dict, weather_data: Dict) -> Tuple[float, float]:
        """Predict delay and recommendation score for a flight"""
        self.model.eval()
        
        # Create single-item dataset
        single_flight_dataset = FlightTimeSeriesDataset(
            [flight_data], self.tokenizer, self.config, 'test'
        )
        
        # Get data
        sample = single_flight_dataset[0]
        
        # Add batch dimension and move to device
        batch = {
            'input_ids': sample['input_ids'].unsqueeze(0).to(self.device),
            'attention_mask': sample['attention_mask'].unsqueeze(0).to(self.device),
            'numerical_features': sample['numerical_features'].unsqueeze(0).to(self.device),
            'time_series': sample['time_series'].unsqueeze(0).to(self.device)
        }
        
        with torch.no_grad():
            outputs = self.model(**batch)
            
            # Get predictions
            delay_pred = outputs['delay_prediction'].cpu().numpy()[0]  # [pred_len]
            rec_score = outputs['recommendation_score'].cpu().item()
            
            # Return average predicted delay and recommendation score
            avg_delay = delay_pred.mean()
            
        return avg_delay, rec_score
class AmadeusFlightAPI:
    """Amadeus API integration for flight data"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expires = None
        self.token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        self.flight_url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        
    def get_access_token(self):
        """Get access token from Amadeus API"""
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        try:
            response = requests.post(self.token_url, headers=headers, data=data)
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.token_expires = datetime.now() + timedelta(seconds=token_data['expires_in'])
                return True
            else:
                print(f"Amadeus token error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error getting access token: {e}")
        
        return False
    
    def search_flights(self, origin: str, destination: str, departure_date: str, 
                      adults: int = 1) -> List[Dict]:
        """Search for flights using Amadeus API"""
        if not self.access_token or datetime.now() >= self.token_expires:
            if not self.get_access_token():
                print("Using mock flights due to API failure")
                return self.get_mock_flights(origin, destination)
        
        headers = {'Authorization': f'Bearer {self.access_token}'}
        params = {
            'originLocationCode': origin,
            'destinationLocationCode': destination,
            'departureDate': departure_date,
            'adults': adults,
            'max': 7  # Get up to 7 flights
        }
        
        try:
            response = requests.get(self.flight_url, headers=headers, params=params)
            if response.status_code == 200:
                return self.parse_amadeus_response(response.json())
            else:
                print(f"Flight search error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error searching flights: {e}")
        
        return self.get_mock_flights(origin, destination)
    
    def parse_amadeus_response(self, data: Dict) -> List[Dict]:
        """Parse Amadeus API response"""
        flights = []
        
        for offer in data.get('data', []):
            # Safely get itinerary segments
            itinerary = offer.get('itineraries', [{}])[0]
            segments = itinerary.get('segments', [])
            
            if not segments:
                continue
                
            segment = segments[0]  # First segment
            departure = segment.get('departure', {})
            arrival = segment.get('arrival', {})
            
            flight = {
                'flight_number': f"{segment.get('carrierCode', '')}{segment.get('number', '')}",
                'airline': segment.get('carrierCode', 'Unknown'),
                'departure_airport': departure.get('iataCode', ''),
                'arrival_airport': arrival.get('iataCode', ''),
                'departure_time': departure.get('at', ''),
                'arrival_time': arrival.get('at', ''),
                'duration_minutes': self.parse_duration(itinerary.get('duration', 'PT0H0M')),
                'stops': len(segments) - 1,
                'price': float(offer.get('price', {}).get('total', 0)),
                'currency': offer.get('price', {}).get('currency', 'USD'),
                'aircraft_type': segment.get('aircraft', {}).get('code', ''),
                'departure_hour': self.extract_hour(departure.get('at', ''))
            }
            flights.append(flight)
        
        return flights
    
    def extract_hour(self, datetime_str: str) -> int:
        """Extract hour from ISO datetime string"""
        try:
            if 'T' in datetime_str:
                dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                return dt.hour
        except:
            pass
        return 12  # Default to noon if parsing fails
    
    def parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to minutes"""
        # PT1H30M format
        hours = 0
        minutes = 0
        
        # Extract hours
        if 'H' in duration_str:
            hours_part = duration_str.split('H')[0]
            hours = int(hours_part.replace('PT', ''))
            duration_str = duration_str.split('H')[1]
        
        # Extract minutes
        if 'M' in duration_str:
            minutes_part = duration_str.split('M')[0]
            minutes = int(minutes_part)
        
        return hours * 60 + minutes
    
    def get_mock_flights(self, origin: str, destination: str) -> List[Dict]:
        """Generate mock flight data for demonstration"""
        # ... (your existing mock flights implementation is correct)
        # Keep your existing mock flights implementation
        return [
            # Your existing mock flights data
        ][:7]
class WeatherDataIntegrator:
    """Integrate weather data for delay prediction"""
    def __init__(self, openweather_api_key: str):
        self.api_key = openweather_api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        # Add more airports as needed
        self.airport_coords = {
            'JFK': (40.6413, -73.7781),
            'LAX': (33.9425, -118.4081),
            'DFW': (32.8998, -97.0403),
            'ORD': (41.9742, -87.9073),  # Chicago O'Hare
            'ATL': (33.6407, -84.4277),  # Atlanta
            'SFO': (37.6213, -122.3790), # San Francisco
            'SEA': (47.4502, -122.3088), # Seattle
            'MIA': (25.7959, -80.2870),  # Miami
            'BOS': (42.3640, -71.0200),  # Boston
            'DEN': (39.8561, -104.6737)  # Denver
        }

    def get_weather_data(self, airport_code: str) -> Dict:
        """Get current weather data for airport"""
        # Use uppercase for consistency
        airport_code = airport_code.upper()
        
        if airport_code not in self.airport_coords:
            return self.get_default_weather()
        
        lat, lon = self.airport_coords[airport_code]
        
        try:
            url = f"{self.base_url}/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data['wind']['speed'],
                    'visibility': data.get('visibility', 10000),
                    'weather_condition': data['weather'][0]['main']
                }
        except Exception as e:
            print(f"Weather API error: {e}")
        
        return self.get_default_weather()
    
    def get_default_weather(self) -> Dict:
        """Return default weather data when API fails"""
        return {
            'temperature': 20,
            'humidity': 50,
            'pressure': 1013,
            'wind_speed': 10,
            'visibility': 10000,
            'weather_condition': 'Clear'
        }
    
    def calculate_weather_delay_factor(self, weather_data: Dict) -> float:
        """Calculate delay factor based on weather conditions"""
        delay_factor = 1.0
        
        # Temperature extremes
        temp = weather_data['temperature']
        if temp < -10 or temp > 40:
            delay_factor += 0.3
        
        # High winds
        wind_speed = weather_data['wind_speed']
        if wind_speed > 25:  # Strong gale
            delay_factor += 0.8
        elif wind_speed > 15:  # Moderate breeze
            delay_factor += 0.3
        
        # Low visibility
        visibility = weather_data['visibility']
        if visibility < 1000:  # Very poor visibility
            delay_factor += 1.2
        elif visibility < 3000:  # Moderate visibility
            delay_factor += 0.5
        
        # Weather conditions
        condition = weather_data['weather_condition']
        condition_factors = {
            'Thunderstorm': 0.9,
            'Rain': 0.4,
            'Snow': 0.7,
            'Fog': 0.6,
            'Haze': 0.4,
            'Dust': 0.5,
            'Clouds': 0.1
        }
        delay_factor += condition_factors.get(condition, 0)
        
        return min(delay_factor, 3.5)  # Cap at 3.5x delay factor

# Updated main components integration
class EnhancedFlightRecommendationSystem:
    """Enhanced flight recommendation system with proper TimeLLM fine-tuning"""
    
    def __init__(self, openweather_api_key: str):
        self.config = EnhancedTimeLLMConfig()
        self.model = EnhancedTimeLLMModel(self.config)
        self.trainer = FlightRecommendationTrainer(
            self.model, 
            self.model.tokenizer, 
            self.config
        )
        self.weather_integrator = WeatherDataIntegrator(openweather_api_key)
        self.is_trained = False
        
    def train_model(self, historical_flights: List[Dict], epochs: int = 5):
        """Train the enhanced TimeLLM model with proper fine-tuning"""
        print("Starting enhanced TimeLLM training with proper fine-tuning...")
        
        if len(historical_flights) < 20:
            print("Insufficient training data. Generating additional synthetic data...")
            historical_flights = self._augment_training_data(historical_flights)
        
        # Split data
        train_flights, val_flights = train_test_split(
            historical_flights, test_size=0.2, random_state=42
        )
        
        # Train model
        self.trainer.train(train_flights, val_flights, epochs)
        self.is_trained = True
        
    def _augment_training_data(self, base_flights: List[Dict]) -> List[Dict]:
        """Augment training data by creating variations"""
        augmented_flights = base_flights.copy()
        
        for _ in range(200 - len(base_flights)):  # Create up to 200 total flights
            base_flight = np.random.choice(base_flights)
            
            # Create variation
            new_flight = base_flight.copy()
            new_flight['price'] = max(50, base_flight.get('price', 300) + np.random.normal(0, 100))
            new_flight['duration_minutes'] = max(60, base_flight.get('duration_minutes', 240) + np.random.normal(0, 60))
            new_flight['departure_hour'] = np.random.randint(0, 24)
            new_flight['historical_delay'] = max(0, np.random.normal(15, 10))
            
            augmented_flights.append(new_flight)
        
        return augmented_flights
    
    def recommend_flights(self, flights: List[Dict], user_preferences: Dict) -> List[Dict]:
        """Generate enhanced recommendations using fine-tuned TimeLLM"""
        print("Generating enhanced flight recommendations...")
        
        scored_flights = []
        
        for flight in flights:
            # Get weather data
            departure_airport = flight.get('departure_airport', 'JFK')
            weather_data = self.weather_integrator.get_weather_data(departure_airport)
            
            if self.is_trained:
                # Use trained model for prediction
                predicted_delay, recommendation_score = self.trainer.predict(flight, weather_data)
            else:
                # Fallback to rule-based prediction
                predicted_delay = 15.0  # Default
                recommendation_score = 0.5
            
            # Calculate additional scores (price, comfort, etc.)
            comfort_score = self._calculate_comfort_score(
                flight, user_preferences.get('passenger_type', 'business')
            )
            
            price_score = self._calculate_price_score(
                flight, user_preferences.get('price_sensitivity', 'medium')
            )
            
            # Weather score
            weather_factor = self.weather_integrator.calculate_weather_delay_factor(weather_data)
            weather_score = max(0, 10 - weather_factor * 2)
            
            # TimeLLM-based composite score
            weights = user_preferences.get('weights', {
                'timellm_score': 0.4,  # Higher weight for AI prediction
                'price': 0.25,
                'comfort': 0.2,
                'weather': 0.15
            })
            
            timellm_score = recommendation_score * 10  # Scale to 0-10
            
            composite_score = (
                weights.get('timellm_score', 0.4) * timellm_score +
                weights.get('price', 0.25) * price_score +
                weights.get('comfort', 0.2) * comfort_score +
                weights.get('weather', 0.15) * weather_score
            )
            
            # Enhanced flight result
            flight_result = {
                **flight,
                'predicted_delay_minutes': round(predicted_delay, 1),
                'timellm_recommendation_score': round(timellm_score, 2),
                'weather_conditions': weather_data,
                'scores': {
                    'timellm_ai': round(timellm_score, 2),
                    'comfort': round(comfort_score, 2),
                    'price': round(price_score, 2),
                    'weather': round(weather_score, 2),
                    'composite': round(composite_score, 2)
                },
                'ai_insights': self._generate_ai_insights(
                    flight, predicted_delay, recommendation_score, weather_data
                )
            }
            
            scored_flights.append(flight_result)
        
        # Sort by composite score
        scored_flights.sort(key=lambda x: x['scores']['composite'], reverse=True)
        
        return scored_flights
    
    def _calculate_comfort_score(self, flight_data: Dict, passenger_type: str) -> float:
        """Calculate comfort score (same as before but enhanced)"""
        score = 5.0
        
        aircraft = flight_data.get('aircraft_type', '').upper()
        if any(x in aircraft for x in ['A380', 'B787', 'A350']):
            score += 2.0
        elif any(x in aircraft for x in ['A320', 'B737']):
            score += 1.0
        
        if passenger_type == 'family':
            duration = flight_data.get('duration_minutes', 0)
            if duration < 180:
                score += 1.5
            elif duration > 480:
                score -= 1.5
            
            stops = flight_data.get('stops', 0)
            score -= stops * 0.7
            
        elif passenger_type == 'business':
            stops = flight_data.get('stops', 0)
            if stops == 0:
                score += 2.0
            else:
                score -= stops * 1.2
                
            departure_hour = flight_data.get('departure_hour', 12)
            if 6 <= departure_hour <= 9:
                score += 1.5
                
        elif passenger_type == 'honeymoon':
            airline = flight_data.get('airline', '').upper()
            premium_airlines = ['EMIRATES', 'SINGAPORE', 'CATHAY', 'QATAR', 'ETIHAD']
            if any(x in airline for x in premium_airlines):
                score += 3.0
            
            duration = flight_data.get('duration_minutes', 0)
            if duration > 360:
                score += 1.5
        
        return min(score, 10.0)
    
    def _calculate_price_score(self, flight_data: Dict, price_sensitivity: str) -> float:
        """Calculate price score (enhanced)"""
        price = flight_data.get('price', 0)
        
        if price == 0:
            return 5.0
        
        if price_sensitivity == 'high':
            if price < 200:
                return 10.0
            elif price < 400:
                return 7.5
            elif price < 600:
                return 5.0
            else:
                return 2.0
        elif price_sensitivity == 'medium':
            if price < 300:
                return 8.5
            elif price < 600:
                return 7.0
            elif price < 900:
                return 5.0
            else:
                return 3.0
        else:  # Low sensitivity
            if price < 500:
                return 6.5
            elif price < 1000:
                return 8.5
            else:
                return 10.0
    
    def _generate_ai_insights(self, flight: Dict, predicted_delay: float, 
                            rec_score: float, weather_data: Dict) -> List[str]:
        """Generate AI-powered insights"""
        insights = []
        
        # TimeLLM-based insights
        if rec_score > 0.8:
            insights.append("🤖 AI Model highly recommends this flight based on comprehensive analysis")
        elif rec_score > 0.6:
            insights.append("🤖 AI Model shows positive indicators for this flight option")
        elif rec_score < 0.4:
            insights.append("🤖 AI Model suggests considering alternative options")
        
        # Delay insights
        if predicted_delay < 10:
            insights.append("⏰ Very low delay risk predicted by time series analysis")
        elif predicted_delay > 30:
            insights.append("⚠️ Higher delay risk detected - consider earlier departures")
        
        # Weather insights
        condition = weather_data.get('weather_condition', '')
        if condition in ['Clear', 'Clouds']:
            insights.append("🌤️ Favorable weather conditions support on-time performance")
        elif condition in ['Rain', 'Thunderstorm']:
            insights.append("🌧️ Weather conditions may impact punctuality")
        
        # Smart recommendations
        price = flight.get('price', 0)
        duration = flight.get('duration_minutes', 0)
        
        if price > 0 and duration > 0:
            value_ratio = duration / price
            if value_ratio > 0.5:
                insights.append("💰 Excellent value proposition - good duration for the price")
            elif value_ratio < 0.2:
                insights.append("💸 Premium pricing - expect enhanced service quality")
        
        return insights

# Update main function with enhanced system
def enhanced_main():
    """Enhanced main function with proper TimeLLM fine-tuning"""
    
    # Check authentication first
    if not HF_TOKEN:
        print("❌ Hugging Face token not found!")
        print("Please run: huggingface-cli login")
        print("Or set HF_TOKEN in your environment variables")
        return
    
    print("✅ Hugging Face authentication configured")
    print("🚀 Initializing Enhanced Flight Recommendation System with TimeLLM Fine-tuning...")
    
    # Initialize components
    amadeus_api = AmadeusFlightAPI(AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET)
    enhanced_system = EnhancedFlightRecommendationSystem(OPENWEATHER_API_KEY)
    
    # User input
    user_input = {
        'origin': 'JFK',
        'destination': 'LAX',
        'departure_date': '2025-06-27',
        'passenger_type': 'business',  # family, business, honeymoon
        'price_sensitivity': 'medium',  # high, medium, low
        'weights': {
            'timellm_score': 0.4,  # AI model weight
            'price': 0.25,
            'comfort': 0.2,
            'weather': 0.15
        }
    }
    
    print(f"🔍 Searching flights from {user_input['origin']} to {user_input['destination']}...")
    
    # Get flights
    flights = amadeus_api.search_flights(
        user_input['origin'],
        user_input['destination'],
        user_input['departure_date']
    )
    
    print(f"✅ Found {len(flights)} flights")
    
    # Generate comprehensive training data
    print("📊 Generating comprehensive historical data for TimeLLM training...")
    historical_flights = []
    
    for i in tqdm(range(300), desc="Creating historical flight records"):
        base_flight = flights[i % len(flights)].copy()
        
        # Add realistic variations
        base_flight['price'] = max(100, np.random.normal(400, 150))
        base_flight['duration_minutes'] = max(90, np.random.normal(300, 90))
        base_flight['departure_hour'] = np.random.randint(0, 24)
        base_flight['historical_delay'] = max(0, np.random.normal(20, 15))
        base_flight['passenger_rating'] = np.random.uniform(3.0, 5.0)
        base_flight['on_time_percentage'] = np.random.uniform(0.7, 0.95)
        
        historical_flights.append(base_flight)
    
    # Train enhanced TimeLLM model
    print("🧠 Training Enhanced TimeLLM with proper fine-tuning...")
    enhanced_system.train_model(historical_flights, epochs=3)  # Reduced for demo
    
    # Generate recommendations
    print("🎯 Generating AI-powered flight recommendations...")
    recommendations = enhanced_system.recommend_flights(flights, user_input)
    
    # Display enhanced results
    print("\n" + "="*100)
    print("🛫 ENHANCED AI-POWERED FLIGHT RECOMMENDATIONS (TimeLLM Fine-tuned)")
    print("="*100)
    
    for i, flight in enumerate(recommendations, 1):
        print(f"\n{i}. ✈️ {flight['airline']} - {flight['flight_number']}")
        print(f"   📍 Route: {flight['departure_airport']} → {flight['arrival_airport']}")
        print(f"   🕐 Departure: {flight['departure_time']}")
        print(f"   ⏱️ Duration: {flight['duration_minutes']} minutes")
        print(f"   🔄 Stops: {flight['stops']}")
        print(f"   💰 Price: ${flight['price']:.2f}")
        print(f"   ⏰ AI Predicted Delay: {flight['predicted_delay_minutes']} minutes")
        print(f"   🤖 TimeLLM Recommendation Score: {flight['timellm_recommendation_score']}/10")
        
        print(f"\n   📊 Detailed Scores:")
        for score_type, score_value in flight['scores'].items():
            emoji_map = {
                'timellm_ai': '🤖',
                'comfort': '😌',
                'price': '💰',
                'weather': '🌤️',
                'composite': '🎯'
            }
            emoji = emoji_map.get(score_type, '📈')
            print(f"     {emoji} {score_type.replace('_', ' ').title()}: {score_value}/10")
        
        print(f"\n   🌤️ Weather Conditions:")
        weather = flight['weather_conditions']
        print(f"     🌡️ Temperature: {weather['temperature']}°C")
        print(f"     ☁️ Condition: {weather['weather_condition']}")
        print(f"     💨 Wind Speed: {weather['wind_speed']} m/s")
        
        print(f"\n   🧠 AI Insights:")
        for insight in flight['ai_insights']:
            print(f"     {insight}")
        
        print("-" * 100)
    
    # Highlight top recommendation
    top_flight = recommendations[0]
    print(f"\n🏆 TOP AI RECOMMENDATION: {top_flight['airline']} {top_flight['flight_number']}")
    print(f"🎯 Overall AI Score: {top_flight['scores']['composite']}/10")
    print(f"🤖 TimeLLM Confidence: {top_flight['timellm_recommendation_score']}/10")
    print(f"✨ This flight has been selected by our fine-tuned TimeLLM model as the best match")
    print(f"   for {user_input['passenger_type']} travel with {user_input['price_sensitivity']} price sensitivity.")

if __name__ == "__main__":
    enhanced_main()
