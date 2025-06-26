# ✈️ TimeLLM Flight Recommendation System

An AI-driven flight recommendation system using machine learning, deep learning, and real-time APIs to enhance travel planning experiences.

---

## 📦 Requirements

### ✅ Python Version
- **Python 3.9** (as specified in `requirements.txt`)

---

## 🔐 API Keys and Client IDs

Store all credentials in a `api_llm.env` file:

```dotenv
# Amadeus API
AMADEUS_CLIENT_ID=your_client_id_here
AMADEUS_CLIENT_SECRET=your_client_secret_here

# OpenWeather API
OPENWEATHER_API_KEY=your_api_key_here

# Hugging Face (optional)
HUGGINGFACE_API_TOKEN=your_token_here
```

> Load this file using `python-dotenv`.

---

## 🔗 API Setup Guide

### 1. ✈️ **Amadeus API** (Flight Data)
- **Register**: https://developers.amadeus.com
- **Usage**: Free with limited data (10 TPS test / 40 TPS prod)
- **Data**:
  - Flight Inspiration Search
  - Flight Cheapest Date Search
  - Flight Offers Search

---

### 2. 🌦️ **OpenWeather API** (Weather Data)
- **Register**: https://openweathermap.org
- **Usage**: 1,000 free calls/day
- **Data**:
  - Current weather
  - 5-day forecast

---

### 3. 🛫 **OpenSky Network API** (Flight Tracking)
- **Register**: https://opensky-network.org
- **Usage**: Free and open
- **Data**:
  - Real-time aircraft positions
  - Altitude, speed, callsign
- **Note**: No API key required for basic access

---

### 4. 🤗 **Hugging Face API** (Optional for Transformers)
- **Register**: https://huggingface.co
- **Access Token**: Settings > Access Tokens
- **Usage**:
  - Download models
  - Use Inference API

---

## ✅ Testing API Connectivity

Use the provided script to verify your `.env` file is correctly loaded and credentials work:

```bash
python test_apis.py
```

---

## 📂 Project Structure

```
project/
├── test_apis.py
├── llm.env
├── requirements.txt
├── ...
```

---

## 👩‍💻 License

This project is released under the MIT License.
