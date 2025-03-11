# AI Hedge Fund Analysis Tool

This tool uses LLMs to analyze stocks and provide comprehensive financial reports with trading strategies and risk assessments through a web interface.

## Features

- Web-based interface for easy analysis of any stock ticker
- Market data analysis with up-to-date financial metrics
- Market sentiment analysis based on news and social media
- Macro-economic environment evaluation
- Trading strategy generation with entry/exit points
- Risk assessment with key metrics
- Ability to download reports for offline viewing

## Quick Start (Docker)

The easiest way to run the application is with Docker:

```bash
# Clone the repository
git clone https://github.com/yourusername/AI_HedgeFund.git
cd AI_HedgeFund

# Build and start the application
make build
make start

# View the application logs
make logs
```

Then, open your browser and go to: http://localhost:8080

On first run, you'll be prompted to enter your Perplexity API key. This key will be saved for future use.

## Setup Instructions

### Option 1: Vercel Deployment

The application can be deployed to Vercel for a serverless experience:

1. Fork this repository to your GitHub account
2. Connect your GitHub repository to your Vercel account
3. Configure the following environment variables in the Vercel dashboard:
   - `FLASK_SECRET_KEY`: A secure random string for session encryption
   - `VERCEL`: Set to "1" to enable Vercel mode
   - `PYTHONPATH`: Set to "."
4. Deploy the application

Note: When deployed on Vercel:
- The API key will persist only for the duration of your browser session
- Reports are stored in session memory and will be lost when the session expires
- For persistent storage, use the Docker deployment option below

### Option 2: Docker Deployment (Recommended for persistence)

1. Make sure you have Docker and Docker Compose installed
2. Build the image and start the container:

```bash
# Using make
make build
make start

# Or using docker-compose directly
docker-compose up -d
```

3. Access the web interface at http://localhost:8080
4. Enter your Perplexity API key when prompted (first run only)

With Docker:
- API keys are stored in a local JSON file
- Reports are saved to the mounted `reports` directory
- Data persists between sessions and container restarts

### Option 3: Local Development (Python)

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the web app
python web_app.py
```

Then access the web interface at http://localhost:8080

For Vercel-specific local development:

```bash
pip install -r requirements-vercel.txt
cd api
python web_app_vercel.py
```

## Using the Web Interface

1. Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA)
2. Click "Analyze Stock"
3. Wait for the analysis to complete (this may take 1-2 minutes)
4. View the analysis report directly in the browser
5. Click "Download Report" to save the HTML file to your computer

## Makefile Commands

```bash
# Build the Docker image
make build

# Start the container
make start

# Stop the container
make stop

# Restart the container
make restart

# View container logs
make logs

# Clean up Docker resources
make clean
```

## Docker Configuration

The application runs in a Docker container named "AI_Hedge_Fund" with the following configuration:

- Web interface available on port 8080
- Reports are saved to a Docker volume mapped to your local `./reports` directory
- API key is stored within the container for persistence across restarts
- Container automatically restarts unless explicitly stopped

## Getting a Perplexity API Key

To use this application, you need a Perplexity API key:

1. Go to [https://docs.perplexity.ai/](https://docs.perplexity.ai/)
2. Sign up for an account or log in
3. Navigate to the API section
4. Generate a new API key
5. Copy the key for use in the application

## Disclaimer

This tool is for educational purposes only and does not constitute investment advice. Always consult with a qualified financial advisor before making investment decisions.