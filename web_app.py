import os
import json
import logging
from flask import Flask, render_template, request, jsonify, send_file, abort, session
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# We'll import ai_hedge_fund later, after potentially setting up the API key

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Constants
API_KEY_FILE = 'api_key.json'
DEFAULT_PORT = 8080

# Check if API key is saved
def get_saved_api_key():
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, 'r') as f:
                data = json.load(f)
                return data.get('api_key')
        except Exception as e:
            logger.error(f"Error reading API key: {str(e)}")
    return None

# Save API key
def save_api_key(api_key):
    try:
        with open(API_KEY_FILE, 'w') as f:
            json.dump({'api_key': api_key}, f)
        return True
    except Exception as e:
        logger.error(f"Error saving API key: {str(e)}")
        return False

# Set API key in environment and update LLM
def set_api_key_env(api_key):
    if api_key:
        os.environ['PPLX_API_KEY'] = api_key
        # Import and update at this point
        try:
            # Import here to avoid circular imports
            import ai_hedge_fund
            return ai_hedge_fund.reload_llm_with_api_key(api_key)
        except Exception as e:
            logger.error(f"Error updating API key: {str(e)}")
            return False
    return False

# Load API key on startup
api_key = get_saved_api_key()
if api_key:
    set_api_key_env(api_key)

# Routes
@app.route('/')
def index():
    """Home page - check if API key is set and show appropriate page"""
    api_key = get_saved_api_key()
    if not api_key:
        return render_template('setup.html')
    else:
        return render_template('index.html')

@app.route('/setup', methods=['POST'])
def setup():
    """Save API key"""
    api_key = request.form.get('api_key')
    if not api_key:
        return jsonify({'success': False, 'message': 'API key is required'}), 400
    
    success = save_api_key(api_key)
    if success:
        set_api_key_env(api_key)
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Failed to save API key'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Run analysis for a ticker"""
    ticker = request.form.get('ticker')
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker symbol is required'}), 400
    
    # Verify API key is set
    api_key = get_saved_api_key()
    if not api_key:
        return jsonify({'success': False, 'message': 'API key is not set or invalid. Please reload the page and set up your API key.'}), 400
    
    # Make sure the API key is loaded into the environment and LLM
    set_api_key_env(api_key)
    
    try:
        # Import here to ensure we have the latest version with the updated API key
        from ai_hedge_fund import run_ai_hedge_fund
        
        # Run analysis without opening browser (handled by web interface)
        logger.info(f"Starting analysis for ticker: {ticker} with API key: {api_key[:5]}...")
        report_path = run_ai_hedge_fund(ticker.upper(), open_browser=False)
        
        if report_path:
            # Store report path in session
            session['last_report'] = report_path
            
            # Extract the filename from path
            filename = os.path.basename(report_path)
            
            logger.info(f"Analysis completed successfully for {ticker}, report: {filename}")
            return jsonify({
                'success': True, 
                'report_path': report_path,
                'filename': filename
            })
        else:
            logger.error(f"Report path was None for {ticker}")
            return jsonify({'success': False, 'message': 'Failed to generate report. Please check the container logs for details.'}), 500
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error analyzing {ticker}: {error_msg}", exc_info=True)
        
        # Provide more user-friendly error messages
        if "401 Authorization Required" in error_msg or "AuthenticationError" in error_msg:
            return jsonify({'success': False, 'message': 'API key authentication failed. Please verify your Perplexity API key is valid and try again.'}), 401
        elif "429" in error_msg or "Too Many Requests" in error_msg:
            return jsonify({'success': False, 'message': 'Rate limit exceeded for the Perplexity API. Please try again later.'}), 429
        else:
            return jsonify({'success': False, 'message': f'Error: {error_msg[:200]}...'}), 500

@app.route('/report/<path:filename>')
def view_report(filename):
    """View a report"""
    reports_dir = os.path.join(os.getcwd(), 'reports')
    try:
        return send_file(os.path.join(reports_dir, filename))
    except Exception as e:
        logger.error(f"Error serving report {filename}: {str(e)}")
        abort(404)

@app.route('/download/<path:filename>')
def download_report(filename):
    """Download a report"""
    reports_dir = os.path.join(os.getcwd(), 'reports')
    try:
        return send_file(
            os.path.join(reports_dir, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading report {filename}: {str(e)}")
        abort(404)

@app.route('/reports')
def list_reports():
    """List all reports"""
    reports_dir = os.path.join(os.getcwd(), 'reports')
    reports = []
    
    try:
        if os.path.exists(reports_dir):
            for f in os.listdir(reports_dir):
                if f.endswith('.html') and f.startswith('ai_hedge_fund_'):
                    # Extract ticker from filename
                    parts = f.replace('ai_hedge_fund_', '').split('_')
                    if len(parts) >= 2:
                        ticker = parts[0]
                        # Extract date and time
                        date_time = '_'.join(parts[1:]).replace('.html', '')
                        
                        reports.append({
                            'filename': f,
                            'ticker': ticker,
                            'date_time': date_time
                        })
            
            # Sort by most recent first
            reports = sorted(reports, key=lambda x: x['date_time'], reverse=True)
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
    
    return jsonify({'reports': reports})

def main():
    """Run the web server"""
    from waitress import serve
    
    port = int(os.environ.get('PORT', DEFAULT_PORT))
    host = '0.0.0.0'  # Listen on all interfaces
    
    logger.info(f"Starting AI Hedge Fund web server on http://{host}:{port}")
    logger.info(f"Press Ctrl+C to stop the server")
    
    serve(app, host=host, port=port)

if __name__ == '__main__':
    main()