import os
import sys
import json
import logging
import time
import traceback
from flask import Flask, render_template, request, jsonify, send_file, abort, session, Response
from io import BytesIO
import base64
import uuid
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check if we're running on Vercel
is_vercel = os.environ.get('VERCEL') == '1'

# Constants
DEFAULT_PORT = 8080

# Create Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Set a secret key for sessions
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24).hex())

# Storage backends for Vercel vs local
class LocalStorage:
    """File-based storage for local development"""
    
    @staticmethod
    def get_api_key():
        api_key_file = 'api_key.json'
        if os.path.exists(api_key_file):
            try:
                with open(api_key_file, 'r') as f:
                    data = json.load(f)
                    return data.get('api_key')
            except Exception as e:
                logger.error(f"Error reading API key: {str(e)}")
        return None
    
    @staticmethod
    def save_api_key(api_key):
        try:
            with open('api_key.json', 'w') as f:
                json.dump({'api_key': api_key}, f)
            return True
        except Exception as e:
            logger.error(f"Error saving API key: {str(e)}")
            return False
    
    @staticmethod
    def save_report(ticker, html_content):
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(reports_dir, f'ai_hedge_fund_{ticker}_{timestamp}.html')
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    @staticmethod
    def get_report(filename):
        reports_dir = 'reports'
        file_path = os.path.join(reports_dir, filename)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return f.read()
        return None
    
    @staticmethod
    def list_reports():
        reports = []
        reports_dir = 'reports'
        
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
        
        return reports

class VercelStorage:
    """Session-based storage for Vercel deployment"""
    
    @staticmethod
    def get_api_key():
        return session.get('api_key')
    
    @staticmethod
    def save_api_key(api_key):
        try:
            session['api_key'] = api_key
            return True
        except Exception as e:
            logger.error(f"Error saving API key in session: {str(e)}")
            return False
    
    @staticmethod
    def save_report(ticker, html_content):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'ai_hedge_fund_{ticker}_{timestamp}.html'
        
        # Store in session
        if 'reports' not in session:
            session['reports'] = {}
        
        # Generate unique ID
        report_id = str(uuid.uuid4())
        
        # Store report metadata and content
        session['reports'][report_id] = {
            'filename': filename,
            'ticker': ticker,
            'date_time': timestamp,
            'content': html_content
        }
        
        # Add to index
        if 'report_index' not in session:
            session['report_index'] = []
        
        session['report_index'].append({
            'id': report_id,
            'filename': filename,
            'ticker': ticker,
            'date_time': timestamp
        })
        
        return filename
    
    @staticmethod
    def get_report(filename):
        if 'reports' not in session:
            return None
        
        # Find the report with matching filename
        for report_id, report_data in session['reports'].items():
            if report_data['filename'] == filename:
                return report_data['content']
        
        return None
    
    @staticmethod
    def list_reports():
        if 'report_index' not in session:
            return []
        
        # Sort by most recent first
        reports = sorted(session.get('report_index', []), key=lambda x: x['date_time'], reverse=True)
        return reports

# Select storage based on environment
storage = VercelStorage if is_vercel else LocalStorage

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

# Routes
@app.route('/')
def index():
    """Home page - check if API key is set and show appropriate page"""
    api_key = storage.get_api_key()
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
    
    success = storage.save_api_key(api_key)
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
    api_key = storage.get_api_key()
    if not api_key:
        return jsonify({'success': False, 'message': 'API key is not set or invalid. Please reload the page and set up your API key.'}), 400
    
    # Make sure the API key is loaded into the environment and LLM
    set_api_key_env(api_key)
    
    try:
        # Import here to ensure we have the latest version with the updated API key
        from ai_hedge_fund import generate_html_report
        
        # Run analysis directly, not using the run_ai_hedge_fund function which tries to save to disk
        logger.info(f"Starting analysis for ticker: {ticker} with API key: {api_key[:5]}...")
        
        # Import sequential_agent directly to run the chain
        from ai_hedge_fund import sequential_agent
        
        try:
            result = sequential_agent({"ticker": ticker.upper()})
            
            # Generate HTML content
            html_content = generate_html_report(result, ticker.upper())
            
            # Save the report using the appropriate storage backend
            filename = storage.save_report(ticker.upper(), html_content)
            
            logger.info(f"Analysis completed successfully for {ticker}, report: {filename}")
            return jsonify({
                'success': True, 
                'filename': filename
            })
        except Exception as inner_e:
            error_msg = str(inner_e)
            logger.error(f"Error in sequential_agent or generate_html_report: {error_msg}", exc_info=True)
            return jsonify({'success': False, 'message': f'Error during analysis: {error_msg[:200]}...'}), 500
            
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
    html_content = storage.get_report(filename)
    if html_content:
        return Response(html_content, mimetype='text/html')
    else:
        abort(404)

@app.route('/download/<path:filename>')
def download_report(filename):
    """Download a report"""
    html_content = storage.get_report(filename)
    if html_content:
        buffer = BytesIO(html_content.encode('utf-8'))
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='text/html',
            as_attachment=True,
            download_name=filename
        )
    else:
        abort(404)

@app.route('/reports')
def list_reports():
    """List all reports"""
    reports = storage.list_reports()
    return jsonify({'reports': reports})

@app.route('/debug', methods=['GET'])
def debug():
    """Endpoint for debugging the environment"""
    try:
        # Try importing key modules
        import_results = {}
        
        for module in ['ai_hedge_fund', 'langchain', 'dotenv', 'flask']:
            try:
                __import__(module)
                import_results[module] = "success"
            except Exception as e:
                import_results[module] = f"error: {str(e)}"
        
        # Directory structure
        api_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(api_dir, '..'))
        
        return jsonify({
            "status": "ok",
            "python_version": sys.version,
            "current_dir": os.getcwd(),
            "dir_listing": os.listdir(),
            "imports": import_results,
            "environment": {k: v for k, v in os.environ.items() if k.startswith(('PYTHONPATH', 'VERCEL', 'FLASK'))},
            "session_data": {
                "has_api_key": 'api_key' in session,
                "has_reports": 'reports' in session,
                "report_count": len(session.get('reports', {})) if 'reports' in session else 0
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

# Main function for local development
if __name__ == '__main__':
    # Create a test report for debugging
    if not is_vercel:
        from waitress import serve
        port = int(os.environ.get('PORT', DEFAULT_PORT))
        host = '0.0.0.0'
        logger.info(f"Starting AI Hedge Fund web server on http://{host}:{port}")
        serve(app, host=host, port=port)