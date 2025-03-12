import os
import sys
import json
import logging
import time
import traceback
from flask import Flask, render_template, request, jsonify, send_file, abort, session, Response, stream_with_context
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

@app.route('/streaming')
def streaming():
    """Streaming version of the analysis page"""
    api_key = storage.get_api_key()
    if not api_key:
        return render_template('setup.html')
    else:
        return render_template('streaming.html')

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
        # Log request to help with debugging
        logger.info(f"Starting analysis for ticker: {ticker} with API key starting with {api_key[:5]}...")
        
        # Pre-check imports
        import_errors = []
        try:
            from ai_hedge_fund import generate_html_report, sequential_agent
        except ImportError as ie:
            import_errors.append(f"Failed to import from ai_hedge_fund: {str(ie)}")
            logger.error(f"Import error: {str(ie)}", exc_info=True)
            return jsonify({
                'success': False, 
                'message': f"Import error: {str(ie)}",
                'errors': import_errors
            }), 500
        
        # Check environment variables
        environment_info = {
            'has_api_key': bool(api_key),
            'path': sys.path,
            'cwd': os.getcwd(),
            'python_version': sys.version,
            'env_keys': list(os.environ.keys())
        }
        logger.info(f"Environment info: {environment_info}")
        
        # Run the analysis with a simplified approach
        try:
            # First, create a mock response to test if basic functionality works
            mock_html = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Test Report for {ticker.upper()}</title></head>
            <body>
                <h1>Test Report for {ticker.upper()}</h1>
                <p>This is a simplified test report to check if basic functionality works.</p>
                <p>API Key: {api_key[:3]}...{api_key[-3:]}</p>
            </body>
            </html>
            """
            
            # Save this test report
            test_filename = f"test_report_{ticker.upper()}_{time.strftime('%Y%m%d_%H%M%S')}.html"
            storage.save_report(ticker.upper(), mock_html)
            
            # Now try the real analysis
            logger.info(f"Starting real analysis for {ticker.upper()}...")
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
            
            # Check if the error is due to timeout
            if "timed out" in error_msg.lower() or "deadline" in error_msg.lower():
                return jsonify({
                    'success': False, 
                    'message': 'Analysis timed out. Vercel has a 10-second execution limit for serverless functions. Try using the Docker version for unlimited analysis time.',
                    'error': error_msg[:200]
                }), 504
                
            # Return the test report if we at least got that far
            if 'test_filename' in locals():
                return jsonify({
                    'success': False, 
                    'message': f'Error during analysis, but created a test report: {test_filename}. Error: {error_msg[:200]}',
                    'filename': test_filename,
                    'is_test': True
                }), 500
            else:
                return jsonify({
                    'success': False, 
                    'message': f'Error during analysis: {error_msg[:200]}',
                    'error_details': traceback.format_exc()[:500]
                }), 500
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error analyzing {ticker}: {error_msg}", exc_info=True)
        
        # Provide more user-friendly error messages
        if "401 Authorization Required" in error_msg or "AuthenticationError" in error_msg:
            return jsonify({'success': False, 'message': 'API key authentication failed. Please verify your Perplexity API key is valid and try again.'}), 401
        elif "429" in error_msg or "Too Many Requests" in error_msg:
            return jsonify({'success': False, 'message': 'Rate limit exceeded for the Perplexity API. Please try again later.'}), 429
        elif "timed out" in error_msg.lower() or "deadline" in error_msg.lower():
            return jsonify({
                'success': False, 
                'message': 'Analysis timed out. Vercel has a 10-second execution limit for serverless functions. Try using the Docker version for unlimited analysis time.',
                'error': error_msg[:200]
            }), 504
        else:
            return jsonify({
                'success': False, 
                'message': f'Error: {error_msg[:200]}',
                'traceback': traceback.format_exc()[:500]
            }), 500

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
        
        for module in ['ai_hedge_fund', 'langchain', 'langchain_community', 'dotenv', 'flask', 'openai', 'typing_inspect']:
            try:
                __import__(module)
                import_results[module] = "success"
            except Exception as e:
                import_results[module] = f"error: {str(e)}"
        
        # Try importing specifically from ai_hedge_fund
        ai_hedge_fund_details = {}
        try:
            import ai_hedge_fund
            ai_hedge_fund_details = {
                "imported": True,
                "version": getattr(ai_hedge_fund, "__version__", "Not set"),
                "available_functions": [f for f in dir(ai_hedge_fund) if not f.startswith("_")],
                "has_sequential_agent": hasattr(ai_hedge_fund, "sequential_agent"),
                "has_generate_html_report": hasattr(ai_hedge_fund, "generate_html_report"),
                "file_path": getattr(ai_hedge_fund, "__file__", "Unknown")
            }
        except Exception as e:
            ai_hedge_fund_details = {
                "imported": False,
                "error": str(e)
            }
        
        # Directory structure
        api_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(api_dir, '..'))
        
        # Check for API key and test it
        api_key = storage.get_api_key()
        api_key_status = "Not set"
        if api_key:
            api_key_status = f"Set (length: {len(api_key)}, starts with: {api_key[:3]}...)"
            # Try setting it
            try:
                set_api_key_env(api_key)
                api_key_status += ", Successfully set in environment"
            except Exception as e:
                api_key_status += f", Error setting: {str(e)}"
        
        # Memory usage info (only available on some systems)
        memory_info = {}
        try:
            import resource
            memory_info = {
                "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "current_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_idrss
            }
        except (ImportError, AttributeError):
            memory_info = {"error": "resource module not available"}
        
        # Check jobs from streaming module
        jobs_info = []
        try:
            from api.streaming import active_jobs
            for job_id, job in active_jobs.items():
                current_time = time.time()
                jobs_info.append({
                    "job_id": job_id,
                    "ticker": job.get("ticker", "unknown"),
                    "status": job.get("status", "unknown"),
                    "progress": job.get("progress", 0),
                    "complete": job.get("complete", False),
                    "elapsed_seconds": round(current_time - job.get("started_at", current_time), 1),
                    "last_updated_seconds_ago": round(current_time - job.get("last_updated", current_time), 1),
                    "steps": [{"name": s["name"], "status": s["status"]} for s in job.get("steps", [])],
                    "has_errors": len(job.get("errors", [])) > 0,
                    "error_count": len(job.get("errors", []))
                })
        except Exception as e:
            jobs_info = [{"error": f"Error getting jobs: {str(e)}"}]
        
        return jsonify({
            "status": "ok",
            "time": time.time(),
            "python_version": sys.version,
            "current_dir": os.getcwd(),
            "dir_listing": os.listdir(),
            "imports": import_results,
            "ai_hedge_fund_details": ai_hedge_fund_details,
            "environment": {k: v for k, v in os.environ.items() if k.startswith(('PYTHONPATH', 'VERCEL', 'FLASK', 'PPLX'))},
            "session_data": {
                "has_api_key": 'api_key' in session,
                "api_key_status": api_key_status,
                "has_reports": 'reports' in session,
                "report_count": len(session.get('reports', {})) if 'reports' in session else 0,
                "reports_index": session.get('report_index', [])[:5] if 'report_index' in session else []
            },
            "active_jobs": jobs_info,
            "sys_path": sys.path,
            "memory_info": memory_info,
            "vercel_info": {
                "is_vercel": is_vercel,
                "region": os.environ.get("VERCEL_REGION", "unknown"),
                "environment": os.environ.get("VERCEL_ENV", "unknown"),
                "project_id": os.environ.get("VERCEL_PROJECT_ID", "unknown")
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all active streaming jobs"""
    try:
        from api.streaming import active_jobs
        
        jobs_list = []
        current_time = time.time()
        
        for job_id, job in active_jobs.items():
            jobs_list.append({
                "job_id": job_id,
                "ticker": job.get("ticker", "unknown"),
                "status": job.get("status", "unknown"),
                "progress": job.get("progress", 0),
                "complete": job.get("complete", False),
                "elapsed_seconds": round(current_time - job.get("started_at", current_time), 1),
                "last_updated_seconds_ago": round(current_time - job.get("last_updated", current_time), 1),
                "error_count": len(job.get("errors", [])),
                "steps": [{"name": s["name"], "status": s["status"]} for s in job.get("steps", [])]
            })
        
        # Sort by most recent first
        jobs_list.sort(key=lambda x: x["elapsed_seconds"])
        
        return jsonify({
            "success": True,
            "job_count": len(jobs_list),
            "jobs": jobs_list
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# Streaming routes
from api.streaming import start_analysis_job, stream_job_status, get_job_status

@app.route('/start-streaming-analysis', methods=['POST'])
def start_streaming_analysis():
    """Start a streaming analysis job"""
    ticker = request.form.get('ticker')
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker symbol is required'}), 400
    
    # Verify API key is set
    api_key = storage.get_api_key()
    if not api_key:
        return jsonify({'success': False, 'message': 'API key is not set or invalid. Please reload the page and set up your API key.'}), 400
    
    try:
        # Start the job
        job_id = start_analysis_job(ticker, api_key)
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Analysis started for {ticker}'
        })
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error starting analysis: {str(e)}'
        }), 500

@app.route('/stream-job-status/<job_id>')
def stream_status(job_id):
    """Stream the status of a job"""
    return stream_job_status(job_id)

@app.route('/job-result/<job_id>')
def job_result(job_id):
    """Get the final HTML result of a job"""
    job = get_job_status(job_id)
    
    if not job:
        return jsonify({'success': False, 'message': 'Job not found'}), 404
    
    if not job.get('complete'):
        return jsonify({'success': False, 'message': 'Job not complete yet'}), 400
    
    if job.get('status') == 'error':
        return jsonify({
            'success': False, 
            'message': 'Job failed', 
            'errors': job.get('errors', [])
        }), 500
    
    if job.get('html_content') and job.get('filename'):
        # Save the report in session storage
        filename = job.get('filename')
        html_content = job.get('html_content')
        storage.save_report(job.get('ticker', 'UNKNOWN'), html_content)
        
        return jsonify({
            'success': True,
            'filename': filename
        })
    
    return jsonify({
        'success': False,
        'message': 'No report generated'
    }), 500

# Main function for local development
if __name__ == '__main__':
    # Create a test report for debugging
    if not is_vercel:
        from waitress import serve
        port = int(os.environ.get('PORT', DEFAULT_PORT))
        host = '0.0.0.0'
        logger.info(f"Starting AI Hedge Fund web server on http://{host}:{port}")
        serve(app, host=host, port=port)