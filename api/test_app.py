import os
import sys
import traceback
from flask import Flask, jsonify, request

# Create a minimal test app
test_app = Flask(__name__)

@test_app.route('/')
def home():
    # Collect system information for debugging
    python_version = sys.version
    env_vars = {k: v for k, v in os.environ.items() if k.startswith('PYTHONPATH') or k.startswith('VERCEL')}
    
    return jsonify({
        "status": "ok",
        "message": "API is working!",
        "python_version": python_version,
        "environment": env_vars,
        "cwd": os.getcwd(),
        "ls": os.listdir()
    })

@test_app.route('/api/test')
def test():
    return jsonify({
        "status": "ok",
        "message": "Test endpoint is working!"
    })

@test_app.route('/api/debug', methods=['GET', 'POST'])
def debug():
    """Endpoint for debugging the environment"""
    try:
        # Try importing key modules
        import_results = {}
        
        for module in ['flask', 'jinja2', 'langchain', 'dotenv', 'werkzeug']:
            try:
                __import__(module)
                import_results[module] = "success"
            except Exception as e:
                import_results[module] = f"error: {str(e)}"
        
        # Try to get system path
        sys_path = sys.path
        
        # Check the directory structure
        api_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(api_dir, '..'))
        
        dir_structure = {
            "api_dir": {
                "path": api_dir,
                "files": os.listdir(api_dir) if os.path.exists(api_dir) else "not found"
            },
            "parent_dir": {
                "path": parent_dir,
                "files": os.listdir(parent_dir) if os.path.exists(parent_dir) else "not found"
            }
        }
        
        return jsonify({
            "status": "ok",
            "python_version": sys.version,
            "sys_path": sys_path,
            "imports": import_results,
            "directories": dir_structure,
            "environment": dict(os.environ)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })