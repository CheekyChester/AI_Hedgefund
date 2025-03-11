import json
import time
import threading
import uuid
import os
from flask import Response, stream_with_context

# Store active analysis jobs
active_jobs = {}

def generate_job_id():
    """Generate a unique job ID"""
    return str(uuid.uuid4())

def get_job_status(job_id):
    """Get the status of a job"""
    if job_id not in active_jobs:
        return None
    return active_jobs[job_id]

def start_analysis_job(ticker, api_key):
    """Start an analysis job in a background thread"""
    job_id = generate_job_id()
    
    # Initialize job status
    active_jobs[job_id] = {
        'status': 'initialized',
        'ticker': ticker,
        'progress': 0,
        'steps': [
            {'name': 'market_data', 'status': 'pending', 'result': None},
            {'name': 'sentiment_analysis', 'status': 'pending', 'result': None},
            {'name': 'macro_analysis', 'status': 'pending', 'result': None},
            {'name': 'strategy', 'status': 'pending', 'result': None},
            {'name': 'risk_assessment', 'status': 'pending', 'result': None},
            {'name': 'summary', 'status': 'pending', 'result': None}
        ],
        'errors': [],
        'started_at': time.time(),
        'html_content': None,
        'filename': None,
        'complete': False
    }
    
    # Start a background thread for analysis
    thread = threading.Thread(
        target=run_analysis_in_background,
        args=(job_id, ticker, api_key)
    )
    thread.daemon = True
    thread.start()
    
    return job_id

def run_analysis_in_background(job_id, ticker, api_key):
    """Run the analysis in a background thread"""
    job = active_jobs[job_id]
    job['status'] = 'running'
    
    try:
        # Set API key in environment
        os.environ['PPLX_API_KEY'] = api_key
        
        # Import AI hedge fund here to avoid circular imports
        try:
            from ai_hedge_fund import sequential_agent, generate_html_report
            
            # Update job status
            job['status'] = 'analyzing'
            
            # Override sequential_agent to report progress per chain
            job['progress'] = 5
            
            # Create a capture handler for the sequential agent
            original_invoke = sequential_agent.invoke
            
            def capture_invoke(inputs):
                # This will capture the results from each chain in the sequence
                result = original_invoke(inputs)
                
                # Update the steps based on the result
                step_names = ['market_data', 'sentiment_analysis', 'macro_analysis', 'strategy', 'risk_assessment', 'summary']
                total_steps = len(step_names)
                
                for i, step_name in enumerate(step_names):
                    if step_name in result:
                        step_idx = next((idx for idx, s in enumerate(job['steps']) if s['name'] == step_name), None)
                        if step_idx is not None:
                            job['steps'][step_idx]['status'] = 'complete'
                            # Store just a preview of the result (first 100 chars)
                            job['steps'][step_idx]['result'] = result[step_name][:100] + "..."
                            # Update overall progress
                            job['progress'] = min(90, 5 + (i + 1) * (85 // total_steps))
                
                return result
            
            # Replace the invoke method temporarily
            sequential_agent.invoke = capture_invoke
            
            # Run the analysis
            try:
                result = sequential_agent({"ticker": ticker.upper()})
                
                # Generate HTML content
                job['progress'] = 95
                html_content = generate_html_report(result, ticker.upper())
                
                # Create a timestamped filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'ai_hedge_fund_{ticker.upper()}_{timestamp}.html'
                
                # Set the HTML content and filename
                job['html_content'] = html_content
                job['filename'] = filename
                job['progress'] = 100
                job['status'] = 'complete'
                
            finally:
                # Restore the original invoke method
                sequential_agent.invoke = original_invoke
                
        except Exception as e:
            job['errors'].append(f"Error in analysis: {str(e)}")
            job['status'] = 'error'
            raise
            
    except Exception as e:
        job['errors'].append(f"Fatal error: {str(e)}")
        job['status'] = 'error'
    finally:
        job['complete'] = True

def stream_job_status(job_id):
    """Stream the status of a job as server-sent events"""
    def generate():
        job = active_jobs.get(job_id)
        if not job:
            # Job not found
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return
        
        # Initial status
        yield f"data: {json.dumps({'status': job['status'], 'progress': job['progress']})}\n\n"
        
        # Stream updates until job is complete
        start_time = time.time()
        timeout = 300  # 5 minutes timeout
        
        while not job.get('complete', False) and (time.time() - start_time) < timeout:
            time.sleep(1)  # Check every second
            
            # Send current status
            status_data = {
                'status': job['status'],
                'progress': job['progress'],
                'steps': [{'name': s['name'], 'status': s['status']} for s in job['steps']],
                'errors': job['errors'],
                'filename': job['filename'],
                'complete': job['complete'],
                'elapsed': round(time.time() - job['started_at'], 1)
            }
            
            yield f"data: {json.dumps(status_data)}\n\n"
        
        # Final status
        final_data = {
            'status': job['status'],
            'progress': job['progress'],
            'steps': [{'name': s['name'], 'status': s['status']} for s in job['steps']],
            'errors': job['errors'],
            'filename': job['filename'],
            'complete': True,
            'elapsed': round(time.time() - job['started_at'], 1)
        }
        
        yield f"data: {json.dumps(final_data)}\n\n"
    
    return Response(stream_with_context(generate()), mimetype="text/event-stream")