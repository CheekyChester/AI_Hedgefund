import json
import time
import threading
import uuid
import os
import logging
from flask import Response, stream_with_context, jsonify

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        'complete': False,
        'last_updated': time.time()  # Add timestamp for debugging
    }
    
    # Log job creation
    logger.info(f"Creating new job {job_id} for ticker {ticker}")
    
    try:
        # Start a background thread for analysis
        thread = threading.Thread(
            target=run_analysis_in_background,
            args=(job_id, ticker, api_key)
        )
        thread.daemon = True
        thread.start()
        logger.info(f"Background thread started for job {job_id}")
        
        # For Vercel, add a status check to make the job immediately visible
        if os.environ.get('VERCEL') == '1':
            active_jobs[job_id]['status'] = 'running'
            active_jobs[job_id]['progress'] = 1
            logger.info(f"Initial status set for job {job_id} on Vercel")
    except Exception as e:
        logger.error(f"Error starting thread for job {job_id}: {str(e)}")
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['errors'].append(f"Failed to start analysis: {str(e)}")
        active_jobs[job_id]['complete'] = True
    
    return job_id

def run_analysis_in_background(job_id, ticker, api_key):
    """Run the analysis in a background thread"""
    if job_id not in active_jobs:
        logger.error(f"Job {job_id} not found in active_jobs dictionary")
        return
        
    job = active_jobs[job_id]
    job['status'] = 'running'
    job['last_updated'] = time.time()
    
    logger.info(f"Starting background analysis for job {job_id}, ticker: {ticker}")
    
    try:
        # Set API key in environment
        os.environ['PPLX_API_KEY'] = api_key
        logger.info(f"Set API key in environment for job {job_id}")
        
        # Import AI hedge fund here to avoid circular imports
        try:
            logger.info(f"Importing AI hedge fund modules for job {job_id}")
            from ai_hedge_fund import (
                market_data_chain, sentiment_chain, macro_analysis_chain, 
                strategy_chain, risk_chain, summary_chain, generate_html_report,
                reload_llm_with_api_key
            )
            
            # Properly set the API key in the LLM
            logger.info(f"Setting Perplexity API key (starts with {api_key[:5]}...) for job {job_id}")
            success = reload_llm_with_api_key(api_key)
            
            if not success:
                logger.error(f"Failed to set API key in LLM for job {job_id}")
                raise Exception("Failed to set API key in LLM")
                
            # Double-check environment variable was set
            if os.environ.get('PPLX_API_KEY') != api_key:
                os.environ['PPLX_API_KEY'] = api_key
                logger.warning(f"Had to manually set API key in environment for job {job_id}")
            
            # Update job status
            job['status'] = 'analyzing'
            job['last_updated'] = time.time()
            logger.info(f"Starting analysis steps for job {job_id}, ticker: {ticker}")
            
            # Override sequential_agent to report progress per chain
            job['progress'] = 5
            
            # Update progress monitoring
            def update_job_progress(step_name, result_value=None):
                step_names = ['market_data', 'sentiment_analysis', 'macro_analysis', 'strategy', 'risk_assessment', 'summary']
                step_idx = next((idx for idx, name in enumerate(step_names) if name == step_name), -1)
                
                if step_idx >= 0:
                    # Find the step in our job
                    job_step_idx = next((idx for idx, s in enumerate(job['steps']) if s['name'] == step_name), None)
                    if job_step_idx is not None:
                        job['steps'][job_step_idx]['status'] = 'complete'
                        if result_value:
                            # Store just a preview of the result (first 100 chars)
                            preview = str(result_value)[:100]
                            if len(str(result_value)) > 100:
                                preview += "..."
                            job['steps'][job_step_idx]['result'] = preview
                        
                        # Update overall progress (each step is worth ~15% of progress)
                        total_steps = len(step_names)
                        job['progress'] = min(90, 5 + (step_idx + 1) * (85 // total_steps))
                        job['last_updated'] = time.time()
                        logger.info(f"Updated progress for job {job_id}, step: {step_name}, progress: {job['progress']}%")
            
            # Run the analysis step by step - this is safer than monkey-patching sequential_agent
            try:
                # Create an empty result dictionary
                result = {}
                
                # Execute each chain individually with progress updates
                job['status'] = 'analyzing'
                job['last_updated'] = time.time()
                
                # Step 1: Market Data
                job['steps'][0]['status'] = 'running'
                job['progress'] = 10
                job['last_updated'] = time.time()
                logger.info(f"Executing market_data_chain for job {job_id}, ticker: {ticker.upper()}")
                try:
                    result['market_data'] = market_data_chain.invoke({"ticker": ticker.upper()})
                    logger.info(f"Successfully completed market_data_chain for job {job_id}")
                except Exception as e:
                    logger.error(f"Error in market_data_chain for job {job_id}: {str(e)}")
                    raise Exception(f"Error in market data analysis: {str(e)}")
                update_job_progress('market_data', result['market_data'])
                
                # Step 2: Sentiment Analysis
                job['steps'][1]['status'] = 'running'
                job['progress'] = 25
                job['last_updated'] = time.time()
                logger.info(f"Executing sentiment_chain for job {job_id}")
                result['sentiment_analysis'] = sentiment_chain.invoke({"ticker": ticker.upper()})
                logger.info(f"Successfully completed sentiment_chain for job {job_id}")
                update_job_progress('sentiment_analysis', result['sentiment_analysis'])
                
                # Step 3: Macro Analysis
                job['steps'][2]['status'] = 'running'
                job['progress'] = 40
                job['last_updated'] = time.time()
                logger.info(f"Executing macro_analysis_chain for job {job_id}")
                result['macro_analysis'] = macro_analysis_chain.invoke({"ticker": ticker.upper()})
                logger.info(f"Successfully completed macro_analysis_chain for job {job_id}")
                update_job_progress('macro_analysis', result['macro_analysis'])
                
                # Step 4: Strategy
                job['steps'][3]['status'] = 'running'
                job['progress'] = 55
                job['last_updated'] = time.time()
                logger.info(f"Executing strategy_chain for job {job_id}")
                result['strategy'] = strategy_chain.invoke({
                    "market_data": result['market_data'], 
                    "sentiment_analysis": result['sentiment_analysis'],
                    "macro_analysis": result['macro_analysis']
                })
                logger.info(f"Successfully completed strategy_chain for job {job_id}")
                update_job_progress('strategy', result['strategy'])
                
                # Step 5: Risk Assessment
                job['steps'][4]['status'] = 'running'
                job['progress'] = 70
                job['last_updated'] = time.time()
                logger.info(f"Executing risk_chain for job {job_id}")
                result['risk_assessment'] = risk_chain.invoke({"strategy": result['strategy']})
                logger.info(f"Successfully completed risk_chain for job {job_id}")
                update_job_progress('risk_assessment', result['risk_assessment'])
                
                # Step 6: Summary
                job['steps'][5]['status'] = 'running'
                job['progress'] = 85
                job['last_updated'] = time.time()
                logger.info(f"Executing summary_chain for job {job_id}")
                result['summary'] = summary_chain.invoke({
                    "market_data": result['market_data'], 
                    "sentiment_analysis": result['sentiment_analysis'],
                    "macro_analysis": result['macro_analysis'],
                    "strategy": result['strategy'],
                    "risk_assessment": result['risk_assessment']
                })
                logger.info(f"Successfully completed summary_chain for job {job_id}")
                update_job_progress('summary', result['summary'])
                
                # Generate HTML content
                job['progress'] = 95
                job['last_updated'] = time.time()
                logger.info(f"Generating HTML report for job {job_id}")
                html_content = generate_html_report(result, ticker.upper())
                
                # Create a timestamped filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'ai_hedge_fund_{ticker.upper()}_{timestamp}.html'
                logger.info(f"Generated filename for job {job_id}: {filename}")
                
                # Set the HTML content and filename
                job['html_content'] = html_content
                job['filename'] = filename
                job['progress'] = 100
                job['status'] = 'complete'
                job['last_updated'] = time.time()
                logger.info(f"Job {job_id} completed successfully")
            
            except Exception as chain_error:
                error_msg = str(chain_error)
                logger.error(f"Error in AI chain execution for job {job_id}: {error_msg}")
                job['errors'].append(f"Error in AI chain execution: {error_msg}")
                job['status'] = 'error'
                job['last_updated'] = time.time()
                # Set status of current step to error
                current_step = next((s for s in job['steps'] if s['status'] == 'running'), None)
                if current_step:
                    current_step['status'] = 'error'
                    logger.info(f"Marked step {current_step['name']} as error for job {job_id}")
                raise
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in analysis for job {job_id}: {error_msg}")
            job['errors'].append(f"Error in analysis: {error_msg}")
            job['status'] = 'error'
            job['last_updated'] = time.time()
            raise
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Fatal error for job {job_id}: {error_msg}")
        job['errors'].append(f"Fatal error: {error_msg}")
        job['status'] = 'error'
        job['last_updated'] = time.time()
    finally:
        job['complete'] = True
        job['last_updated'] = time.time()
        logger.info(f"Job {job_id} marked as complete with status: {job['status']}")

def stream_job_status(job_id):
    """Stream the status of a job as server-sent events 
    
    IMPORTANT: On Vercel, this will only send a single update due to serverless limitations
    The client will need to poll using AJAX calls to get continuous updates"""
    
    # Check if we're running on Vercel
    is_vercel = os.environ.get('VERCEL') == '1'
    
    if is_vercel:
        # On Vercel, just return the current status once (client will poll)
        job = active_jobs.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'})
        
        status_data = {
            'status': job['status'],
            'progress': job['progress'],
            'steps': [{'name': s['name'], 'status': s['status']} for s in job['steps']],
            'errors': job['errors'],
            'filename': job['filename'],
            'complete': job['complete'],
            'elapsed': round(time.time() - job['started_at'], 1),
            'last_updated': job.get('last_updated', time.time())
        }
        
        # Log debug info about the job
        if job['progress'] > 0 and 'market_data' in [s['name'] for s in job['steps'] if s['status'] == 'running']:
            logger.info(f"Job {job_id} status: {job['status']}, progress: {job['progress']}%, " +
                       f"elapsed: {round(time.time() - job['started_at'], 1)}s, " +
                       f"last_updated: {round(time.time() - job.get('last_updated', time.time()), 1)}s ago")
        
        return jsonify(status_data)
    else:
        # For local development, use server-sent events
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