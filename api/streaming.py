import json
import time
import threading
import uuid
import os
import logging
import pickle
from flask import Response, stream_with_context, jsonify, session

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Store active analysis jobs (global dict for development, session-based for Vercel)
active_jobs = {}

# Helper functions for job storage that works across Vercel serverless invocations
def save_job(job_id, job_data):
    """Save job data in a way that persists between serverless invocations"""
    is_vercel = os.environ.get('VERCEL') == '1'
    
    if is_vercel:
        # On Vercel, we need to use session storage since in-memory storage doesn't persist
        if 'jobs' not in session:
            session['jobs'] = {}
        
        # Convert any non-serializable data to strings
        serializable_job = {}
        for key, value in job_data.items():
            if key == 'steps':
                # Handle steps specially to preserve structure
                serializable_job[key] = [{
                    'name': s['name'],
                    'status': s['status'],
                    'result': str(s.get('result', ''))[:100] if s.get('result') else None
                } for s in value]
            else:
                # For other fields, convert to string if not serializable
                try:
                    json.dumps({key: value})
                    serializable_job[key] = value
                except (TypeError, OverflowError):
                    serializable_job[key] = str(value)
        
        # Store in session
        session['jobs'][job_id] = serializable_job
        logger.info(f"Saved job {job_id} to session storage")
        return True
    else:
        # In development, use the in-memory dictionary
        active_jobs[job_id] = job_data
        return True

def get_job(job_id):
    """Get job data in a way that works across serverless invocations"""
    is_vercel = os.environ.get('VERCEL') == '1'
    
    if is_vercel:
        # On Vercel, get from session
        if 'jobs' not in session or job_id not in session.get('jobs', {}):
            return None
        return session['jobs'][job_id]
    else:
        # In development, use the in-memory dictionary
        return active_jobs.get(job_id)

def update_job(job_id, updates):
    """Update job data in a way that works across serverless invocations"""
    job = get_job(job_id)
    if not job:
        return False
    
    # Apply updates
    for key, value in updates.items():
        job[key] = value
    
    # Save updated job
    return save_job(job_id, job)

def get_all_jobs():
    """Get all jobs in a way that works across serverless invocations"""
    is_vercel = os.environ.get('VERCEL') == '1'
    
    if is_vercel:
        # On Vercel, get from session
        return session.get('jobs', {})
    else:
        # In development, use the in-memory dictionary
        return active_jobs

def generate_job_id():
    """Generate a unique job ID"""
    return str(uuid.uuid4())

def get_job_status(job_id):
    """Get the status of a job"""
    return get_job(job_id)

def start_analysis_job(ticker, api_key):
    """Start an analysis job in a background thread"""
    job_id = generate_job_id()
    
    # Initialize job status
    job_data = {
        'status': 'initialized',
        'ticker': ticker,
        'api_key': api_key,  # Store API key with the job for later processing
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
    
    # Save initial job status
    save_job(job_id, job_data)
    
    # Log job creation
    logger.info(f"Creating new job {job_id} for ticker {ticker}")
    
    try:
        # For Vercel environment, we need to start the job immediately
        # with a small initial analysis, since background threads don't work well
        if os.environ.get('VERCEL') == '1':
            # Update job status to running
            update_job(job_id, {
                'status': 'running',
                'progress': 1,
                'last_updated': time.time()
            })
            logger.info(f"Setting job {job_id} to running status on Vercel")
            
            # Instead of a real background thread, on Vercel we do just enough work to
            # show the job has started, then rely on the client polling to trigger 
            # additional computation in subsequent serverless invocations
            try:
                # Import without running analysis
                import ai_hedge_fund
                # Just set the API key
                success = ai_hedge_fund.reload_llm_with_api_key(api_key)
                if success:
                    logger.info(f"Successfully set API key for job {job_id}")
                else:
                    logger.warning(f"Failed to set API key for job {job_id}")
                    
                # Update job to signal it's ready for polling
                update_job(job_id, {
                    'status': 'ready',
                    'progress': 2,
                    'last_updated': time.time()
                })
            except Exception as e:
                logger.error(f"Error in initial job setup for {job_id}: {str(e)}")
                update_job(job_id, {
                    'status': 'error',
                    'errors': [f"Failed to initialize job: {str(e)}"],
                    'complete': True,
                    'last_updated': time.time()
                })
        else:
            # For local development, use background thread
            thread = threading.Thread(
                target=run_analysis_in_background,
                args=(job_id, ticker, api_key)
            )
            thread.daemon = True
            thread.start()
            logger.info(f"Background thread started for job {job_id}")
    except Exception as e:
        logger.error(f"Error starting job {job_id}: {str(e)}")
        update_job(job_id, {
            'status': 'error',
            'errors': [f"Failed to start analysis: {str(e)}"],
            'complete': True,
            'last_updated': time.time()
        })
    
    return job_id

def run_analysis_in_background(job_id, ticker, api_key):
    """Run the analysis in a background thread - only used in local development mode"""
    # This function is only used in local development mode, not on Vercel
    if os.environ.get('VERCEL') == '1':
        logger.warning(f"run_analysis_in_background called on Vercel for job {job_id} - this shouldn't happen")
        return
    
    job = get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} not found")
        return
        
    # Update job status
    update_job(job_id, {
        'status': 'running',
        'last_updated': time.time()
    })
    
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
            update_job(job_id, {
                'status': 'analyzing',
                'progress': 5,
                'last_updated': time.time()
            })
            logger.info(f"Starting analysis steps for job {job_id}, ticker: {ticker}")
            
            # Update progress monitoring
            def update_job_progress(step_name, result_value=None):
                step_names = ['market_data', 'sentiment_analysis', 'macro_analysis', 'strategy', 'risk_assessment', 'summary']
                step_idx = next((idx for idx, name in enumerate(step_names) if name == step_name), -1)
                
                if step_idx >= 0:
                    job = get_job(job_id)  # Get latest job state
                    if not job:
                        logger.error(f"Job {job_id} not found during progress update")
                        return
                    
                    # Find the step in our job
                    job_step_idx = next((idx for idx, s in enumerate(job['steps']) if s['name'] == step_name), None)
                    if job_step_idx is not None:
                        # Update the step
                        steps = job['steps']
                        steps[job_step_idx]['status'] = 'complete'
                        if result_value:
                            # Store just a preview of the result (first 100 chars)
                            preview = str(result_value)[:100]
                            if len(str(result_value)) > 100:
                                preview += "..."
                            steps[job_step_idx]['result'] = preview
                        
                        # Calculate progress
                        total_steps = len(step_names)
                        progress = min(90, 5 + (step_idx + 1) * (85 // total_steps))
                        
                        # Update job
                        update_job(job_id, {
                            'steps': steps,
                            'progress': progress,
                            'last_updated': time.time()
                        })
                        logger.info(f"Updated progress for job {job_id}, step: {step_name}, progress: {progress}%")
            
            # Run the analysis step by step - this is safer than monkey-patching sequential_agent
            try:
                # Create an empty result dictionary
                result = {}
                job = get_job(job_id)  # Get latest job state
                
                # Execute each chain individually with progress updates
                update_job(job_id, {
                    'status': 'analyzing',
                    'last_updated': time.time()
                })
                
                # Step 1: Market Data
                steps = job['steps']
                steps[0]['status'] = 'running'
                update_job(job_id, {
                    'steps': steps,
                    'progress': 10,
                    'last_updated': time.time()
                })
                logger.info(f"Executing market_data_chain for job {job_id}, ticker: {ticker.upper()}")
                try:
                    result['market_data'] = market_data_chain.invoke({"ticker": ticker.upper()})
                    logger.info(f"Successfully completed market_data_chain for job {job_id}")
                except Exception as e:
                    logger.error(f"Error in market_data_chain for job {job_id}: {str(e)}")
                    raise Exception(f"Error in market data analysis: {str(e)}")
                update_job_progress('market_data', result['market_data'])
                
                # Step 2: Sentiment Analysis
                job = get_job(job_id)  # Get latest job state
                steps = job['steps']
                steps[1]['status'] = 'running'
                update_job(job_id, {
                    'steps': steps,
                    'progress': 25,
                    'last_updated': time.time()
                })
                logger.info(f"Executing sentiment_chain for job {job_id}")
                result['sentiment_analysis'] = sentiment_chain.invoke({"ticker": ticker.upper()})
                logger.info(f"Successfully completed sentiment_chain for job {job_id}")
                update_job_progress('sentiment_analysis', result['sentiment_analysis'])
                
                # Step 3: Macro Analysis
                job = get_job(job_id)  # Get latest job state
                steps = job['steps']
                steps[2]['status'] = 'running'
                update_job(job_id, {
                    'steps': steps,
                    'progress': 40,
                    'last_updated': time.time()
                })
                logger.info(f"Executing macro_analysis_chain for job {job_id}")
                result['macro_analysis'] = macro_analysis_chain.invoke({"ticker": ticker.upper()})
                logger.info(f"Successfully completed macro_analysis_chain for job {job_id}")
                update_job_progress('macro_analysis', result['macro_analysis'])
                
                # Step 4: Strategy
                job = get_job(job_id)  # Get latest job state
                steps = job['steps']
                steps[3]['status'] = 'running'
                update_job(job_id, {
                    'steps': steps,
                    'progress': 55,
                    'last_updated': time.time()
                })
                logger.info(f"Executing strategy_chain for job {job_id}")
                result['strategy'] = strategy_chain.invoke({
                    "market_data": result['market_data'], 
                    "sentiment_analysis": result['sentiment_analysis'],
                    "macro_analysis": result['macro_analysis']
                })
                logger.info(f"Successfully completed strategy_chain for job {job_id}")
                update_job_progress('strategy', result['strategy'])
                
                # Step 5: Risk Assessment
                job = get_job(job_id)  # Get latest job state
                steps = job['steps']
                steps[4]['status'] = 'running'
                update_job(job_id, {
                    'steps': steps,
                    'progress': 70,
                    'last_updated': time.time()
                })
                logger.info(f"Executing risk_chain for job {job_id}")
                result['risk_assessment'] = risk_chain.invoke({"strategy": result['strategy']})
                logger.info(f"Successfully completed risk_chain for job {job_id}")
                update_job_progress('risk_assessment', result['risk_assessment'])
                
                # Step 6: Summary
                job = get_job(job_id)  # Get latest job state
                steps = job['steps']
                steps[5]['status'] = 'running'
                update_job(job_id, {
                    'steps': steps,
                    'progress': 85,
                    'last_updated': time.time()
                })
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
                update_job(job_id, {
                    'progress': 95,
                    'last_updated': time.time()
                })
                logger.info(f"Generating HTML report for job {job_id}")
                html_content = generate_html_report(result, ticker.upper())
                
                # Create a timestamped filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'ai_hedge_fund_{ticker.upper()}_{timestamp}.html'
                logger.info(f"Generated filename for job {job_id}: {filename}")
                
                # Set the HTML content and filename
                update_job(job_id, {
                    'html_content': html_content,
                    'filename': filename,
                    'progress': 100,
                    'status': 'complete',
                    'complete': True,
                    'last_updated': time.time()
                })
                logger.info(f"Job {job_id} completed successfully")
            
            except Exception as chain_error:
                error_msg = str(chain_error)
                logger.error(f"Error in AI chain execution for job {job_id}: {error_msg}")
                
                # Find the currently running step
                job = get_job(job_id)
                if job:
                    steps = job['steps']
                    current_step_idx = next((i for i, s in enumerate(steps) if s['status'] == 'running'), None)
                    if current_step_idx is not None:
                        steps[current_step_idx]['status'] = 'error'
                        logger.info(f"Marked step {steps[current_step_idx]['name']} as error for job {job_id}")
                
                update_job(job_id, {
                    'errors': job.get('errors', []) + [f"Error in AI chain execution: {error_msg}"],
                    'status': 'error',
                    'steps': steps if 'steps' in locals() else job.get('steps', []),
                    'complete': True,
                    'last_updated': time.time()
                })
                raise
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in analysis for job {job_id}: {error_msg}")
            job = get_job(job_id)
            if job:
                update_job(job_id, {
                    'errors': job.get('errors', []) + [f"Error in analysis: {error_msg}"],
                    'status': 'error',
                    'complete': True,
                    'last_updated': time.time()
                })
            raise
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Fatal error for job {job_id}: {error_msg}")
        job = get_job(job_id)
        if job:
            update_job(job_id, {
                'errors': job.get('errors', []) + [f"Fatal error: {error_msg}"],
                'status': 'error',
                'complete': True,
                'last_updated': time.time()
            })
    finally:
        # Ensure job is marked complete
        job = get_job(job_id)
        if job and not job.get('complete', False):
            update_job(job_id, {
                'complete': True,
                'last_updated': time.time()
            })
            logger.info(f"Job {job_id} marked as complete with status: {job.get('status', 'unknown')}")
            
# Function to process a single step of analysis on demand (for Vercel)
def process_analysis_step(job_id, ticker, api_key):
    """Process a single step of analysis - for Vercel serverless model"""
    logger.info(f"Processing next analysis step for job {job_id}")
    
    job = get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} not found for step processing")
        return False
    
    # Check if job is already complete
    if job.get('complete', False):
        logger.info(f"Job {job_id} is already complete, no steps to process")
        return True
    
    # Set API key
    os.environ['PPLX_API_KEY'] = api_key
    
    try:
        # Import AI hedge fund
        from ai_hedge_fund import (
            market_data_chain, sentiment_chain, macro_analysis_chain, 
            strategy_chain, risk_chain, summary_chain, generate_html_report,
            reload_llm_with_api_key
        )
        
        # Make sure API key is set
        reload_llm_with_api_key(api_key)
        
        # Find the next step to process
        next_step_idx = None
        result_so_far = {}
        for i, step in enumerate(job['steps']):
            if step['status'] == 'pending' or step['status'] == 'running':
                next_step_idx = i
                break
            # For completed steps, retrieve results if available
            if step['status'] == 'complete' and step.get('result'):
                result_so_far[step['name']] = step.get('result')
        
        if next_step_idx is None:
            # All steps are done, generate the final report if needed
            if not job.get('filename'):
                logger.info(f"All steps complete for job {job_id}, generating final report")
                
                # Create a timestamped filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'ai_hedge_fund_{ticker.upper()}_{timestamp}.html'
                
                # We don't have the full result objects, so create a simple report
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>AI Hedge Fund Analysis: {ticker.upper()}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1 {{ color: #333; }}
                        .section {{ margin-bottom: 30px; }}
                        .section h2 {{ color: #0066cc; }}
                    </style>
                </head>
                <body>
                    <h1>Investment Analysis for {ticker.upper()}</h1>
                    
                    <div class="section">
                        <h2>Market Data Analysis</h2>
                        <p>{result_so_far.get('market_data', 'Data not available')}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Sentiment Analysis</h2>
                        <p>{result_so_far.get('sentiment_analysis', 'Data not available')}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Macroeconomic Analysis</h2>
                        <p>{result_so_far.get('macro_analysis', 'Data not available')}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Trading Strategy</h2>
                        <p>{result_so_far.get('strategy', 'Data not available')}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Risk Assessment</h2>
                        <p>{result_so_far.get('risk_assessment', 'Data not available')}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Executive Summary</h2>
                        <p>{result_so_far.get('summary', 'Data not available')}</p>
                    </div>
                    
                    <footer>
                        <p>Generated by AI Hedge Fund - {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </footer>
                </body>
                </html>
                """
                
                update_job(job_id, {
                    'html_content': html_content,
                    'filename': filename,
                    'progress': 100,
                    'status': 'complete',
                    'complete': True,
                    'last_updated': time.time()
                })
                logger.info(f"Final report generated for job {job_id}")
            return True
        
        # Process the next step
        next_step = job['steps'][next_step_idx]
        step_name = next_step['name']
        logger.info(f"Processing step {step_name} for job {job_id}")
        
        # Mark step as running
        steps = job['steps']
        steps[next_step_idx]['status'] = 'running'
        progress = 5 + next_step_idx * 15  # Simple progress calculation
        update_job(job_id, {
            'steps': steps,
            'progress': progress,
            'status': 'analyzing',
            'last_updated': time.time()
        })
        
        # Execute the appropriate chain
        result = None
        try:
            if step_name == 'market_data':
                result = market_data_chain.invoke({"ticker": ticker.upper()})
            elif step_name == 'sentiment_analysis':
                result = sentiment_chain.invoke({"ticker": ticker.upper()})
            elif step_name == 'macro_analysis':
                result = macro_analysis_chain.invoke({"ticker": ticker.upper()})
            elif step_name == 'strategy':
                # Need data from previous steps
                market_data = result_so_far.get('market_data', '')
                sentiment = result_so_far.get('sentiment_analysis', '')
                macro = result_so_far.get('macro_analysis', '')
                result = strategy_chain.invoke({
                    "market_data": market_data,
                    "sentiment_analysis": sentiment,
                    "macro_analysis": macro
                })
            elif step_name == 'risk_assessment':
                strategy = result_so_far.get('strategy', '')
                result = risk_chain.invoke({"strategy": strategy})
            elif step_name == 'summary':
                market_data = result_so_far.get('market_data', '')
                sentiment = result_so_far.get('sentiment_analysis', '')
                macro = result_so_far.get('macro_analysis', '')
                strategy = result_so_far.get('strategy', '')
                risk = result_so_far.get('risk_assessment', '')
                result = summary_chain.invoke({
                    "market_data": market_data,
                    "sentiment_analysis": sentiment,
                    "macro_analysis": macro,
                    "strategy": strategy,
                    "risk_assessment": risk
                })
            
            # Store result and mark step as complete
            steps = job['steps']  # Get latest steps
            steps[next_step_idx]['status'] = 'complete'
            if result:
                # Store preview of result
                preview = str(result)[:100]
                if len(str(result)) > 100:
                    preview += "..."
                steps[next_step_idx]['result'] = preview
            
            progress = 5 + (next_step_idx + 1) * 15  # Simple progress calculation
            update_job(job_id, {
                'steps': steps,
                'progress': progress,
                'last_updated': time.time()
            })
            logger.info(f"Completed step {step_name} for job {job_id}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing step {step_name} for job {job_id}: {error_msg}")
            
            # Mark step as error
            steps = job['steps']
            steps[next_step_idx]['status'] = 'error'
            update_job(job_id, {
                'steps': steps,
                'errors': job.get('errors', []) + [f"Error in {step_name}: {error_msg}"],
                'status': 'error',
                'complete': True,
                'last_updated': time.time()
            })
            return False
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error setting up step processing for job {job_id}: {error_msg}")
        update_job(job_id, {
            'errors': job.get('errors', []) + [f"Error setting up analysis: {error_msg}"],
            'status': 'error', 
            'complete': True,
            'last_updated': time.time()
        })
        return False

def stream_job_status(job_id):
    """Stream the status of a job as server-sent events 
    
    IMPORTANT: On Vercel, this will only send a single update due to serverless limitations
    The client will need to poll using AJAX calls to get continuous updates"""
    
    # Check if we're running on Vercel
    is_vercel = os.environ.get('VERCEL') == '1'
    
    if is_vercel:
        # On Vercel, just return the current status once (client will poll)
        job = get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'})
        
        # Current time to calculate elapsed time
        current_time = time.time()
        
        # If job is not complete and not in error state, try to process the next step
        if (not job.get('complete', False) and 
            job.get('status') != 'error' and 
            job.get('api_key')):
            
            # Only process a step if it's been at least 5 seconds since the last update
            if current_time - job.get('last_updated', 0) > 5:
                try:
                    # Process one step of the job
                    process_analysis_step(job_id, job.get('ticker'), job.get('api_key'))
                    # Get updated job state
                    job = get_job(job_id)
                    if not job:
                        return jsonify({'error': 'Job lost during processing'})
                except Exception as e:
                    logger.error(f"Error processing step during status check: {str(e)}")
        
        # Prepare status data for response
        status_data = {
            'status': job.get('status', 'unknown'),
            'progress': job.get('progress', 0),
            'steps': [{'name': s['name'], 'status': s['status']} for s in job.get('steps', [])],
            'errors': job.get('errors', []),
            'filename': job.get('filename'),
            'complete': job.get('complete', False),
            'elapsed': round(current_time - job.get('started_at', current_time), 1),
            'last_updated': job.get('last_updated', current_time)
        }
        
        # Log debug info about the job
        step_running = next((s['name'] for s in job.get('steps', []) if s['status'] == 'running'), None)
        if job.get('progress', 0) > 0 and step_running:
            logger.info(f"Job {job_id} status: {job.get('status', 'unknown')}, " +
                       f"progress: {job.get('progress', 0)}%, " +
                       f"current step: {step_running}, " +
                       f"elapsed: {round(current_time - job.get('started_at', current_time), 1)}s, " +
                       f"last_updated: {round(current_time - job.get('last_updated', current_time), 1)}s ago")
        
        return jsonify(status_data)
    else:
        # For local development, use server-sent events
        def generate():
            job = get_job(job_id)
            if not job:
                # Job not found
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                return
            
            # Initial status
            yield f"data: {json.dumps({'status': job.get('status', 'unknown'), 'progress': job.get('progress', 0)})}\n\n"
            
            # Stream updates until job is complete
            start_time = time.time()
            timeout = 300  # 5 minutes timeout
            
            while not job.get('complete', False) and (time.time() - start_time) < timeout:
                time.sleep(1)  # Check every second
                
                # Get latest job state
                job = get_job(job_id)
                if not job:
                    yield f"data: {json.dumps({'error': 'Job lost during processing'})}\n\n"
                    return
                
                # Send current status
                current_time = time.time()
                status_data = {
                    'status': job.get('status', 'unknown'),
                    'progress': job.get('progress', 0),
                    'steps': [{'name': s['name'], 'status': s['status']} for s in job.get('steps', [])],
                    'errors': job.get('errors', []),
                    'filename': job.get('filename'),
                    'complete': job.get('complete', False),
                    'elapsed': round(current_time - job.get('started_at', current_time), 1),
                    'last_updated': job.get('last_updated', current_time)
                }
                
                yield f"data: {json.dumps(status_data)}\n\n"
            
            # Final status with latest job state
            job = get_job(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job lost during completion'})}\n\n"
                return
                
            current_time = time.time()
            final_data = {
                'status': job.get('status', 'unknown'),
                'progress': job.get('progress', 0),
                'steps': [{'name': s['name'], 'status': s['status']} for s in job.get('steps', [])],
                'errors': job.get('errors', []),
                'filename': job.get('filename'),
                'complete': True,
                'elapsed': round(current_time - job.get('started_at', current_time), 1),
                'last_updated': job.get('last_updated', current_time)
            }
            
            yield f"data: {json.dumps(final_data)}\n\n"
        
        return Response(stream_with_context(generate()), mimetype="text/event-stream")