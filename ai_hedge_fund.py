# -------------------- Libraries and Modules -------------
from langchain import PromptTemplate, LLMChain
from langchain_community.chat_models.perplexity import ChatPerplexity
from langchain.chains import SequentialChain
from datetime import datetime
import jinja2
import webbrowser
import os
import logging
from dotenv import load_dotenv

# -------------------- Initialization --------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PPLX_API_KEY = os.getenv("PPLX_API_KEY", "dummy_key_placeholder")

# Model initialization with dummy key (will be replaced later)
llm = ChatPerplexity(model="sonar-reasoning", 
                     temperature=0.5, 
                     pplx_api_key="dummy_key_placeholder")

# Global variables to store chain references
chains = []

# Function to reload the API key (used by web_app.py)
def reload_llm_with_api_key(api_key):
    """Updates the LLM with a new API key"""
    global llm
    global PPLX_API_KEY
    global chains
    
    # Update the environment variable
    os.environ["PPLX_API_KEY"] = api_key
    PPLX_API_KEY = api_key
    
    # Create a new LLM instance with the provided API key
    llm = ChatPerplexity(model="sonar-reasoning", 
                         temperature=0.5, 
                         pplx_api_key=api_key)
    
    # Update the API key in the chain definitions
    if chains:
        for chain in chains:
            if hasattr(chain, 'llm'):
                chain.llm = llm
            
    return True

# -------------------- Chain Definitions --------------------
# 1. Market Data Analyst: Retrieve financial data
market_data_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["ticker"],
        template=(
            "📈 For {ticker}, provide detailed and up-to-date financial data. Include current stock price, "
            "Market capitalisation, Price to earnings ratio P/E, Earnings per share EPS, the last four quarters revenue and %net margin,"
            "price performance (last month and last 12-months), RSI, Sharpe Ratio, and any other relevant metrics or ratios.\n\n"
            "IMPORTANT: Present the data in a structured format:\n"
            "1. First, write a brief overview paragraph (max 3 sentences) about the stock\n"
            "2. Then, present the core financial metrics in an HTML table format with the following structure:\n"
            "```html\n"
            "<table class='financial-metrics'>\n"
            "  <tr><th>Metric</th><th>Value</th></tr>\n"
            "  <tr><td>Current Price</td><td>$XXX.XX</td></tr>\n"
            "  <tr><td>Market Cap</td><td>$XX.XX Billion</td></tr>\n"
            "  <tr><td>P/E Ratio</td><td>XX.XX</td></tr>\n"
            "  <!-- Add all relevant metrics here -->\n"
            "</table>\n"
            "```\n"
            "3. Present quarterly revenue data in a separate HTML table with this format:\n"
            "```html\n"
            "<table class='quarterly-data'>\n"
            "  <tr><th>Quarter</th><th>Revenue</th><th>Net Margin</th></tr>\n"
            "  <tr><td>Q1 20XX</td><td>$XXX Million</td><td>XX.X%</td></tr>\n"
            "  <!-- Add all quarters -->\n"
            "</table>\n"
            "```\n"
            "4. After the tables, you may add 2-3 bullet points with additional context about the financial performance.\n\n"
            "Do NOT include any reasoning, thinking process, or step-by-step analysis of how you arrived at your conclusions. "
            "Present information directly as facts with proper HTML table formatting that will render correctly on a web page."
        )
    ),
    output_key="market_data"
)

# 2. Sentiment Analyst: Analyze news and social media sentiment
sentiment_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["ticker"],
        template=(
            "📰 For {ticker}, analyze recent news articles, financial analyst recommendations and sentiment, social media posts, and expert commentary. "
            "Summarize the overall sentiment, highlight any key events, and note emerging trends that may impact the stock.\n\n"
            "IMPORTANT: Present only the final polished analysis. Do NOT include any reasoning, thinking process, or step-by-step analysis "
            "of how you arrived at your conclusions. Do NOT include phrases like 'looking at the data' or 'based on my analysis'. "
            "Present information directly as facts with proper formatting."
        )
    ),
    output_key="sentiment_analysis"
)

# 3. Macro-Economic Analyst: Evaluate macro-economic conditions
macro_analysis_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["ticker"],
        template=(
            "🌐 For {ticker}, analyze the current macro-economic environment. "
            "Include key indicators such as GDP growth, inflation rates, interest rates, money supply, stock market performance, unemployment trends, "
            "and central bank policies. Summarize how these factors could impact the overall market and the asset.\n\n"
            "IMPORTANT: Present only the final polished analysis. Do NOT include any reasoning, thinking process, or step-by-step analysis "
            "of how you arrived at your conclusions. Do NOT include phrases like 'looking at the data' or 'based on my analysis'. "
            "Present information directly as facts with proper formatting."
        )
    ),
    output_key="macro_analysis"
)

# 4. Quantitative Strategist: Develop a trading strategy
strategy_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["market_data", "sentiment_analysis", "macro_analysis"],
        template=(
            "📊 Using the detailed market data:\n{market_data}\n"
            "the sentiment analysis:\n{sentiment_analysis}\n"
            "and the macro-economic analysis:\n{macro_analysis}\n"
            "develop a sophisticated trading strategy. Outline specify entry and exit points, "
            "detail risk management measures such as stoploss levels, and provide estimated expected returns. "
            "If applicable, incorporate algorithmic signals and technical indicators.\n\n"
            "IMPORTANT: Present only the final polished strategy. Do NOT include any reasoning, thinking process, or step-by-step analysis "
            "of how you arrived at your conclusions. Do NOT include phrases like 'looking at the data' or 'based on my analysis'. "
            "Present your strategy directly as a professional trading plan with proper formatting."
        )
    ),
    output_key="strategy"
)

# 5. Risk Manager: Assess the strategy's risk
risk_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["strategy"],
        template=(
            "⚠️ Evaluate the following trading strategy:\n{strategy}\n"
            "Identify potential risks such as market volatility, liquidity issues, or unexpected market events. "
            "Summarize your risk assessment in 4 concise bullet points, and state in the final bullet point whether the strategy meets an acceptable risk tolerance.\n\n"
            "IMPORTANT: Present only the final polished risk assessment. Do NOT include any reasoning, thinking process, or step-by-step analysis "
            "of how you arrived at your conclusions. Do NOT include phrases like 'looking at the data' or 'based on my analysis'. "
            "Present your assessment directly as professional risk points with proper formatting."
        )
    ),
    output_key="risk_assessment"
)

# 6. Summary: Summarise the analysis
summary_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["market_data", "sentiment_analysis", "macro_analysis","strategy","risk_assessment"],
        template=(
            "💰 Using the market data:\n{market_data}\n"
            "the sentiment analysis:\n{sentiment_analysis}\n"
            "the macro-economic analysis:\n{macro_analysis}\n"
            "the trading strategy:\n{strategy}\n"
            "the risk assessment:\n{risk_assessment}\n"
            "Produce a concise executive summary and 5 bullet points that encapsulate the key insights and recommendations from the analysis.\n\n"
            "IMPORTANT: Present only the final polished summary. Do NOT include any reasoning, thinking process, or step-by-step analysis "
            "of how you arrived at your conclusions. Do NOT include phrases like 'looking at the data' or 'based on my analysis'. "
            "Present your executive summary directly as a professional document with proper formatting."
        )
    ),
    output_key="summary"
)
# -------------------- Sequential Orchestration --------------------
sequential_agent = SequentialChain(
    chains=[market_data_chain, sentiment_chain, macro_analysis_chain, strategy_chain, risk_chain, summary_chain],
    input_variables=["ticker"],
    output_variables=["market_data", "sentiment_analysis", "macro_analysis", "strategy", "risk_assessment", "summary"],
    verbose=True
)

# -------------------- Run the Analysis --------------------
def generate_html_report(result: dict, ticker: str) -> str:
    """Generate an HTML report from the analysis results with improved formatting and styling"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create HTML template with improved styling
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Hedge Fund Analysis - {{ticker}}</title>
        <style>
            :root {
                --primary-color: #2c3e50;
                --secondary-color: #3498db;
                --accent-color: #e74c3c;
                --light-bg: #f8f9fa;
                --border-color: #dee2e6;
                --text-color: #333;
                --muted-color: #6c757d;
            }
            
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                color: var(--text-color);
                line-height: 1.6;
                background-color: #fff;
            }
            
            .header { 
                background-color: var(--light-bg); 
                padding: 30px; 
                border-radius: 8px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .section { 
                margin: 30px 0; 
                padding: 25px; 
                border: 1px solid var(--border-color); 
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                transition: transform 0.2s ease;
            }
            
            .section:hover {
                transform: translateY(-5px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            
            h1 { 
                color: var(--primary-color); 
                margin-top: 0;
                font-size: 2.5rem;
            }
            
            h2 { 
                color: var(--primary-color);
                font-size: 1.8rem;
                border-bottom: 2px solid var(--secondary-color);
                padding-bottom: 8px;
                margin-top: 0;
            }
            
            .timestamp { 
                color: var(--muted-color); 
                font-size: 0.9em;
                margin-bottom: 20px;
            }
            
            .icon { 
                font-size: 1.6em; 
                margin-right: 10px;
                vertical-align: middle;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid var(--border-color);
            }
            
            th {
                background-color: var(--light-bg);
                font-weight: bold;
            }
            
            tr:hover {
                background-color: rgba(0,0,0,0.02);
            }
            
            ul, ol {
                padding-left: 20px;
                margin: 15px 0;
                list-style-position: outside;
            }
            
            li {
                margin-bottom: 8px;
                line-height: 1.5;
                padding-left: 5px;
            }
            
            /* Fix for nested lists */
            ul ul, ol ol, ul ol, ol ul {
                margin: 8px 0 8px 20px;
            }
            
            /* Clear any floating elements to avoid diagonal formatting */
            ul:after, ol:after {
                content: "";
                display: table;
                clear: both;
            }
            
            .risk-indicator {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 16px;
                font-weight: bold;
                margin-top: 15px;
                background-color: #f8d7da;
                color: #721c24;
            }
            
            footer {
                margin-top: 40px;
                text-align: center;
                font-size: 0.9em;
                color: var(--muted-color);
                padding: 20px;
                border-top: 1px solid var(--border-color);
            }
            
            /* Improved markdown rendering */
            p {
                margin-bottom: 16px;
            }
            
            code {
                background-color: var(--light-bg);
                padding: 2px 4px;
                border-radius: 4px;
                font-family: Consolas, Monaco, 'Andale Mono', monospace;
            }
            
            blockquote {
                border-left: 4px solid var(--secondary-color);
                margin-left: 0;
                padding-left: 16px;
                color: var(--muted-color);
            }
            
            /* Tables styling */
            .financial-metrics, .quarterly-data, .data-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 3px rgba(0,0,0,0.1);
            }
            
            .financial-metrics th, .quarterly-data th, .data-table th,
            .financial-metrics thead, .quarterly-data thead, .data-table thead {
                background-color: var(--secondary-color);
                color: white;
                font-weight: bold;
                padding: 12px;
                text-align: left;
            }
            
            .financial-metrics td, .quarterly-data td, .data-table td {
                padding: 10px 12px;
                border-bottom: 1px solid var(--border-color);
            }
            
            .financial-metrics tr:nth-child(even), .quarterly-data tr:nth-child(even), .data-table tr:nth-child(even) {
                background-color: rgba(0,0,0,0.02);
            }
            
            .financial-metrics tr:hover, .quarterly-data tr:hover, .data-table tr:hover {
                background-color: rgba(52, 152, 219, 0.05);
            }
            
            @media print {
                body {
                    font-size: 12pt;
                }
                .section {
                    break-inside: avoid;
                    page-break-inside: avoid;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 AI Hedge Fund Analysis Report</h1>
            <p class="timestamp">Generated on {{timestamp}}</p>
            <h2>Analysis for {{ticker}}</h2>
        </div>
             
        <div class="section">
            <h2><span class="icon">💰</span>Summary</h2>
            <div class="content">{{summary|safe}}</div>
        </div> 

        <div class="section">
            <h2><span class="icon">📈</span>Market Data Analysis</h2>
            <div class="content">{{market_data|safe}}</div>
        </div>
        
        <div class="section">
            <h2><span class="icon">📰</span>Market Sentiment Analysis</h2>
            <div class="content">{{sentiment_analysis|safe}}</div>
        </div>
        
        <div class="section">
            <h2><span class="icon">🌐</span>Macro-Economic Analysis</h2>
            <div class="content">{{macro_analysis|safe}}</div>
        </div>
        
        <div class="section">
            <h2><span class="icon">📊</span>Trading Strategy</h2>
            <div class="content">{{strategy|safe}}</div>
        </div>
        
        <div class="section">
            <h2><span class="icon">⚠️</span>Risk Assessment</h2>
            <div class="content">{{risk_assessment|safe}}</div>
        </div>
        
        <footer>
            AI Hedge Fund Analysis Tool - This information is AI generated for educational purposes only and does not constitute investment advice. Always do your own research and consider seeking professional advice before making investment decisions.
        </footer>
    </body>
    </html>
    """
    
    # Helper function to convert markdown-style formatting to HTML
    def markdown_to_html(text):
        # First, remove any <think> tags and their content
        if '<think>' in text:
            start_idx = text.find('<think>')
            end_idx = text.find('</think>')
            if start_idx >= 0 and end_idx >= 0:
                text = text[:start_idx] + text[end_idx + 8:]
        
        # Remove ```html and ``` markers that might be in the text
        text = text.replace('```html', '')
        text = text.replace('```', '')
        
        # Process markdown tables
        if '|' in text and '-|-' in text:
            # Find table sections
            lines = text.split('\n')
            in_table = False
            table_lines = []
            non_table_lines = []
            current_table = []
            
            for line in lines:
                if line.strip().startswith('|') and not in_table:
                    # Start of table
                    in_table = True
                    current_table = [line]
                elif line.strip().startswith('|') and in_table:
                    # Continue table
                    current_table.append(line)
                elif in_table:
                    # End of table
                    in_table = False
                    # Convert table to HTML
                    table_html = convert_markdown_table_to_html(''.join(current_table))
                    table_lines.append(table_html)
                    current_table = []
                    non_table_lines.append(line)
                else:
                    non_table_lines.append(line)
            
            # Handle case where table is at the end of text
            if in_table:
                table_html = convert_markdown_table_to_html(''.join(current_table))
                table_lines.append(table_html)
            
            # Reassemble text, placing HTML tables where they were
            if table_lines:
                # Combine lines with tables inserted
                processed_lines = []
                table_idx = 0
                for line in lines:
                    if line.strip().startswith('|'):
                        if table_idx < len(table_lines) and (table_idx == 0 or line != lines[lines.index(line) - 1]):
                            processed_lines.append(table_lines[table_idx])
                            table_idx += 1
                    else:
                        processed_lines.append(line)
                
                text = '\n'.join(processed_lines)
        
        # Skip existing HTML tables but process everything else
        if '<table' in text:
            # For text with HTML tables, split by table tags and process each part
            parts = text.split('<table')
            processed_parts = []
            
            for i, part in enumerate(parts):
                if i == 0:
                    # First part (before any table)
                    processed_parts.append(process_non_table_text(part))
                else:
                    # Parts containing tables
                    table_parts = part.split('</table>')
                    if len(table_parts) > 1:
                        # Add the table part with its tags
                        processed_parts.append('<table' + table_parts[0] + '</table>')
                        # Process any text after the table
                        if table_parts[1]:
                            processed_parts.append(process_non_table_text(table_parts[1]))
                    else:
                        # No closing tag found, treat as non-table
                        processed_parts.append(process_non_table_text(part))
            
            return ''.join(processed_parts)
        else:
            # No HTML tables, process everything
            return process_non_table_text(text)

    def convert_markdown_table_to_html(markdown_table):
        """Convert a markdown table to HTML table"""
        lines = markdown_table.strip().split('\n')
        
        # Check if this is a valid markdown table
        is_valid = False
        for line in lines:
            if line.strip().startswith('|') and line.strip().endswith('|'):
                is_valid = True
                break
        
        if not is_valid:
            # Not a valid markdown table, return as is
            return markdown_table
        
        # Find the separator row (row with dashes)
        separator_row_index = -1
        for i, line in enumerate(lines):
            if i > 0 and '---' in line and line.strip().startswith('|'):
                separator_row_index = i
                break
        
        # If no separator row found, assume a simplified table format
        # where the first row is header and the rest are body
        if separator_row_index == -1:
            separator_row_index = 0
        
        # Clean up the table rows - remove empty or separator-only rows
        valid_lines = []
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
            # Skip separator line
            if i == separator_row_index:
                continue
            # Only include lines with actual content
            if '|' in line and not all(cell.strip() in ['', '-', ''] for cell in line.split('|')):
                valid_lines.append(line)
        
        # Begin table HTML
        html_table = '<table class="data-table">\n'
        
        # Process header (first row)
        if len(valid_lines) > 0:
            header_row = valid_lines[0].strip()
            headers = [cell.strip() for cell in header_row.split('|')]
            # Remove empty first/last cells if they exist
            if headers[0] == '':
                headers = headers[1:]
            if headers[-1] == '':
                headers = headers[:-1]
                
            # Filter out completely empty headers and headers that are just separators
            headers = [h for h in headers if h.strip() != '' and not all(c == '-' for c in h.strip())]
            
            # Create header
            html_table += '  <thead>\n    <tr>\n'
            for header in headers:
                # Remove markdown formatting from headers
                clean_header = header.replace('**', '').replace('<strong>', '').replace('</strong>', '')
                html_table += f'      <th>{clean_header}</th>\n'
            html_table += '    </tr>\n  </thead>\n'
        
        # Process body rows
        html_table += '  <tbody>\n'
        for i, line in enumerate(valid_lines):
            if i == 0:  # Skip header row
                continue
                
            # Split the line into cells
            cells = [cell.strip() for cell in line.split('|')]
            # Remove empty first/last cells if they exist
            if cells[0] == '':
                cells = cells[1:]
            if cells[-1] == '':
                cells = cells[:-1]
                
            # Filter out completely empty cells and rows that are just separators
            cells = [c for c in cells if c.strip() != '' and not all(ch == '-' for ch in c.strip())]
            if not cells:
                continue
                
            # Create table row
            html_table += '    <tr>\n'
            
            # If we have fewer cells than headers, add empty cells to fill
            if len(headers) > len(cells):
                cells.extend([''] * (len(headers) - len(cells)))
            
            # If we have more cells than headers, use a reasonable number of columns (don't truncate data)
            column_count = max(len(headers), min(6, len(cells))) if len(headers) > 0 else min(6, len(cells))
            cells = cells[:column_count]
            
            for cell in cells:
                # Process cell content (preserve formatting)
                processed_cell = cell.replace('**', '<strong>').replace('<strong><strong>', '<strong>')
                
                # Balance strong tags
                open_count = processed_cell.count('<strong>')
                close_count = processed_cell.count('</strong>')
                if open_count > close_count:
                    processed_cell += '</strong>' * (open_count - close_count)
                    
                html_table += f'      <td>{processed_cell}</td>\n'
            html_table += '    </tr>\n'
            
        html_table += '  </tbody>\n</table>\n'
        
        return html_table
        
    def process_non_table_text(text):
        """Process text that doesn't contain HTML tables"""
        # Headers
        text = text.replace('\n## ', '\n<h2>').replace('\n### ', '\n<h3>')
        
        # Add closing tags to headers
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('<h2>') and not line.endswith('</h2>'):
                lines[i] = f"{line}</h2>"
            if line.startswith('<h3>') and not line.endswith('</h3>'):
                lines[i] = f"{line}</h3>"
        
        text = '\n'.join(lines)
        
        # Bold and italics
        text = text.replace('**', '<strong>').replace('__', '<strong>')
        
        # Balance strong tags
        open_count = text.count('<strong>')
        close_count = text.count('</strong>')
        if open_count > close_count:
            text += '</strong>' * (open_count - close_count)
        
        # Improved list handling
        if '\n- ' in text:
            # Split the text by lines
            lines = text.split('\n')
            in_list = False
            result_lines = []
            current_list = []
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('- '):
                    # This is a list item
                    if not in_list:
                        in_list = True
                        current_list = []
                    # Add item to current list
                    current_list.append('<li>' + stripped[2:].strip() + '</li>')
                else:
                    # Not a list item
                    if in_list:
                        # End the list
                        in_list = False
                        result_lines.append('<ul>' + ''.join(current_list) + '</ul>')
                    # Add non-list line
                    result_lines.append(line)
            
            # If the last item was a list
            if in_list:
                result_lines.append('<ul>' + ''.join(current_list) + '</ul>')
            
            text = '\n'.join(result_lines)
        
        # Handle line breaks and paragraph wrapping
        paragraphs = text.split('\n\n')
        processed_paragraphs = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Skip wrapping if already HTML
                if paragraph.strip().startswith('<h') or paragraph.strip().startswith('<ul'):
                    processed_paragraphs.append(paragraph)
                else:
                    processed_paragraphs.append('<p>' + paragraph.replace('\n', '<br>') + '</p>')
        
        text = '\n'.join(processed_paragraphs)
        
        # Fix any nested tags
        text = text.replace('<p><h2>', '<h2>').replace('</h2></p>', '</h2>')
        text = text.replace('<p><h3>', '<h3>').replace('</h3></p>', '</h3>')
        text = text.replace('<p><ul>', '<ul>').replace('</ul></p>', '</ul>')
        
        return text
    
    # Process result values to convert markdown formatting to HTML
    processed_results = {}
    for key, value in result.items():
        processed_results[key] = markdown_to_html(value)
    
    # Render template
    template = jinja2.Template(template)
    html_content = template.render(
        ticker=ticker,
        timestamp=timestamp,
        market_data=processed_results["market_data"],
        sentiment_analysis=processed_results["sentiment_analysis"],
        macro_analysis=processed_results["macro_analysis"],
        strategy=processed_results["strategy"],
        risk_assessment=processed_results["risk_assessment"],
        summary=processed_results["summary"]
    )
    
    return html_content

def run_ai_hedge_fund(ticker: str, open_browser: bool = True) -> None:
    """
    Run the analysis and generate an HTML report for the given ticker symbol
    
    Args:
        ticker: Stock ticker symbol to analyze
        open_browser: Whether to open the browser automatically (set to False in Docker)
    """
    try:
        logger.info(f"Starting analysis for {ticker}")
        
        # Validate ticker input
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        # Run the sequential chain
        result = sequential_agent({"ticker": ticker})
        logger.info(f"Analysis completed for {ticker}")
        
        # Generate HTML report
        html_content = generate_html_report(result, ticker)
        
        # Create reports directory if it doesn't exist
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(reports_dir, f'ai_hedge_fund_{ticker}_{timestamp}.html')
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        # Check if running in Docker container
        in_docker = os.path.exists('/.dockerenv')
        
        # Open in default browser if not in Docker and if open_browser is True
        if open_browser and not in_docker:
            try:
                abs_path = os.path.abspath(report_path)
                webbrowser.open('file://' + abs_path)
            except Exception as e:
                logger.warning(f"Could not open browser: {str(e)}")
        
        logger.info(f"Report generated and saved to: {report_path}")
        print(f"\nReport generated and saved to: {report_path}")
        
        return report_path
    
    except Exception as e:
        logger.error(f"Error during analysis for {ticker}: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return None

def main():
    """Main entry point with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Hedge Fund Analysis Tool')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol to analyze')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()
    
    # Run the analysis with browser control
    run_ai_hedge_fund(args.ticker.upper(), open_browser=not args.no_browser)

# Store all chains in the chains list
chains = [market_data_chain, sentiment_chain, macro_analysis_chain, strategy_chain, risk_chain, summary_chain]

# Run the analysis for a given ticker
if __name__ == "__main__":
    main()