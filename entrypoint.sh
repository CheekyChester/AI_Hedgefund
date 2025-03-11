#!/bin/bash
set -e

# Print a welcome message
echo "🚀 Starting AI Hedge Fund Analysis Tool"
echo "--------------------------------------"

# Check if API key is set
if [[ -z "${PPLX_API_KEY}" || "${PPLX_API_KEY}" == "replace_with_your_key_at_runtime" ]]; then
    echo "⚠️  WARNING: PPLX_API_KEY is not set or still has the default value."
    echo "Please provide a valid API key by running the container with:"
    echo "docker run -e PPLX_API_KEY=your_api_key ..."
fi

# Check if a ticker was provided
TICKER=${1:-AAPL}
echo "📊 Analyzing ticker: ${TICKER}"

# Run the script with the --no-browser flag since we're in Docker
python ai_hedge_fund.py "${TICKER}" --no-browser

# Print final message
echo "✅ Analysis complete! Check the reports directory for the generated HTML file."

# Keep the container running if in interactive mode
if [[ -t 0 ]]; then
    echo "Press Ctrl+C to exit"
    # Sleep indefinitely
    tail -f /dev/null
fi