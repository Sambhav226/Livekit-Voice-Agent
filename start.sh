#!/bin/bash

# Start the MCP server in the background
echo "Starting MCP Server..."
python3 mcp_server.py &
MCP_PID=$!

# Wait a bit to ensure MCP server is up
sleep 3

# Now start the agent
echo "Starting Agent..."
python3 agent1.py

# Optional: Wait and then kill the MCP server after agent exits
kill $MCP_PID
