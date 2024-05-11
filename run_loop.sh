#!/bin/bash

log() {
	local timestamp=$(date +"%Y-%m-%d %T")
	echo "$timestamp: $@"
}

# Function to kill the background process
cleanup() {
	echo ''
	log "Caught SIGINT signal. Killing the background process..."
	kill $PID 2>/dev/null
	exit 1
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

while true; do
	# Start the command in the background
	log 'Starting whisper-server process in the background'
	poetry run python main.py serve &

	# Get the Process ID (PID) of the last background process
	PID=$!

	# Sleep 1500 seconds (about 30 min)
	sleep 1500

	# Terminate the process
	log 'Killing existing whisper-server process'
	kill $PID

	# Optional: Wait a bit before restarting the process
	sleep 10
	# log 'After sleep 2'
done
