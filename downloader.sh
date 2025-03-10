#!/bin/bash

# Check if urls.txt exists
if [ ! -f urls.txt ]; then
    echo "Error: urls.txt not found"
    exit 1
fi

# Create a directory for downloads if it doesn't exist
mkdir -p downloads
cd downloads

# Function to download a single file
download_file() {
    local url=$1
    local output=$(basename "$url")
    
    # Extract host and path from URL
    local host=$(echo "$url" | awk -F/ '{print $3}')
    local path=$(echo "$url" | sed -e "s|https\?://[^/]*||")
    
    # Create temporary request file
    local tmp_request=$(mktemp)
    echo -e "GET $path HTTP/1.1\r\nHost: $host\r\nConnection: close\r\n\r\n" > "$tmp_request"
    
    echo "Downloading $output..."
    
    # Open TCP connection and send request
    exec 3<>/dev/tcp/$host/80
    cat "$tmp_request" >&3
    
    # Read the response and save to output file
    {
        # Skip headers
        while IFS= read -r line; do
            [[ $line == $'\r' ]] && break
        done
        # Save body to file
        cat > "$output"
    } <&3
    
    # Close the connection and cleanup
    exec 3>&-
    rm "$tmp_request"
    
    echo "Finished downloading $output"
}

# Maximum number of concurrent downloads
MAX_PARALLEL=5
count=0

# Read URLs and download in parallel
while IFS= read -r url; do
    # Skip empty lines and comments
    [[ -z "$url" || "$url" =~ ^#.*$ ]] && continue
    
    # Start download in background
    download_file "$url" &
    
    # Increment counter
    ((count++))
    
    # If we've reached max parallel downloads, wait for one to finish
    if ((count >= MAX_PARALLEL)); then
        wait -n
        ((count--))
    fi
done < ../urls.txt

# Wait for remaining downloads to complete
wait

echo "All downloads completed!"


# Type the following command to make the script executable:
# chmod +x downloader.sh

# Run the script using the following command:
# ./downloader.sh

# To generate the urls in the urls.txt file, use the following command:
# wget -q -O - https://web.ais.dk/aisdata/ | grep -o 'aisdk-2025-02[^"]*\.zip' | sort -u | awk '{print "https://web.ais.dk/aisdata/" $0}' > urls.txt