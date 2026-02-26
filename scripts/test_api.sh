#!/bin/bash

# Test the API with all test images and show NLP results

TEST_IMAGE_DIR="./tests/integration/images"
API_URL="http://localhost:8000/analyze"

echo "Testing API with test images..."
echo "==============================="
echo ""

total_start=$(date +%s%3N)

for image_path in "$TEST_IMAGE_DIR"/*.jpg; do
    image_name=$(basename "$image_path" .jpg)

    echo "📚 $image_name"

    # Send request and extract only NLP results
    req_start=$(date +%s%3N)
    response=$(curl -s -X POST "$API_URL" -F "file=@$image_path")
    req_end=$(date +%s%3N)
    elapsed_ms=$(( req_end - req_start ))
    echo "  ⏱  ${elapsed_ms}ms"

    # Extract potential authors and titles
    authors=$(echo "$response" | jq -r '.nlpAnalysis.potentialAuthors[]? // empty')
    titles=$(echo "$response" | jq -r '.nlpAnalysis.potentialTitles[]? // empty')

    # Check if analysis was successful
    is_success=$(echo "$response" | jq -r '.analysisStatus.isSuccess')

    if [ "$is_success" = "true" ]; then
        if [ -z "$authors" ] && [ -z "$titles" ]; then
            echo "  ⚠️  No authors or titles detected"
        else
            if [ -n "$authors" ]; then
                echo "  Authors:"
                echo "$authors" | sed 's/^/    - /'
            fi
            if [ -n "$titles" ]; then
                echo "  Titles:"
                echo "$titles" | sed 's/^/    - /'
            fi
        fi
    else
        error=$(echo "$response" | jq -r '.analysisStatus.errorMessage')
        echo "  ❌ Error: $error"
    fi

    echo ""
done

total_end=$(date +%s%3N)
total_ms=$(( total_end - total_start ))
total_s=$(( total_ms / 1000 ))
remainder_ms=$(( total_ms % 1000 ))
echo "==============================="
echo "Total time: ${total_s}.$(printf '%03d' $remainder_ms)s"
