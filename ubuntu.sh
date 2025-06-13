#!/bin/bash
set -o pipefail

# Logging functions
log_message() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[LAUNCHER_LOGGER] [MODEL_INSTALL_LLAMA] --message \"$message\""
    fi
}

log_error() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[LAUNCHER_LOGGER] [MODEL_INSTALL_LLAMA] --error \"$message\"" >&2
    fi
}

# Error handling function
handle_error() {
    local exit_code=$1
    local error_msg=$2
    log_error "$error_msg (Exit code: $exit_code)"
    
    # Clean up if needed
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_message "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
    
    exit $exit_code
}

command_exists() {
    command -v "$1" &> /dev/null
}

# Step 1: Check system Python version and decide on installation
PYTHON_CMD=""
log_message "Searching for suitable Python (>= 3.11) in PATH..."

# Find all python3* executables in PATH and select the highest version >= 3.11
HIGHEST_VERSION=""
HIGHEST_CMD=""
PYTHON_CANDIDATES=$(compgen -c | grep -E '^python3(\.[0-9]+)?$' | sort -u)
for candidate in $PYTHON_CANDIDATES; do
    if command_exists "$candidate"; then
        VERSION=$($candidate --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; then
            # Compare versions to select the highest
            if [[ -z "$HIGHEST_VERSION" ]] || [[ $(printf '%s\n' "$VERSION" "$HIGHEST_VERSION" | sort -V | tail -n1) == "$VERSION" ]]; then
                HIGHEST_VERSION="$VERSION"
                HIGHEST_CMD="$candidate"
            fi
        fi
    fi
done

if [ -n "$HIGHEST_CMD" ]; then
    log_message "Found suitable Python: $HIGHEST_CMD ($HIGHEST_VERSION)"
    PYTHON_CMD="$HIGHEST_CMD"
fi

if [ -z "$PYTHON_CMD" ]; then
    handle_error 1 "No suitable Python (>= 3.11) found in PATH. Please install Python 3.11 or higher."
fi

log_message "Using Python at: $(which $PYTHON_CMD)"
log_message "Python setup complete."

# Step 2: Install all required packages at once
log_message "Installing required packages..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y pigz cmake libcurl4-openssl-dev python3-venv python3-pip
elif command -v yum &> /dev/null; then
    sudo yum install -y pigz cmake libcurl-openssl-dev python3-venv python3-pip
elif command -v dnf &> /dev/null; then
    sudo dnf install -y pigz cmake libcurl-openssl-dev python3-venv python3-pip
else
    log_error "No supported package manager found (apt-get, yum, or dnf)"
    exit 1
fi
log_message "All required packages installed successfully"

# Step 3: Check Docker installation
if ! command -v docker &> /dev/null; then
    log_message "Docker is not installed, installing..."
    sudo apt-get install -y docker.io
else
    log_message "Docker is already installed"
fi

# Step 4: Check NVIDIA Container Toolkit
if ! command -v nvidia-container-toolkit &> /dev/null; then
    log_message "NVIDIA Container Toolkit is not installed, installing..."
    
    # Detect package manager and install appropriate repository
    if command -v apt-get &> /dev/null; then
        # For DEB-based systems (Ubuntu/Debian)
        log_message "Setting up NVIDIA repository for DEB-based system..."
        
        # Download and add GPG key
        if ! curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg; then
            log_error "Failed to download NVIDIA GPG key"
            exit 1
        fi
        
        # Download and configure repository
        if ! curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null; then
            log_error "Failed to configure NVIDIA repository"
            exit 1
        fi
        
        # Update package list and install toolkit
        log_message "Installing NVIDIA Container Toolkit..."
        if ! sudo apt-get update; then
            log_error "Failed to update package lists"
            exit 1
        fi
        
        # Install with verbose output
        if ! sudo apt-get install -y nvidia-container-toolkit 2>&1 | tee /tmp/nvidia-install.log; then
            log_error "Failed to install NVIDIA Container Toolkit. Installation log:"
            cat /tmp/nvidia-install.log
            exit 1
        fi
        
        # Verify installation
        if ! command -v nvidia-ctk &> /dev/null; then
            log_error "nvidia-ctk command not found after installation. Installation log:"
            cat /tmp/nvidia-install.log
            exit 1
        fi
        
        log_message "NVIDIA Container Toolkit installation verified"
        
    elif command -v yum &> /dev/null || command -v dnf &> /dev/null; then
        # For RPM-based systems (RHEL/Fedora)
        log_message "Setting up NVIDIA repository for RPM-based system..."
        
        # Download and configure repository
        if ! curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
            sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null; then
            log_error "Failed to configure NVIDIA repository"
            exit 1
        fi
        
        # Install toolkit
        log_message "Installing NVIDIA Container Toolkit..."
        if command -v dnf &> /dev/null; then
            if ! sudo dnf install -y nvidia-container-toolkit; then
                log_error "Failed to install NVIDIA Container Toolkit"
                exit 1
            fi
        else
            if ! sudo yum install -y nvidia-container-toolkit; then
                log_error "Failed to install NVIDIA Container Toolkit"
                exit 1
            fi
        fi
    else
        log_error "Unsupported package manager. Only apt-get (DEB) and yum/dnf (RPM) are supported."
        exit 1
    fi
    
    # Configure Docker to use NVIDIA runtime
    log_message "Configuring Docker to use NVIDIA runtime..."
    if ! command -v nvidia-ctk &> /dev/null; then
        log_error "nvidia-ctk command not found. Installation may have failed."
        exit 1
    fi
    
    if ! sudo nvidia-ctk runtime configure --runtime=docker; then
        log_error "Failed to configure Docker runtime"
        exit 1
    fi
    
    # Restart Docker
    log_message "Restarting Docker service..."
    if ! sudo systemctl restart docker; then
        log_error "Failed to restart Docker service"
        exit 1
    fi
    
    log_message "NVIDIA Container Toolkit installed and configured successfully"
else
    log_message "NVIDIA Container Toolkit is already installed"
fi

# Step 5: Pull llama-server cuda image
log_message "Pulling llama-server cuda image..."
docker pull lmsysorg/sglang:latest

# Step 6: Create llama-server wrapper script
log_message "Creating llama-server wrapper script..."
LLAMA_WRAPPER_DIR="$HOME/.local/bin"
mkdir -p "$LLAMA_WRAPPER_DIR"

cat > "$LLAMA_WRAPPER_DIR/llama-server" << 'EOF'
#!/bin/bash
# Wrapper script to run llama-server using Docker
docker run --rm -it \
    --gpus all \
    -p 11434:11434 \
    -v "$(pwd):/app" \
    lmsysorg/sglang:latest \
    llama-server "$@"
EOF

chmod +x "$LLAMA_WRAPPER_DIR/llama-server"
log_message "Created llama-server wrapper at $LLAMA_WRAPPER_DIR/llama-server"

# Add to PATH if not already there
if [[ ":$PATH:" != *":$LLAMA_WRAPPER_DIR:"* ]]; then
    log_message "Adding $LLAMA_WRAPPER_DIR to PATH..."
    echo "export PATH=\"$LLAMA_WRAPPER_DIR:\$PATH\"" >> ~/.bashrc
    export PATH="$LLAMA_WRAPPER_DIR:$PATH"
    log_message "PATH updated for current session and future sessions"
fi

# Verify llama-server is available
if ! command -v llama-server &> /dev/null; then
    handle_error 1 "Failed to make llama-server available in PATH"
fi
log_message "llama-server command is now available"

# Step 7: Create and activate virtual environment
log_message "Creating virtual environment 'local_ai'..."
"$PYTHON_CMD" -m venv local_ai || handle_error $? "Failed to create virtual environment"

log_message "Activating virtual environment..."
source local_ai/bin/activate || handle_error $? "Failed to activate virtual environment"
log_message "Virtual environment activated."

# Step 8: Install local-ai toolkit
log_message "Setting up local-ai toolkit..."
if pip show local-ai &>/dev/null; then
    log_message "local-ai is installed. Checking for updates..."
    
    # Get installed version
    INSTALLED_VERSION=$(pip show local-ai | grep Version | awk '{print $2}')
    log_message "Current version: $INSTALLED_VERSION"
    
    # Get remote version (from GitHub repository without installing)
    log_message "Checking latest version from repository..."
    TEMP_VERSION_FILE=$(mktemp)
    if curl -s https://raw.githubusercontent.com/eternalai-org/local-ai/main/local_ai/__init__.py | grep -o "__version__ = \"[0-9.]*\"" | cut -d'"' -f2 > "$TEMP_VERSION_FILE"; then
        REMOTE_VERSION=$(cat "$TEMP_VERSION_FILE")
        rm "$TEMP_VERSION_FILE"
        
        log_message "Latest version: $REMOTE_VERSION"
        
        if [ "$(printf '%s\n' "$INSTALLED_VERSION" "$REMOTE_VERSION" | sort -V | head -n1)" = "$INSTALLED_VERSION" ] && [ "$INSTALLED_VERSION" != "$REMOTE_VERSION" ]; then
            log_message "New version available. Updating..."
            pip uninstall local-ai -y || handle_error $? "Failed to uninstall local-ai"
            pip install -q git+https://github.com/eternalai-org/local-ai.git || handle_error $? "Failed to update local-ai toolkit"
            log_message "local-ai toolkit updated to version $REMOTE_VERSION."
        else
            log_message "Already running the latest version. No update needed."
        fi
    else
        log_message "Could not check latest version. Proceeding with update to be safe..."
        pip uninstall local-ai -y || handle_error $? "Failed to uninstall local-ai"
        pip install -q git+https://github.com/eternalai-org/local-ai.git || handle_error $? "Failed to update local-ai toolkit"
        log_message "local-ai toolkit updated."
    fi
else
    log_message "Installing local-ai toolkit..."
    pip install -q git+https://github.com/eternalai-org/local-ai.git || handle_error $? "Failed to install local-ai toolkit"
    log_message "local-ai toolkit installed."
fi

log_message "Setup completed successfully."
