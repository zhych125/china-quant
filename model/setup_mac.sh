#!/bin/bash

# Setup script for Mac users to install XGBoost dependencies

echo "Setting up XGBoost model dependencies for Mac..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew found. Updating..."
    brew update
fi

# Install OpenMP
echo "Installing OpenMP..."
brew install libomp

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r ../requirements.txt

# Set environment variable
echo "Setting KMP_DUPLICATE_LIB_OK environment variable..."
export KMP_DUPLICATE_LIB_OK=TRUE

# Add the environment variable to .zshrc or .bash_profile
if [[ "$SHELL" == *"zsh"* ]]; then
    if ! grep -q "KMP_DUPLICATE_LIB_OK" ~/.zshrc; then
        echo "export KMP_DUPLICATE_LIB_OK=TRUE" >> ~/.zshrc
        echo "Added environment variable to ~/.zshrc"
    fi
else
    if ! grep -q "KMP_DUPLICATE_LIB_OK" ~/.bash_profile; then
        echo "export KMP_DUPLICATE_LIB_OK=TRUE" >> ~/.bash_profile
        echo "Added environment variable to ~/.bash_profile"
    fi
fi

echo "Setup complete!"
echo "You may need to restart your terminal or run 'source ~/.zshrc' (or source ~/.bash_profile) for the changes to take effect."
echo "To run the model, use: python boost_model.py"