#!/bin/bash
echo "Installing system dependencies for SUMO"
# openai-baselines dependencies
brew install cmake openmpi
# rllab dependencies
brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi
# sumo dependencies
brew install Caskroom/cask/xquartz autoconf automake pkg-config libtool gdal proj xerces-c fox

echo "Installing sumo binaries"
mkdir -p $HOME/sumo_binaries/bin
pushd $HOME/sumo_binaries/bin
wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.4.0/binaries-mac.tar.xz
tar -xf binaries-mac.tar.xz
rm binaries-mac.tar.xz
chmod +x *
popd
export SUMO_HOME="$HOME/sumo_binaries/bin"
export PATH="$SUMO_HOME:$PATH"

echo 'Add the following to your ~/.bashrc:'
echo ''
echo 'export SUMO_HOME="$HOME/sumo_binaries/bin"'
echo 'export PATH="$SUMO_HOME:$PATH"'
