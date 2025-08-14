# CSF Descriptor Installation Guide

This guide provides comprehensive instructions for installing the CSF Descriptor package on various platforms.

## 🚀 Quick Start

### Prerequisites
- **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2019+)
- **CMake 3.10+**
- **HDF5 library** (libhdf5-dev or hdf5-devel)
- **OpenMP** (optional, for parallel processing)

### System-Specific Prerequisites

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libhdf5-dev libopenmpi-dev
```

#### CentOS/RHEL/Fedora
```bash
# CentOS/RHEL
sudo yum install gcc-c++ cmake hdf5-devel openmpi-devel

# Fedora
sudo dnf install gcc-c++ cmake hdf5-devel openmpi-devel
```

#### macOS
```bash
# Using Homebrew
brew install cmake hdf5 open-mpi

# Using MacPorts
sudo port install cmake hdf5 openmpi
```

#### Windows
- Install [Visual Studio 2019+](https://visualstudio.microsoft.com/) with C++ development tools
- Install [CMake](https://cmake.org/download/)
- Install [HDF5](https://www.hdfgroup.org/downloads/hdf5/) (Windows installer)

## 📦 Installation Methods

### Method 1: Pre-built Packages (Recommended)

#### Linux
```bash
# Download from releases page
wget https://github.com/csf-team/csf-descriptor/releases/download/v1.0.0/csf-descriptor-1.0.0-Linux.tar.gz

# Extract and install
tar -xzf csf-descriptor-1.0.0-Linux.tar.gz
cd csf-descriptor-1.0.0-Linux
sudo ./install.sh
```

#### Ubuntu/Debian
```bash
# Download .deb package
wget https://github.com/csf-team/csf-descriptor/releases/download/v1.0.0/csf-descriptor_1.0.0_amd64.deb

# Install
sudo dpkg -i csf-descriptor_1.0.0_amd64.deb
sudo apt-get install -f  # Fix any dependency issues
```

#### CentOS/RHEL/Fedora
```bash
# Download .rpm package
wget https://github.com/csf-team/csf-descriptor/releases/download/v1.0.0/csf-descriptor-1.0.0-1.x86_64.rpm

# Install
sudo rpm -i csf-descriptor-1.0.0-1.x86_64.rpm
```

#### macOS
```bash
# Download .dmg package
wget https://github.com/csf-team/csf-descriptor/releases/download/v1.0.0/csf-descriptor-1.0.0-Darwin.dmg

# Mount and install
hdiutil attach csf-descriptor-1.0.0-Darwin.dmg
cp -R "/Volumes/CSF Descriptor/csf-descriptor.app" /Applications/
```

#### Windows
1. Download `csf-descriptor-1.0.0-win64.exe` from releases
2. Run the installer and follow the prompts
3. Add to PATH during installation or manually

### Method 2: Build from Source

#### Linux/macOS
```bash
# Clone repository
git clone https://github.com/csf-team/csf-descriptor.git
cd csf-descriptor

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local

# Build
make -j$(nproc)

# Test
ctest --output-on-failure

# Install
sudo make install
```

#### Windows
```cmd
# Clone repository
git clone https://github.com/csf-team/csf-descriptor.git
cd csf-descriptor

# Create build directory
mkdir build && cd build

# Configure (Visual Studio)
cmake .. -G "Visual Studio 16 2019" -A x64

# Build
cmake --build . --config Release

# Test
ctest -C Release --output-on-failure

# Install (Run as Administrator)
cmake --build . --config Release --target install
```

### Method 3: Package Managers

#### Conda
```bash
# Create new environment
conda create -n csf-env csf-descriptor
conda activate csf-env
```

#### Homebrew (macOS/Linux)
```bash
brew tap csf-team/csf
brew install csf-descriptor
```

#### Chocolatey (Windows)
```cmd
choco install csf-descriptor
```

## 🔧 Verification

After installation, verify the installation:

```bash
# Check version
csf_descriptor --version

# Run basic test
csf_descriptor --help

# Test with sample data
csf_descriptor data/test.csf
```

## 🗂️ Directory Structure

After installation, the following structure will be created:

```
/usr/local/
├── bin/
│   └── csf_descriptor      # Main executable
├── lib/
│   ├── libcsf.a           # Static library
│   ├── pkgconfig/
│   │   └── csf.pc         # pkg-config file
│   └── cmake/csf/         # CMake configuration
├── include/csf/           # Header files
│   ├── csf_types.h
│   ├── csf_parser.h
│   ├── descriptor_generator.h
│   ├── parallel_processor.h
│   └── hdf5_writer.h
└── share/doc/csf-descriptor/
    └── README.md          # Documentation
```

## 🔄 Uninstallation

### Linux/macOS
```bash
# If installed from source
sudo make uninstall

# If installed from .deb
sudo apt-get remove csf-descriptor

# If installed from .rpm
sudo rpm -e csf-descriptor

# If installed from .tar.gz
sudo rm -rf /usr/local/bin/csf_descriptor
sudo rm -rf /usr/local/lib/libcsf.a
sudo rm -rf /usr/local/include/csf/
```

### Windows
- Control Panel → Programs → Uninstall CSF Descriptor
- Or delete installation directory (usually `C:\Program Files\csf-descriptor`)

### macOS
- Drag `csf-descriptor.app` to Trash
- Or use Homebrew: `brew uninstall csf-descriptor`

## 🛠️ Troubleshooting

### Common Issues

#### 1. HDF5 not found
```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev

# macOS
brew install hdf5

# Set HDF5 path manually
cmake .. -DHDF5_ROOT=/path/to/hdf5
```

#### 2. OpenMP not found
```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# macOS (with Homebrew)
brew install libomp

# Disable OpenMP
cmake .. -DENABLE_OPENMP=OFF
```

#### 3. Permission denied on installation
```bash
# Use sudo for system-wide installation
sudo make install

# Or install to user directory
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make install
```

#### 4. Windows: PATH issues
- Add installation directory to PATH environment variable
- Restart command prompt after installation

### Getting Help
- Check [GitHub Issues](https://github.com/csf-team/csf-descriptor/issues)
- Run with debug output: `csf_descriptor --debug`
- Check system compatibility: `csf_descriptor --version`

## 📋 Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Ubuntu 18.04+ | ✅ | Full support |
| CentOS 7+ | ✅ | Full support |
| macOS 10.15+ | ✅ | Full support |
| Windows 10+ | ✅ | Full support |
| Arch Linux | ✅ | Community support |
| FreeBSD | ⚠️ | Limited support |

## 🔗 Additional Resources

- [GitHub Repository](https://github.com/csf-team/csf-descriptor)
- [Release Notes](https://github.com/csf-team/csf-descriptor/releases)
- [Issue Tracker](https://github.com/csf-team/csf-descriptor/issues)
- [Documentation](https://csf-descriptor.readthedocs.io/)

## 📞 Support

For installation issues:
1. Check this guide first
2. Search existing [GitHub Issues](https://github.com/csf-team/csf-descriptor/issues)
3. Create a new issue with system information and error logs