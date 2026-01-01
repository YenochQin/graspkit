#!/usr/bin/env python3
"""
GraspKit Package Build Script

This script builds the graspkit package and moves the generated packages
to ../Graspkit-tools/package directory.

Works with UV, Pixi, or traditional pip environments.

Usage:
    python build_package.py [--clean] [--dev]

Options:
    --clean    Clean previous build artifacts before building
    --dev      Build development version (default: production build)
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(f"Output: {result.stdout}")
    if result.stderr:
        print(f"Error output: {result.stderr}")

    return result


def clean_build_artifacts():
    """Clean previous build artifacts."""
    print("Cleaning previous build artifacts...")

    artifacts = ['build', 'dist', '*.egg-info']
    project_root = Path(__file__).parent

    for artifact in artifacts:
        if artifact.startswith('*.'):
            # Handle glob patterns
            for path in project_root.glob(artifact):
                if path.is_dir():
                    print(f"Removing directory: {path}")
                    shutil.rmtree(path)
                elif path.is_file():
                    print(f"Removing file: {path}")
                    path.unlink()
        else:
            path = project_root / artifact
            if path.exists():
                if path.is_dir():
                    print(f"Removing directory: {path}")
                    shutil.rmtree(path)
                else:
                    print(f"Removing file: {path}")
                    path.unlink()


def detect_environment_manager():
    """Detect which environment manager is being used."""
    project_root = Path(__file__).parent

    # Check for UV environment
    if (project_root / ".venv").exists() and (project_root / "uv.lock").exists():
        return "uv", sys.executable

    # Check for Pixi environment
    pixi_dirs = [
        project_root / ".pixi",
        project_root / ".pixi-env",
    ]
    if any(p.exists() for p in pixi_dirs):
        # Try to find pixi command
        try:
            result = subprocess.run(
                ["pixi", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            return "pixi", "pixi"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Default to current Python
    return "python", sys.executable


def ensure_package_directory():
    """Ensure the target package directory exists."""
    package_dir = Path(__file__).parent.parent / "Graspkit-tools" / "package"
    package_dir.mkdir(parents=True, exist_ok=True)
    print(f"Package directory: {package_dir}")
    return package_dir


def build_package(clean=False, dev=False):
    """Build the package using Python build system."""
    project_root = Path(__file__).parent

    if clean:
        clean_build_artifacts()

    # Detect environment manager
    env_manager, python_cmd = detect_environment_manager()
    print(f"Detected environment manager: {env_manager}")
    print(f"Using Python command: {python_cmd}")

    print("Building package...")

    # Build command based on environment manager
    if env_manager == "pixi":
        build_cmd = ["pixi", "run", "python", "-m", "build"]
    elif env_manager == "uv":
        build_cmd = [python_cmd, "-m", "build"]
    else:
        build_cmd = [python_cmd, "-m", "build"]

    # Check if build is available and install if needed
    try:
        run_command(build_cmd, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        print("Installing build tool...")

        if env_manager == "pixi":
            install_cmd = ["pixi", "add", "build"]
        elif env_manager == "uv":
            install_cmd = [python_cmd, "-m", "pip", "install", "build"]
        else:
            install_cmd = [python_cmd, "-m", "pip", "install", "build"]

        run_command(install_cmd, cwd=project_root)
        run_command(build_cmd, cwd=project_root)

    return project_root / "dist"


def move_packages_to_target(dist_dir, target_dir):
    """Move built packages to target directory."""
    print(f"Moving packages from {dist_dir} to {target_dir}")

    packages_moved = []

    for package_file in dist_dir.glob("*"):
        if package_file.is_file():
            target_file = target_dir / package_file.name

            # Remove existing file if it exists
            if target_file.exists():
                print(f"Removing existing package: {target_file}")
                target_file.unlink()

            # Move the package
            print(f"Moving {package_file.name} to {target_dir}")
            shutil.move(str(package_file), str(target_file))
            packages_moved.append(target_file)

    return packages_moved


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build GraspKit package")
    parser.add_argument("--clean", action="store_true",
                       help="Clean previous build artifacts before building")
    parser.add_argument("--dev", action="store_true",
                       help="Build development version")

    args = parser.parse_args()

    print("=" * 60)
    print("GraspKit Package Build Script")
    print("=" * 60)

    try:
        # Ensure target directory exists
        target_dir = ensure_package_directory()

        # Build the package
        dist_dir = build_package(clean=args.clean, dev=args.dev)

        # Move packages to target directory
        packages_moved = move_packages_to_target(dist_dir, target_dir)

        print("\n" + "=" * 60)
        print("BUILD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Packages moved to: {target_dir}")

        if packages_moved:
            print("\nGenerated packages:")
            for package in packages_moved:
                size = package.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"  - {package.name} ({size_mb:.2f} MB)")

        print(f"\nTarget directory: {target_dir}")
        print("You can now install the package with:")
        print(f"  pip install {target_dir}/*.whl")

    except Exception as e:
        print(f"\nBUILD FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()