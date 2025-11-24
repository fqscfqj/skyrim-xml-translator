import PyInstaller.__main__
import os
import shutil

def build():
    print("Starting build process...")
    
    # Clean up previous build
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')

    # Define data files to include
    # Format: 'source_path;dest_path' for Windows
    add_data = [
        'locales;locales',
        'config.example.json;.'
    ]
    
    args = [
        'main.py',
        '--name=SkyrimXMLTranslator',
        '--windowed', # Hide console
        '--onefile',  # Single executable
        '--clean',
        '--noconfirm',
    ]

    for data in add_data:
        args.append(f'--add-data={data}')

    print(f"Running PyInstaller with args: {args}")
    PyInstaller.__main__.run(args)
    print("Build finished. Executable is in 'dist' folder.")

if __name__ == '__main__':
    build()
