import PyInstaller.__main__
import os
import shutil
import argparse


def collect_data_files():
    """Return a list of --add-data strings for PyInstaller (Windows format).

    Ensures commonly required data files are included if they exist in repo.
    """
    candidates = [
        ('locales', 'locales'),
        ('prompts', 'prompts'),
        ('config.json', '.'),
        ('config.example.json', '.'),
        ('glossary.json', '.'),
        ('data', 'data'),
        ('assets', 'assets'),
    ]
    add_data = []
    for src, dst in candidates:
        if os.path.exists(src):
            add_data.append(f'{src};{dst}')
    return add_data


def copy_runtime_folders_to_dist(dist_root: str):
    """Copy runtime folders (locales/prompts/data) next to the executable.

    PyInstaller may place bundled datas under the internal extraction directory.
    This ensures the end-user can edit/override JSON files directly in dist.
    """
    folders = ['locales', 'prompts', 'data']
    for folder in folders:
        src = os.path.abspath(folder)
        if not os.path.isdir(src):
            continue

        dst = os.path.join(dist_root, folder)
        if os.path.exists(dst):
            shutil.rmtree(dst)

        shutil.copytree(src, dst)


def build(onefile=True, windowed=True, name='SkyrimXMLTranslator', icon=None,
          clean_build=True, copy_runtime_folders=True):
    print('Starting build process...')

    if clean_build:
        # Clean up previous build for deterministic release artifacts.
        if os.path.exists('dist'):
            shutil.rmtree('dist')
        if os.path.exists('build'):
            shutil.rmtree('build')

    args = [
        'main.py',
        f'--name={name}',
        '--noconfirm',
    ]

    if clean_build:
        args.append('--clean')

    if windowed:
        args.append('--windowed')
    else:
        args.append('--console')

    if onefile:
        args.append('--onefile')
    else:
        args.append('--onedir')

    if icon and os.path.exists(icon):
        args.append(f'--icon={icon}')

    for data in collect_data_files():
        args.append(f'--add-data={data}')

    print(f'Running PyInstaller with args: {args}')
    PyInstaller.__main__.run(args)

    if copy_runtime_folders:
        # Ensure key folders are available next to the executable in dist output.
        dist_root = os.path.join('dist', name) if not onefile else 'dist'
        if os.path.isdir(dist_root):
            copy_runtime_folders_to_dist(dist_root)
    print("Build finished. Executable is in 'dist' folder.")



def parse_args():
    p = argparse.ArgumentParser(description='Build SkyrimXMLTranslator with PyInstaller')
    p.add_argument('--onefile', dest='onefile', action='store_true', help='Build single-file executable')
    p.add_argument('--onedir', dest='onefile', action='store_false', help='Build directory executable')
    p.set_defaults(onefile=True)
    p.add_argument('--console', dest='windowed', action='store_false', help='Keep console (no windowed mode)')
    p.add_argument('--windowed', dest='windowed', action='store_true', help='Hide console (windowed)')
    p.set_defaults(windowed=True)
    p.add_argument('--incremental', action='store_true',
                   help='Faster iterative build: skip output cleanup and PyInstaller --clean')
    p.add_argument('--skip-runtime-copy', action='store_true',
                   help='Skip copying locales/prompts/data folders next to dist artifacts')
    p.add_argument('--name', default='SkyrimXMLTranslator')
    p.add_argument('--icon', default='assets/logo.ico', help='Path to .ico file to use as application icon')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build(
        onefile=args.onefile,
        windowed=args.windowed,
        name=args.name,
        icon=args.icon,
        clean_build=not args.incremental,
        copy_runtime_folders=not args.skip_runtime_copy,
    )
