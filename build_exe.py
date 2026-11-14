import os
import shutil
import sys
import argparse


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def check_project_environment():
    expected = os.path.join(PROJECT_ROOT, 'env')
    if os.path.normcase(os.path.realpath(sys.prefix)) != os.path.normcase(os.path.realpath(expected)):
        raise SystemExit(
            f'Builds must use the project env, not {sys.executable}.\n'
            'Create it and install the locked dependencies first:\n'
            '  python -m venv env\n'
            '  env\\Scripts\\python.exe -m pip install -r build-requirements.lock\n'
            'Then run: .\\build.ps1'
        )


def configure_build_path():
    if os.name != 'nt':
        return
    windows = os.environ.get('SystemRoot') or os.environ['WINDIR']
    # PyInstaller searches PATH for DLLs. External tools can ship incompatible
    # copies of Windows ICU (e.g. Poppler's icuuc.dll), breaking Qt at startup.
    paths = [
        os.path.join(windows, 'System32'),
        windows,
        os.path.join(sys.prefix, 'Scripts'),
        sys.base_prefix,
        os.path.join(sys.base_prefix, 'DLLs'),
    ]
    os.environ['PATH'] = os.pathsep.join(paths)


def check_environment():
    """Validate the build interpreter and actual Qt DLL loading before cleanup."""
    check_project_environment()
    configure_build_path()
    try:
        from PyQt6.QtWidgets import QApplication  # noqa: F401
    except ImportError:
        raise SystemExit(
            f'QtWidgets cannot load in {sys.executable}. Reinstall the build dependencies:\n'
            '  env\\Scripts\\python.exe -m pip install -r build-requirements.lock'
        )


def collect_data_files():
    """Return a list of --add-data strings for PyInstaller.

    Uses os.pathsep + absolute paths cross-platform.
    Note: glossary/ vector files are NOT bundled; self-created at runtime.
    """
    candidates = [
        ('locales', 'locales'),
        ('prompts', 'prompts'),
        ('config.example.json', '.'),
        ('assets', 'assets'),
    ]
    add_data = []
    if os.path.exists('config.json'):
        print('Skipping config.json during build to avoid bundling local API keys. Use config.example.json instead.')
    for src, dst in candidates:
        if os.path.exists(src):
            abs_src = os.path.abspath(src)
            add_data.append(f'{abs_src}{os.pathsep}{dst}')
    return add_data


def copy_runtime_folders_to_dist(dist_root: str):
    """Copy runtime folders next to the exe (exe旁).

    PyInstaller may place bundled datas under the internal extraction directory.
    This ensures the end-user can edit/override JSON files directly in dist.
    """
    folders = ['locales', 'prompts', 'assets']
    for folder in folders:
        src = os.path.abspath(folder)
        if not os.path.isdir(src):
            continue

        dst = os.path.join(dist_root, folder)
        if os.path.exists(dst):
            shutil.rmtree(dst)

        shutil.copytree(src, dst)


def resolve_icon_path(icon):
    if not icon:
        return None
    candidates = [icon]
    root, ext = os.path.splitext(icon)
    if os.name != "nt" and ext.lower() == ".ico":
        candidates.append(root + ".png")
        candidates.append(os.path.join("assets", "logo.png"))
    for cand in candidates:
        if cand and os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def build(onefile=True, windowed=True, name='SkyrimXMLTranslator', icon=None,
          clean_build=True, copy_runtime_folders=True):
    print('Starting build process...')
    check_environment()
    import PyInstaller.__main__

    os.chdir(PROJECT_ROOT)
    print(f'Build interpreter: {sys.executable}')

    if not clean_build:
        print('Incremental build requested: reusing previous dist/build without --clean.')

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

    resolved_icon = resolve_icon_path(icon)
    if resolved_icon:
        args.append(f'--icon={resolved_icon}')
    elif icon:
        print(f'Icon not found, skipping: {icon}')

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
                   help='Skip copying locales/prompts folders next to dist artifacts')
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
