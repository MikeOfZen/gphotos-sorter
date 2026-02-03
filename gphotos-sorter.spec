# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for gphotos-sorter."""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect rich unicode data files and submodules
rich_datas = collect_data_files('rich')
rich_hiddenimports = collect_submodules('rich')

a = Analysis(
    ['run_gphotos_sorter.py'],
    pathex=[],
    binaries=[],
    datas=rich_datas,
    hiddenimports=rich_hiddenimports + [
        'typer',
        'click',
        'PIL',
        'imagehash',
        'pydantic',
        'yaml',
        'rich',
        'rich._unicode_data',
        'gphotos_sorter',
        'gphotos_sorter.cli',
        'gphotos_sorter.config',
        'gphotos_sorter.scanner',
        'gphotos_sorter.scanner_mp',
        'gphotos_sorter.db',
        'gphotos_sorter.hash_utils',
        'gphotos_sorter.metadata_utils',
        'gphotos_sorter.date_utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='gphotos-sorter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
