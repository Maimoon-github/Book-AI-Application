# -*- mode: python ; coding: utf-8 -*-


import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_all

# Collect Streamlit data files and metadata
streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all('streamlit')

a = Analysis(
    ['book_ai_standalone.py'],
    pathex=[],
    binaries=streamlit_binaries,
    datas=streamlit_datas + collect_data_files('streamlit'),
    hiddenimports=[
        'streamlit',
        'streamlit.web.cli',
        'streamlit.web.server', 
        'streamlit.web.bootstrap',
        'streamlit.runtime.scriptrunner.script_runner',
        'streamlit.runtime.state',
        'streamlit.components.v1',
        'altair',
        'pydeck',
        'watchdog',
        'click',
        'tornado',
        'pyarrow',
        'streamlit.runtime.caching.hashing',
        'streamlit.runtime.metrics_util',
        'streamlit.web.server.server',
        'streamlit.web.server.browser_util',
        'streamlit.web.server.routes',
        'streamlit.web.server.media_file_handler',
        'streamlit.web.server.upload_file_request_handler',
        'streamlit.web.server.component_request_handler',
        'streamlit.web.server.health_handler',
        'streamlit.web.server.message_cache_handler',
        'streamlit.web.server.stats_request_handler'
    ] + streamlit_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BookAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
