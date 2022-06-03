# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


names = [
    "HSImageViewer",
    "HSTivitaAnalyzer",
    "HSCoFitAnalyzer",
    "HSLipidAnalyzer",
    "HSMultiAnalyzer"
]

analyses = [
	Analysis(
		['apps\\Q%s.py' % name],
		pathex=['.\\'],
		binaries=[],
		datas=[
			('.\\hsi\\version.txt', 'hsi'),
			('.\\hsi\\data\\*', 'hsi\\data'),
			('.\\hsi\\materials\\*', 'hsi\\materials')
			],
		hiddenimports=[],
		hookspath=[],
		hooksconfig={},
		runtime_hooks=[],
		excludes=[],
		win_no_prefer_redirects=False,
		win_private_assemblies=False,
		cipher=block_cipher,
		noarchive=False,
	) for name in names]
	
#pacs = [
#	PYZ(a.pure, a.zipped_data, cipher=block_cipher)
#	for a in analyses] 

executables = [
	EXE(
		PYZ(a.pure, a.zipped_data, cipher=block_cipher),
		a.scripts,
		[],
		exclude_binaries=True,
		name=name,
		debug=False,
		bootloader_ignore_signals=False,
		strip=False,
		upx=True,
		console=False,
		disable_windowed_traceback=False,
		argv_emulation=False,
		target_arch=None,
		codesign_identity=None,
		entitlements_file=None,
	) for (name, a) in zip(names, analyses)]

arg_list = []
for (a, exe) in zip(analyses, executables):
	arg_list.extend([exe, a.binaries, a.zipfiles, a.datas])
args = tuple(arg_list)

coll = COLLECT( 
    *args,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HSApps',
)
