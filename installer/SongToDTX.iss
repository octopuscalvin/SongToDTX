; Song to DTX installer (Inno Setup 6). Build each edition from a folder made by stage.py:
;   ISCC /DEdition=NVIDIA /DStage=<stage dir> /DOut=<output dir> SongToDTX.iss
;   ISCC /DEdition=CPU    /DStage=<stage dir> /DOut=<output dir> SongToDTX.iss
; The NVIDIA edition is over 2 GB packed, so it is split into Setup.exe + .bin files (keep them together).
; /DSlice=1300000000 makes the .bin parts that size (a GitHub release takes files under 2 GB).

#ifndef Edition
  #define Edition "NVIDIA"
#endif
#ifndef Slice
  #define Slice "max"
#endif
#define AppVer Trim(FileRead(FileOpen(Stage + "\VERSION")))

[Setup]
AppId={{6E3C1B52-8D0A-4C1E-9B7A-5D2F3A9C7E41}
AppName=Song to DTX
AppVersion={#AppVer}
AppVerName=Song to DTX {#AppVer} ({#Edition})
AppPublisher=Song to DTX (hobby project)
DefaultDirName={autopf}\Song to DTX
DefaultGroupName=Song to DTX
DisableProgramGroupPage=yes
; installs for the current user: the app writes its cache and settings next to itself
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile={#Stage}\THIRD_PARTY_NOTICES.txt
OutputDir={#Out}
OutputBaseFilename=SongToDTX-{#AppVer}-{#Edition}-Setup
Compression=lzma2/normal
SolidCompression=yes
LZMANumBlockThreads=8
#if Edition == "NVIDIA"
DiskSpanning=yes
DiskSliceSize={#Slice}
#endif
WizardStyle=modern
SetupIconFile={#Stage}\app\icon.ico
UninstallDisplayIcon={app}\app\icon.ico
UninstallDisplayName=Song to DTX ({#Edition})

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\Song to DTX"; Filename: "{app}\Song to DTX.bat"; WorkingDir: "{app}"; IconFilename: "{app}\app\icon.ico"
Name: "{autodesktop}\Song to DTX"; Filename: "{app}\Song to DTX.bat"; WorkingDir: "{app}"; IconFilename: "{app}\app\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\python.exe"; Parameters: "-s ""{app}\download_parts.py"""; WorkingDir: "{app}"; StatusMsg: "Downloading the drum transcription parts (about 16 MB) ..."; Flags: waituntilterminated
Filename: "{app}\Song to DTX.bat"; WorkingDir: "{app}"; Description: "Start Song to DTX now"; Flags: postinstall nowait skipifsilent shellexec

[InstallDelete]
; an older version's plain rule files: the Library now ships compiled (.pyd)
Type: files; Name: "{app}\tempo.py"
Type: files; Name: "{app}\sounds.py"
Type: files; Name: "{app}\genres.py"
Type: files; Name: "{app}\drumkit.py"
Type: files; Name: "{app}\live_mv.py"
Type: files; Name: "{app}\preview.py"
Type: files; Name: "{app}\dtx.py"
Type: files; Name: "{app}\archive.py"
Type: files; Name: "{app}\library_check.py"

[UninstallDelete]
; made after install: downloaded parts, caches and settings (your charts in DTXFiles are not touched)
Type: filesandordirs; Name: "{app}\vendor"
Type: filesandordirs; Name: "{app}\cache"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\python"
Type: files; Name: "{app}\settings.json"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\app\__pycache__"
