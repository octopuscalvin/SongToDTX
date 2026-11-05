; Song to DTX update patch (Inno Setup 6): the app and its charting rules only, over an existing
; install of either edition. Build it from a folder made by make_patch.py:
;   ISCC /DStage=<patch dir> /DOut=<output dir> SongToDTXPatch.iss
; Your settings, kit, cache and charts are not touched.

#define AppVer Trim(FileRead(FileOpen(Stage + "\VERSION")))
#define LibVer Trim(FileRead(FileOpen(Stage + "\LIBRARY")))

[Setup]
; the full installer's AppId: the patch goes into the same folder and updates its entry in Apps
AppId={{6E3C1B52-8D0A-4C1E-9B7A-5D2F3A9C7E41}
AppName=Song to DTX
AppVersion={#AppVer} (Library {#LibVer})
AppVerName=Song to DTX {#AppVer} (Library {#LibVer})
AppPublisher=Song to DTX (hobby project)
DefaultDirName={autopf}\Song to DTX
UsePreviousAppDir=yes
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#Out}
OutputBaseFilename=SongToDTX-{#AppVer}-Library-{#LibVer}-Patch
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#Stage}\app\icon.ico
UninstallDisplayIcon={app}\app\icon.ico
CloseApplications=yes

[Messages]
WelcomeLabel2=This updates Song to DTX to {#AppVer} with charting library {#LibVer}.%n%nOnly the app and its charting rules are replaced. Your settings, drum kit, cache and charts stay as they are.%n%nClose Song to DTX before you continue.

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

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
; compiled copies of the old code
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\app\__pycache__"

[Code]
function InstalledDir(): String;
begin
  Result := '';
  if not RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{6E3C1B52-8D0A-4C1E-9B7A-5D2F3A9C7E41}_is1',
                             'InstallLocation', Result) then
    RegQueryStringValue(HKLM, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{6E3C1B52-8D0A-4C1E-9B7A-5D2F3A9C7E41}_is1',
                        'InstallLocation', Result);
end;

function InitializeSetup(): Boolean;
var
  Dir: String;
begin
  Dir := InstalledDir();
  Result := (Dir <> '') and FileExists(AddBackslash(Dir) + 'python\python.exe');
  if not Result then
    MsgBox('Song to DTX is not installed on this PC.' #13#10#13#10 +
           'This is an update patch. Install the full Song to DTX installer (NVIDIA or CPU edition) first.', mbError, MB_OK);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Old, New: AnsiString;
begin
  Result := '';
  ExtractTemporaryFile('requirements.txt');
  if LoadStringFromFile(ExpandConstant('{app}\requirements.txt'), Old) and
     LoadStringFromFile(ExpandConstant('{tmp}\requirements.txt'), New) and (Old <> New) then
    Result := 'This update needs new Python packages, which a patch cannot add. ' +
              'Please download the full Song to DTX installer instead.';
end;
