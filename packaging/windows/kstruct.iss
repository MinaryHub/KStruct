; Inno Setup script: builds KStruct-Setup-<version>.exe from dist\KStruct
; Build: iscc /DAppVersion=0.1.0 packaging\windows\kstruct.iss

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6E0F3C9B-6C1E-4C55-9A0E-4B2E5C1F7A11}
AppName=KStruct
AppVersion={#AppVersion}
AppPublisher=MinaryHub
DefaultDirName={autopf}\KStruct
DefaultGroupName=KStruct
OutputDir=..\..\dist
OutputBaseFilename=KStruct-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
LicenseFile=..\..\LICENSE

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "assockst"; Description: "KStruct 프로젝트 파일(.kst)을 KStruct로 열기"; GroupDescription: "파일 연결:"

[Files]
Source: "..\..\dist\KStruct\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"
Source: "..\..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"

[Icons]
Name: "{group}\KStruct"; Filename: "{app}\KStruct.exe"
Name: "{group}\{cm:UninstallProgram,KStruct}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\KStruct"; Filename: "{app}\KStruct.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.kst"; ValueType: string; ValueData: "KStruct.Project"; Flags: uninsdeletevalue; Tasks: assockst
Root: HKA; Subkey: "Software\Classes\KStruct.Project"; ValueType: string; ValueData: "KStruct 프로젝트"; Flags: uninsdeletekey; Tasks: assockst
Root: HKA; Subkey: "Software\Classes\KStruct.Project\shell\open\command"; ValueType: string; ValueData: """{app}\KStruct.exe"" ""%1"""; Tasks: assockst

[Run]
Filename: "{app}\KStruct.exe"; Description: "{cm:LaunchProgram,KStruct}"; Flags: nowait postinstall skipifsilent
; in-app update (setup run with /SILENT): start the new version again when done
Filename: "{app}\KStruct.exe"; Flags: nowait skipifnotsilent
