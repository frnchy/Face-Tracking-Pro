#define MyAppName       "Face Tracker Pro"
#define MyAppVersion    "1.0.0"
#define MyAppPublisher  "frnchy"
#define MyAppURL        "https://github.com/frnchy"
#define MyAppExeName    "FaceTrackerPro.exe"
#define MyAppId         "{{B5F8B2E1-7C46-4B89-9D2E-FACETRACK0001}"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
VersionInfoVersion={#MyAppVersion}.0
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} setup
VersionInfoCopyright=(c) 2026 {#MyAppPublisher}

DefaultDirName={autopf}\FaceTrackerPro
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=no
DisableDirPage=no
AllowNoIcons=yes
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
CloseApplications=force
RestartApplications=no

PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline

OutputDir=installer
OutputBaseFilename=FaceTrackerPro_Setup_{#MyAppVersion}
SetupIconFile=assets\brand.ico
Compression=lzma2/ultra
SolidCompression=yes
LZMAUseSeparateProcess=yes

WizardStyle=modern
WizardSizePercent=110
DisableWelcomePage=no
ShowLanguageDialog=no

ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; \
    GroupDescription: "Additional shortcuts:"; Flags: unchecked
Name: "quicklaunchicon"; Description: "Create a &Quick Launch shortcut"; \
    GroupDescription: "Additional shortcuts:"; \
    Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
Source: "exe\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\brand.png"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "assets\brand.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    IconFilename: "{app}\assets\brand.ico"; \
    Comment: "Launch {#MyAppName}"; \
    WorkingDir: "{app}"
Name: "{group}\{#MyAppName} README"; Filename: "{app}\README.md"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    IconFilename: "{app}\assets\brand.ico"; \
    WorkingDir: "{app}"; \
    Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; \
    Filename: "{app}\{#MyAppExeName}"; \
    IconFilename: "{app}\assets\brand.ico"; \
    WorkingDir: "{app}"; \
    Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; \
    Description: "Launch {#MyAppName} now"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\assets"
Type: dirifempty;     Name: "{app}"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
