; Inno Setup 6 installer script for NeuraBreak
; Download Inno Setup 6 from https://jrsoftware.org/isinfo.php
;
; Build steps:
;   1. Build the app:     uv run pyinstaller packaging/windows/build.spec --noconfirm
;   2. Compile installer: "C:\Program Files (x86)\Inno Setup 6\iscc.exe" packaging\windows\installer.iss
;      or open this file in the Inno Setup IDE and press F9.
;
; Output: dist\installer\NeuraBreak-0.1.0-Setup.exe

#define AppName        "NeuraBreak"
#define AppVersion     "0.1.0"
#define AppPublisher   "NeuraBreak Contributors"
#define AppURL         "https://github.com/abhijeetnishal/neurabreak"
#define AppExeName     "NeuraBreak.exe"
#define AppDescription "AI-Powered Break & Posture Guardian"
#define BuildDir       "..\..\dist\NeuraBreak"
#define SetupIconPath  "icon.ico"

#ifexist SetupIconPath
  #define HasSetupIcon 1
#else
  #define HasSetupIcon 0
#endif

[Setup]
; Unique GUID — do NOT change once the app is released; Windows uses it for
; upgrade detection (old install gets removed automatically on reinstall).
AppId={{A7F3C2E1-8B4D-4A9F-B3E2-1C5D6F7A8B9C}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
AppComments={#AppDescription}
AppReadmeFile={#AppURL}#readme

; Install under the current user's AppData by default so no UAC prompt is
; needed. The user can override to Program Files during setup.
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes

; No admin required for per-user install; admin prompt only if user switches
; to Program Files location.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Silently close the running app before upgrading so files aren't locked
CloseApplications=yes
CloseApplicationsFilter=*.exe
RestartApplications=yes

OutputDir=..\..\dist\installer
OutputBaseFilename=NeuraBreak-{#AppVersion}-Setup

#if HasSetupIcon
SetupIconFile={#SetupIconPath}
#endif
UninstallDisplayIcon={app}\{#AppExeName}

Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMADictionarySize=1048576

WizardStyle=modern

; Minimum Windows version: Windows 10 (build 1809+) for modern toast APIs
MinVersion=10.0.17763

; Show a "Did you know?" sidebar page during installation
DisableWelcomePage=no
DisableDirPage=no
DisableProgramGroupPage=yes
DisableReadyPage=no

; Version comparison — allows in-place upgrades without uninstalling first
VersionInfoVersion={#AppVersion}.0
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppDescription}
VersionInfoCopyright=MIT License


[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon";  Description: "Create a &desktop shortcut";               GroupDescription: "Shortcuts:"; Flags: unchecked
Name: "startupentry"; Description: "Launch NeuraBreak when &Windows starts";   GroupDescription: "Startup:";   Flags: unchecked

[Dirs]
; Ensure the user-data config directory exists at install time
Name: "{localappdata}\.neurabreak"

[Files]
; The PyInstaller one-folder output — everything the app needs
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";             Filename: "{app}\{#AppExeName}"; \
      Comment: "{#AppDescription}"
Name: "{group}\Uninstall {#AppName}";   Filename: "{uninstallexe}"
Name: "{commondesktop}\{#AppName}";     Filename: "{app}\{#AppExeName}"; \
      Comment: "{#AppDescription}"; Tasks: desktopicon

[Registry]
; Optional autostart (only written when the user ticked the startup task).
; Placed under HKCU so no elevation is needed.
Root: HKCU; \
  Subkey:    "Software\Microsoft\Windows\CurrentVersion\Run"; \
  ValueType: string; \
  ValueName: "{#AppName}"; \
  ValueData: """{app}\{#AppExeName}"" --minimized"; \
  Flags:     uninsdeletevalue; \
  Tasks:     startupentry

[Run]
; Launch the app after installation (skippable)
Filename: "{app}\{#AppExeName}"; \
  Description: "Launch {#AppName} now"; \
  Flags: nowait postinstall skipifsilent

[UninstallRun]
; Gracefully ask the running instance to quit before the uninstaller removes files
Filename: "{app}\{#AppExeName}"; Parameters: "--quit"; RunOnceId: "QuitRunningApp"; Flags: nowait skipifdoesntexist
; Give it a moment to exit cleanly
Filename: "ping"; Parameters: "127.0.0.1 -n 3 > nul"; RunOnceId: "UninstallDelay"; Flags: nowait runhidden skipifdoesntexist

[UninstallDelete]
; Remove the user config directory only if the user agrees (see Code section below)
; We intentionally do NOT delete {localappdata}\.neurabreak here automatically —
; health data belongs to the user. The Code section asks them.

[Code]
{ Ask user if they want to delete their health data on uninstall }

function InitializeUninstall(): Boolean;
var
  Answer: Integer;
  DataDir: String;
begin
  Result := True;
  DataDir := ExpandConstant('{localappdata}\.neurabreak');
  if DirExists(DataDir) then
  begin
    Answer := MsgBox(
      'Do you want to delete your NeuraBreak health data and settings?' + #13#10 +
      '(' + DataDir + ')' + #13#10#13#10 +
      'Click Yes to permanently remove all data, No to keep it.',
      mbConfirmation,
      MB_YESNO
    );
    if Answer = IDYES then
      DelTree(DataDir, True, True, True);
  end;
end;

