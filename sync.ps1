# Pull / push / status for the code repo (26_UROP) and the private notes repo (notes/) together.
# Usage: .\sync.ps1 [pull|push|status]   (default: status)
# If scripts are blocked: powershell -ExecutionPolicy Bypass -File .\sync.ps1 pull
param([ValidateSet('pull', 'push', 'status')][string]$Cmd = 'status')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$repos = @('.')
if (Test-Path 'notes/.git') { $repos += 'notes' }
else { Write-Host '[warn] notes/ repo not found - processing the code repo only' }

foreach ($r in $repos) {
    Write-Host "== $r"
    switch ($Cmd) {
        'pull' {
            git -C $r pull --ff-only
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        'push' {
            $dirty = git -C $r status --porcelain
            if ($dirty) {
                Write-Host '  Uncommitted changes - commit first:'
                git -C $r status --short
                exit 1
            }
            git -C $r push
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        'status' { git -C $r status --short --branch }
    }
}
