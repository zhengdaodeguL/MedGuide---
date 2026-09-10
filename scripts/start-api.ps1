$ErrorActionPreference = 'Stop'

# Run from PowerShell 7: ./scripts/start-api.ps1 [-UseSystemProxy] [-Port 8000]
# Only child-process proxy variables change; Windows settings are never edited.
$useSystemProxy = $args -contains '-UseSystemProxy'
$port = 8000
$portIndex = [Array]::IndexOf($args, '-Port')
if ($portIndex -ge 0) {
    if ($portIndex + 1 -ge $args.Count) { throw '-Port requires a number' }
    $port = [int]$args[$portIndex + 1]
}
if ($port -lt 1 -or $port -gt 65535) { throw 'Port must be between 1 and 65535' }
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $projectRoot '.venv/bin/python'
}
if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv and install backend/requirements.txt first' }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) { throw 'Configure .env first' }
$originalProxy = @{}
foreach ($name in @('HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY')) {
    $originalProxy[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location $projectRoot
try {
    if ($useSystemProxy) {
        if (-not $IsWindows) { throw '-UseSystemProxy requires Windows; use HTTP_PROXY/HTTPS_PROXY elsewhere' }
        $settings = Get-ItemProperty -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings'
        if ($settings.ProxyEnable -ne 1 -or -not $settings.ProxyServer) { throw 'No enabled Windows static proxy was found' }
        $proxies = @{}
        foreach ($entry in ($settings.ProxyServer -split ';')) {
            if ($entry -match '^(http|https)=(.+)$') { $proxies[$Matches[1]] = $Matches[2] }
            elseif ($entry -notmatch '=') { $proxies['http'] = $entry; $proxies['https'] = $entry }
        }
        foreach ($protocol in @('http', 'https')) {
            $value = $proxies[$protocol]
            if ($value) {
                if ($value -notmatch '^[a-z]+://') { $value = "http://$value" }
                [Environment]::SetEnvironmentVariable("$($protocol.ToUpper())_PROXY", $value, 'Process')
            }
        }
        $env:NO_PROXY = (@($env:NO_PROXY, 'localhost', '127.0.0.1', '::1') | Where-Object { $_ }) -join ','
        Write-Host 'Using the enabled Windows proxy for this API process.'
    }
    & $python -m uvicorn app.main:app --app-dir backend --env-file .env --host 127.0.0.1 --port $port
    if ($LASTEXITCODE -ne 0) { throw "API exited with code $LASTEXITCODE" }
}
finally {
    foreach ($name in $originalProxy.Keys) { [Environment]::SetEnvironmentVariable($name, $originalProxy[$name], 'Process') }
    Pop-Location
}
