<#
.SYNOPSIS
    Launch Llama-QLoRA training with only checkpoint-loading bars, XPU messages, and clean tqdm updates.

.PARAMETER Config
    Path to your YAML config file.

.PARAMETER Resume
    (Optional) Path to resume checkpoint directory.

#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Config,
    [Parameter()][string]$Resume
)

function Filter-Line {
    param([string]$Line)

    # Any tqdm-style bar; strip off trailing “[W…]” junk
    if ($Line -match "^(.*?%\|.*?])(\[.*)$") {
        return $matches[1]
    }
    if ($Line -match "\d+%.*\|") {
        return $Line
    }

    # Standalone “[XPU] Gradient checkpointing enabled”
    if ($Line -match "^\[XPU\] Gradient checkpointing enabled") {
        return $Line
    }

    # Otherwise drop
    return $null
}

# Build python invocation
$pythonExe  = "python"
$scriptPath = Join-Path $PSScriptRoot "src\main.py"
$args       = @("--config", $Config)
if ($Resume) { $args += @("--resume", $Resume) }

# Execute and filter both stdout+stderr
& $pythonExe $scriptPath @args 2>&1 |
  ForEach-Object {
    $line = $_

    # 1) If it's the combined Loading+XPU on one line, split it right here
    if ($line -match "^(Loading checkpoint shards:.*?\])(\[XPU\] Gradient checkpointing enabled)") {
        # print the bar in-place...
        Write-Host "`r$($matches[1])" -NoNewline
        # then a newline so we flush the bar...
        Write-Host ""
        # then the XPU message on its own line
        Write-Host $matches[2]
        return
    }

    # 2) Otherwise let Filter-Line decide
    $out = Filter-Line $line
    if ($null -eq $out) { return }

    # 3) Print bars in-place, everything else normally
    if ($out -match "\d+%.*\|") {
        Write-Host "`r$out" -NoNewline
    }
    else {
        Write-Host $out
    }
  }

# Final newline so your prompt lands cleanly
Write-Host ""
