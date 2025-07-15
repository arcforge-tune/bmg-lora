<#
.SYNOPSIS
    Launch Llama‑QLoRA training with only checkpoint‑loading bars, XPU messages, and clean tqdm updates.

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

    # 1) Combined Loading+XPU → split into two lines
    if ($Line -match "^(Loading checkpoint shards:.*?\])") {
        return $Line
    }

    # 2) Standalone “[XPU] Gradient checkpointing enabled”
    if ($Line -match "^\[XPU\].*") {
        return "`n" + $Line
    }

    # 2) Standalone “[XPU] Gradient checkpointing enabled”
    if ($Line -match "(Checkpoint saved|Scheduler state|Resuming):.*") {
        return "`n" + $Line
    }

    # 3) Any tqdm‑style bar: strip off any trailing “[W...”, keep only up through the first closing “]”
    if ($Line -match "^(.*?%\|.*?])(\[.*)$") {
        return $matches[1]
    }
    if ($Line -match "\d+%.*\|") {
        return $Line
    }

    # Everything else → drop
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
    $out = Filter-Line $_
    if ($null -ne $out) {
        # Any line containing “XX%|…” is a live bar → overwrite in place
        if ($out -match "\d+%.*\|") {
            Write-Host "`r$out" -NoNewline
        }
        else {
            Write-Host $out
        }
    }
}

# Final newline so your prompt lands cleanly
Write-Host ""
