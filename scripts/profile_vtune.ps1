[CmdletBinding()]
param(
    [string]$TestAppPath,
    [string]$DllDir,
    [string[]]$AppArgs = @(),
    [string]$VtunePath,
    [string]$Collection = "hotspots",
    [string]$ResultRoot,
    [string]$ResultDir,
    [string[]]$SearchDir = @(),
    [string[]]$SourceSearchDir = @(),
    [string[]]$Knob = @(),
    [string]$ReportType = "hotspots",
    [string]$GroupBy = "function",
    [string]$LineGroupBy = "source-function,source-line",
    [string]$AssemblyObject,
    [string]$AssemblyGroupBy = "address",
    [ValidateSet("text", "csv")]
    [string]$ReportFormat = "text",
    [int]$ReportWidth = 160,
    [string]$WorkingDirectory,
    [string]$BuildPreset = "windows-test-app-release",
    [switch]$BuildIfMissing,
    [switch]$NoReport,
    [switch]$DryRun,
    [switch]$ShowCommand,
    [switch]$ProfilerOutput,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))

function Show-Usage {
    @"
Usage:
  .\scripts\profile_vtune.ps1
  .\scripts\profile_vtune.ps1 -BuildIfMissing
  .\scripts\profile_vtune.ps1 -TestAppPath .\out\build\windows-cpp-solution\siglib\test_app\Release\pysiglib_test_app.exe
  .\scripts\profile_vtune.ps1 -DllDir .\pysiglib -AppArgs @("extra", "test", "args")

Important options:
  -Collection hotspots              VTune collection type.
  -ResultRoot <dir>                 Parent directory for VTune results, default out\vtune.
  -ResultDir <dir>                  Exact VTune result directory.
  -Knob @("name=value")             Extra VTune collection knobs.
  -GroupBy function                 Report grouping.
  -LineGroupBy source-function,source-line
                                    Extra line-level report grouping. Use "" to disable.
  -AssemblyObject function=name     Generate assembly view for a VTune source object.
  -AssemblyGroupBy address          Assembly report grouping.
  -BuildIfMissing                   Build the test app before profiling if needed.
  -NoReport                         Skip text or CSV report generation.
  -DryRun                           Print commands without running VTune.
  -ShowCommand                      Print full VTune commands during a real run.
  -ProfilerOutput                   Show raw VTune and target application output.

The test app receives the DLL directory as its first argument.
Relative paths are resolved from the repository root.
"@ | Write-Host
}

function ConvertTo-FullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$BasePath
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

function Format-DisplayPath {
    param([string]$Path)

    if (-not $Path) {
        return $Path
    }

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $root = $repoRoot.TrimEnd('\', '/')
    if ($fullPath.Equals($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        return "."
    }
    if ($fullPath.StartsWith($root + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
        return $fullPath.Substring($root.Length + 1)
    }

    return $fullPath
}

function Resolve-Executable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [bool]$Required
    )

    if (Test-Path -LiteralPath $Name -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Name).ProviderPath
    }

    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }

    if ($Required) {
        throw "Could not find executable '$Name'."
    }

    return $Name
}

function Resolve-VTune {
    param(
        [string]$RequestedPath,
        [bool]$Required
    )

    if ($RequestedPath) {
        return Resolve-Executable -Name $RequestedPath -Required $Required
    }

    $command = Get-Command "vtune" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }

    $candidates = @()
    if ($env:VTUNE_PROFILER_DIR) {
        $candidates += (Join-Path $env:VTUNE_PROFILER_DIR "bin64\vtune.exe")
    }
    if ($env:ONEAPI_ROOT) {
        $candidates += (Join-Path $env:ONEAPI_ROOT "vtune\latest\bin64\vtune.exe")
    }
    $candidates += "C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"
    $candidates += "C:\Program Files\Intel\oneAPI\vtune\latest\bin64\vtune.exe"

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).ProviderPath
        }
    }

    if ($Required) {
        throw "VTune CLI was not found. Put vtune on PATH or pass -VtunePath."
    }

    return "vtune"
}

function Get-TestAppCandidates {
    return @(
        "out\build\windows-cpp-solution\siglib\test_app\Release\pysiglib_test_app.exe",
        "out\build\windows-cpp-solution\siglib\test_app\RelWithDebInfo\pysiglib_test_app.exe",
        "out\build\windows-cpp-solution\siglib\test_app\Debug\pysiglib_test_app.exe"
    )
}

function Resolve-TestAppPath {
    param(
        [string]$RequestedPath,
        [bool]$Required
    )

    if ($RequestedPath) {
        $fullPath = ConvertTo-FullPath -Path $RequestedPath -BasePath $repoRoot
        if ((Test-Path -LiteralPath $fullPath -PathType Leaf) -or -not $Required) {
            return $fullPath
        }
        throw "Test app executable does not exist: $fullPath"
    }

    foreach ($candidate in Get-TestAppCandidates) {
        $fullPath = ConvertTo-FullPath -Path $candidate -BasePath $repoRoot
        if (Test-Path -LiteralPath $fullPath -PathType Leaf) {
            return (Resolve-Path -LiteralPath $fullPath).ProviderPath
        }
    }

    $fallback = ConvertTo-FullPath -Path (Get-TestAppCandidates | Select-Object -First 1) -BasePath $repoRoot
    if ($Required) {
        throw "Could not find pysiglib_test_app.exe. Run with -BuildIfMissing or pass -TestAppPath."
    }
    return $fallback
}

function Test-DllDirectory {
    param([string]$Path)

    return (
        (Test-Path -LiteralPath (Join-Path $Path "cpsig.dll") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path "cusig.dll") -PathType Leaf)
    )
}

function Resolve-DllDirectory {
    param(
        [string]$RequestedPath,
        [string]$ResolvedTestAppPath,
        [bool]$Required
    )

    if ($RequestedPath) {
        $fullPath = ConvertTo-FullPath -Path $RequestedPath -BasePath $repoRoot
        if ((Test-Path -LiteralPath $fullPath -PathType Container) -or -not $Required) {
            return $fullPath
        }
        throw "DLL directory does not exist: $fullPath"
    }

    $testAppDir = Split-Path -Parent $ResolvedTestAppPath
    $candidates = @(
        $testAppDir,
        (Join-Path $repoRoot "pysiglib")
    )

    foreach ($candidate in $candidates) {
        if ((Test-Path -LiteralPath $candidate -PathType Container) -and (Test-DllDirectory -Path $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).ProviderPath
        }
    }

    if ($Required) {
        throw "Could not find a DLL directory containing cpsig.dll and cusig.dll."
    }

    return $testAppDir
}

function Format-CommandArg {
    param([string]$Value)

    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }

    return $Value
}

function Format-CommandLine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    return ((@($Command) + $Arguments) | ForEach-Object { Format-CommandArg $_ }) -join " "
}

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    if ($ProfilerOutput) {
        & $Command @Arguments
        $exitCode = $LASTEXITCODE
    }
    else {
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & $Command @Arguments 2>&1
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }
    }

    if ($exitCode -ne 0) {
        if ((-not $ProfilerOutput) -and $output) {
            Write-Host "Profiler output tail:"
            $output | Select-Object -Last 40 | ForEach-Object { Write-Host $_ }
        }
        throw "$FailureMessage with exit code $exitCode."
    }
}

function Resolve-ExistingPathList {
    param([string[]]$Paths)

    $resolved = @()
    foreach ($path in $Paths) {
        $fullPath = ConvertTo-FullPath -Path $path -BasePath $repoRoot
        if (Test-Path -LiteralPath $fullPath) {
            $resolved += (Resolve-Path -LiteralPath $fullPath).ProviderPath
        }
    }
    return @($resolved | Select-Object -Unique)
}

function Invoke-TestAppBuild {
    param([bool]$ShouldRun)

    $cmakeExe = Resolve-Executable -Name "cmake" -Required $ShouldRun
    $buildArgs = @("--build", "--preset", $BuildPreset)
    Write-Host "CMake build command:"
    Write-Host (Format-CommandLine -Command $cmakeExe -Arguments $buildArgs)

    if ($ShouldRun) {
        Push-Location $repoRoot
        try {
            & $cmakeExe @buildArgs
            if ($LASTEXITCODE -ne 0) {
                throw "CMake build failed with exit code $LASTEXITCODE."
            }
        }
        finally {
            Pop-Location
        }
    }
}

if ($Help) {
    Show-Usage
    exit 0
}

$testAppRequired = (-not $DryRun) -and (-not $BuildIfMissing)
$testAppFull = Resolve-TestAppPath -RequestedPath $TestAppPath -Required $testAppRequired
if ($BuildIfMissing -and (-not (Test-Path -LiteralPath $testAppFull -PathType Leaf))) {
    Invoke-TestAppBuild -ShouldRun (-not $DryRun)
    $testAppFull = Resolve-TestAppPath -RequestedPath $TestAppPath -Required (-not $DryRun)
}

$dllDirFull = Resolve-DllDirectory -RequestedPath $DllDir -ResolvedTestAppPath $testAppFull -Required (-not $DryRun)
if ((-not $DryRun) -and (-not (Test-DllDirectory -Path $dllDirFull))) {
    throw "DLL directory must contain cpsig.dll and cusig.dll: $dllDirFull"
}

if (-not $WorkingDirectory) {
    $WorkingDirectory = Split-Path -Parent $testAppFull
}
$workingDirectoryFull = ConvertTo-FullPath -Path $WorkingDirectory -BasePath $repoRoot
if ((-not $DryRun) -and (-not (Test-Path -LiteralPath $workingDirectoryFull -PathType Container))) {
    throw "Working directory does not exist: $workingDirectoryFull"
}

if (-not $ResultRoot) {
    $ResultRoot = Join-Path $repoRoot "out\vtune"
}

$resultRootFull = ConvertTo-FullPath -Path $ResultRoot -BasePath $repoRoot
if ($ResultDir) {
    $resultDirFull = ConvertTo-FullPath -Path $ResultDir -BasePath $repoRoot
}
else {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $resultDirFull = Join-Path $resultRootFull "vtune-$Collection-$timestamp"
}

if ((Test-Path -LiteralPath $resultDirFull) -and -not $DryRun) {
    throw "Result directory already exists: $resultDirFull"
}

$vtuneExe = Resolve-VTune -RequestedPath $VtunePath -Required (-not $DryRun)

if ($SearchDir.Count -eq 0) {
    $SearchDir = @(
        $dllDirFull,
        (Split-Path -Parent $testAppFull),
        "out\build",
        "siglib"
    )
}
$searchDirsFull = Resolve-ExistingPathList -Paths $SearchDir

if ($SourceSearchDir.Count -eq 0) {
    $SourceSearchDir = @(
        ".",
        "siglib",
        "siglib\test_app"
    )
}
$sourceSearchDirsFull = Resolve-ExistingPathList -Paths $SourceSearchDir

$targetArgs = @($dllDirFull) + $AppArgs
$collectArgs = @(
    "-collect", $Collection,
    "-result-dir", $resultDirFull,
    "-quiet",
    "-return-app-exitcode"
)
foreach ($dir in $searchDirsFull) {
    $collectArgs += @("-search-dir", $dir)
}
foreach ($value in $Knob) {
    $collectArgs += @("-knob", $value)
}
$collectArgs += @("--", $testAppFull)
$collectArgs += $targetArgs

$reportPaths = @()
$primaryReportPath = $null
$lineReportPath = $null
$assemblyReportPath = $null
$summaryPath = $null
if (-not $NoReport) {
    $reportExtension = if ($ReportFormat -eq "csv") { "csv" } else { "txt" }
    $safeGroupBy = $GroupBy -replace '[^\w.-]', '_'
    $primaryReportPath = Join-Path $resultDirFull ("$ReportType-by-$safeGroupBy.$reportExtension")
    $reportPaths += $primaryReportPath
    if ($LineGroupBy) {
        $safeLineGroupBy = $LineGroupBy -replace '[^\w.-]', '_'
        if ($safeLineGroupBy -ne $safeGroupBy) {
            $lineReportPath = Join-Path $resultDirFull ("$ReportType-by-$safeLineGroupBy.$reportExtension")
            $reportPaths += $lineReportPath
        }
    }
    if ($AssemblyObject) {
        $safeAssemblyObject = $AssemblyObject -replace '[^\w.-]', '_'
        $assemblyReportPath = Join-Path $resultDirFull ("$ReportType-assembly-$safeAssemblyObject.$reportExtension")
        $reportPaths += $assemblyReportPath
    }
    $summaryPath = Join-Path $resultDirFull ("summary.$reportExtension")
    $reportPaths += $summaryPath
}

Write-Host "VTune profile"
Write-Host "Target: $(Format-DisplayPath $testAppFull)"
Write-Host "DLLs: $(Format-DisplayPath $dllDirFull)"
Write-Host "Workdir: $(Format-DisplayPath $workingDirectoryFull)"
Write-Host "Result: $(Format-DisplayPath $resultDirFull)"
if ($NoReport) {
    Write-Host "Reports: disabled"
}
elseif ($reportPaths.Count -gt 0) {
    Write-Host "Reports:"
    foreach ($path in $reportPaths) {
        Write-Host "  $(Format-DisplayPath $path)"
    }
}

if ($DryRun -or $ShowCommand) {
    Write-Host "Collect command:"
    Write-Host (Format-CommandLine -Command $vtuneExe -Arguments $collectArgs)
}

if ($DryRun) {
    exit 0
}

$resultParent = Split-Path -Parent $resultDirFull
New-Item -ItemType Directory -Force -Path $resultParent | Out-Null

$oldPath = $env:PATH

try {
    $testAppDir = Split-Path -Parent $testAppFull
    $env:PATH = (@($dllDirFull, $testAppDir) + $oldPath) -join [System.IO.Path]::PathSeparator

    Push-Location $workingDirectoryFull
    try {
        Invoke-ExternalCommand -Command $vtuneExe -Arguments $collectArgs -FailureMessage "VTune collect failed"

        if ($NoReport) {
            Write-Host "VTune collection complete."
        }
        else {
            $reportArgs = @(
                "-report", $ReportType,
                "-result-dir", $resultDirFull,
                "-group-by", $GroupBy,
                "-report-output", $primaryReportPath
            )
            if ($ReportFormat -ne "text") {
                $reportArgs += @("-format", $ReportFormat)
            }
            if (($ReportFormat -eq "text") -and ($ReportWidth -gt 0)) {
                $reportArgs += @("-report-width", $ReportWidth)
            }
            foreach ($dir in $sourceSearchDirsFull) {
                $reportArgs += @("-source-search-dir", $dir)
            }
            foreach ($dir in $searchDirsFull) {
                $reportArgs += @("-search-dir", $dir)
            }

            if ($ShowCommand) {
                Write-Host "Report command:"
                Write-Host (Format-CommandLine -Command $vtuneExe -Arguments $reportArgs)
            }
            Invoke-ExternalCommand -Command $vtuneExe -Arguments $reportArgs -FailureMessage "VTune report failed"

            if ($lineReportPath) {
                $lineReportArgs = @(
                    "-report", $ReportType,
                    "-result-dir", $resultDirFull,
                    "-group-by", $LineGroupBy,
                    "-report-output", $lineReportPath
                )
                if ($ReportFormat -ne "text") {
                    $lineReportArgs += @("-format", $ReportFormat)
                }
                if (($ReportFormat -eq "text") -and ($ReportWidth -gt 0)) {
                    $lineReportArgs += @("-report-width", $ReportWidth)
                }
                foreach ($dir in $sourceSearchDirsFull) {
                    $lineReportArgs += @("-source-search-dir", $dir)
                }
                foreach ($dir in $searchDirsFull) {
                    $lineReportArgs += @("-search-dir", $dir)
                }

                if ($ShowCommand) {
                    Write-Host "Line report command:"
                    Write-Host (Format-CommandLine -Command $vtuneExe -Arguments $lineReportArgs)
                }
                Invoke-ExternalCommand -Command $vtuneExe -Arguments $lineReportArgs -FailureMessage "VTune line report failed"
            }

            if ($assemblyReportPath) {
                $assemblyReportArgs = @(
                    "-report", $ReportType,
                    "-result-dir", $resultDirFull,
                    "-source-object", $AssemblyObject,
                    "-group-by", $AssemblyGroupBy,
                    "-report-output", $assemblyReportPath
                )
                if ($ReportFormat -ne "text") {
                    $assemblyReportArgs += @("-format", $ReportFormat)
                }
                if (($ReportFormat -eq "text") -and ($ReportWidth -gt 0)) {
                    $assemblyReportArgs += @("-report-width", $ReportWidth)
                }
                foreach ($dir in $sourceSearchDirsFull) {
                    $assemblyReportArgs += @("-source-search-dir", $dir)
                }
                foreach ($dir in $searchDirsFull) {
                    $assemblyReportArgs += @("-search-dir", $dir)
                }

                if ($ShowCommand) {
                    Write-Host "Assembly report command:"
                    Write-Host (Format-CommandLine -Command $vtuneExe -Arguments $assemblyReportArgs)
                }
                Invoke-ExternalCommand -Command $vtuneExe -Arguments $assemblyReportArgs -FailureMessage "VTune assembly report failed"
            }

            $summaryArgs = @(
                "-report", "summary",
                "-result-dir", $resultDirFull,
                "-report-output", $summaryPath
            )
            if ($ReportFormat -ne "text") {
                $summaryArgs += @("-format", $ReportFormat)
            }
            if (($ReportFormat -eq "text") -and ($ReportWidth -gt 0)) {
                $summaryArgs += @("-report-width", $ReportWidth)
            }

            if ($ShowCommand) {
                Write-Host "Summary command:"
                Write-Host (Format-CommandLine -Command $vtuneExe -Arguments $summaryArgs)
            }
            Invoke-ExternalCommand -Command $vtuneExe -Arguments $summaryArgs -FailureMessage "VTune summary failed"

            Write-Host "VTune reports written."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    $env:PATH = $oldPath
}
