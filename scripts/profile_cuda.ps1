[CmdletBinding()]
param(
    [ValidateSet("ncu", "nsys")]
    [string]$Tool = "ncu",
    [string]$TestAppPath,
    [string]$DllDir,
    [string[]]$AppArgs = @(),
    [string]$ProfilerPath,
    [string]$ResultRoot,
    [string]$ResultBase,
    [string]$WorkingDirectory,
    [string]$BuildPreset = "windows-test-app-release",
    [switch]$BuildIfMissing,
    [string]$NcuSet,
    [string]$KernelName,
    [string]$KernelId,
    [string]$KernelNameBase = "function",
    [int]$LaunchSkip = 0,
    [int]$LaunchCount = 1,
    [string[]]$NcuArgs = @(),
    [switch]$Assembly,
    [ValidateSet("sass", "ptx", "cuda,sass")]
    [string]$AssemblySource = "sass",
    [string[]]$NcuSourceFolders = @(),
    [string]$NsysTrace = "cuda,nvtx",
    [string]$NsysSample = "none",
    [string]$NsysCpuContextSwitch = "none",
    [string[]]$NsysArgs = @(),
    [switch]$Stats,
    [switch]$ProfilerOutput,
    [switch]$UseProfilerConfig,
    [switch]$DryRun,
    [switch]$ShowCommand,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))

function Show-Usage {
    @"
Usage:
  .\scripts\profile_cuda.ps1
  .\scripts\profile_cuda.ps1 -KernelName regex:branched
  .\scripts\profile_cuda.ps1 -Tool nsys
  .\scripts\profile_cuda.ps1 -BuildIfMissing
  .\scripts\profile_cuda.ps1 -DryRun

Important options:
  -Tool ncu                         Use Nsight Compute for kernel metrics.
  -Tool nsys                        Use Nsight Systems for a CUDA timeline.
  -KernelName regex:name            Nsight Compute kernel-name filter.
  -LaunchSkip 0                     Matching kernel launches to skip.
  -LaunchCount 1                    Matching kernel launches to profile.
  -NcuSet full                      Optional Nsight Compute metric set.
  -Assembly                         Write Nsight Compute source or assembly text output.
  -AssemblySource sass              Source view for -Assembly: sass, ptx, or cuda,sass.
  -NcuSourceFolders <dirs>          Source folders to import when -Assembly is set.
  -NsysTrace cuda,nvtx              Nsight Systems trace list.
  -ResultRoot <dir>                 Parent directory, default out\cuda-profile.
  -Stats                            Print Nsight Systems stats tables.
  -ProfilerOutput                   Show raw profiler and target application output.
  -BuildIfMissing                   Build the test app before profiling if needed.
  -DryRun                           Print commands without profiling.
  -ShowCommand                      Print full profiler command during a real run.

The test app receives the DLL directory as its first argument.
Relative paths are resolved from the repository root.
Nsight tools may need an unrestricted process launch. Nsight Compute also
needs NVIDIA GPU performance counter permission for kernel metrics.
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

function Find-FirstPathFromGlobs {
    param([string[]]$Globs)

    foreach ($glob in $Globs) {
        $match = Get-ChildItem -Path $glob -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -ne $match) {
            return $match.FullName
        }
    }

    return $null
}

function Resolve-NvidiaProfiler {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RequestedTool,
        [string]$RequestedPath,
        [bool]$Required
    )

    if ($RequestedPath) {
        return Resolve-Executable -Name $RequestedPath -Required $Required
    }

    $exeName = if ($RequestedTool -eq "ncu") { "ncu" } else { "nsys" }
    $command = Get-Command $exeName -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }

    $programFiles = @(
        [Environment]::GetEnvironmentVariable("ProgramFiles"),
        [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    ) | Where-Object { $_ }

    $globs = @()
    foreach ($root in $programFiles) {
        if ($RequestedTool -eq "ncu") {
            $globs += (Join-Path $root "NVIDIA Corporation\Nsight Compute *\ncu.exe")
            $globs += (Join-Path $root "NVIDIA Corporation\Nsight Compute *\target\windows-*\ncu.exe")
        }
        else {
            $globs += (Join-Path $root "NVIDIA Corporation\Nsight Systems *\nsys.exe")
            $globs += (Join-Path $root "NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe")
        }
    }

    $found = Find-FirstPathFromGlobs -Globs $globs
    if ($found) {
        return $found
    }

    if ($Required) {
        throw "Could not find '$exeName'. Put it on PATH or pass -ProfilerPath."
    }

    return $exeName
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

if ($LaunchSkip -lt 0) {
    throw "-LaunchSkip must be non-negative."
}
if ($LaunchCount -lt 0) {
    throw "-LaunchCount must be non-negative."
}
if ($Assembly -and $Tool -ne "ncu") {
    throw "-Assembly is only supported with -Tool ncu."
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
    $ResultRoot = Join-Path $repoRoot "out\cuda-profile"
}
$resultRootFull = ConvertTo-FullPath -Path $ResultRoot -BasePath $repoRoot

if (-not $ResultBase) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $ResultBase = "$Tool-$timestamp"
}
$resultBaseFull = ConvertTo-FullPath -Path $ResultBase -BasePath $resultRootFull
$resultParent = Split-Path -Parent $resultBaseFull

$profilerExe = Resolve-NvidiaProfiler -RequestedTool $Tool -RequestedPath $ProfilerPath -Required (-not $DryRun)
$targetArgs = @($dllDirFull) + $AppArgs
$assemblyOutputPath = $null
$assemblyArgs = @()

if ($Tool -eq "ncu") {
    $profilerArgs = @("--force-overwrite")
    if (-not $ProfilerOutput) {
        $profilerArgs += @("--quiet")
    }
    if (-not $UseProfilerConfig) {
        $profilerArgs += @("--config-file", "off")
    }
    $profilerArgs += @("--export", $resultBaseFull)
    if ($NcuSet) {
        $profilerArgs += @("--set", $NcuSet)
    }
    if ($KernelNameBase) {
        $profilerArgs += @("--kernel-name-base", $KernelNameBase)
    }
    if ($KernelName) {
        $profilerArgs += @("--kernel-name", $KernelName)
    }
    if ($KernelId) {
        $profilerArgs += @("--kernel-id", $KernelId)
    }
    if ($LaunchSkip -gt 0) {
        $profilerArgs += @("--launch-skip", [string]$LaunchSkip)
    }
    if ($LaunchCount -gt 0) {
        $profilerArgs += @("--launch-count", [string]$LaunchCount)
    }
    if ($Assembly) {
        if ($NcuSourceFolders.Count -eq 0) {
            $NcuSourceFolders = @("siglib", "plugins\cuda")
        }
        $ncuSourceFoldersFull = Resolve-ExistingPathList -Paths $NcuSourceFolders
        $profilerArgs += @("--import-source", "yes")
        if ($ncuSourceFoldersFull.Count -gt 0) {
            $profilerArgs += @("--source-folders", ($ncuSourceFoldersFull -join ","))
        }
    }
    $profilerArgs += $NcuArgs
    $profilerArgs += @($testAppFull)
    $profilerArgs += $targetArgs
    $expectedFiles = @("$resultBaseFull.ncu-rep")
    if ($Assembly) {
        $safeAssemblySource = $AssemblySource -replace '[^\w.-]', '_'
        $assemblyOutputPath = "$resultBaseFull.source-$safeAssemblySource.txt"
        $expectedFiles += $assemblyOutputPath
        $assemblyArgs = @(
            "--config-file", "off",
            "--import", "$resultBaseFull.ncu-rep",
            "--page", "source",
            "--print-source", $AssemblySource,
            "--log-file", $assemblyOutputPath
        )
    }
}
else {
    $statsValue = if ($Stats) { "true" } else { "false" }
    $profilerArgs = @(
        "profile",
        "--trace=$NsysTrace",
        "--stats=$statsValue",
        "--force-overwrite=true",
        "--output", $resultBaseFull
    )
    if ($NsysSample) {
        $profilerArgs += @("--sample=$NsysSample")
    }
    if ($NsysCpuContextSwitch) {
        $profilerArgs += @("--cpuctxsw=$NsysCpuContextSwitch")
    }
    $profilerArgs += $NsysArgs
    $profilerArgs += @($testAppFull)
    $profilerArgs += $targetArgs
    $expectedFiles = @("$resultBaseFull.nsys-rep")
    if ($Stats) {
        $expectedFiles += "$resultBaseFull.sqlite"
    }
}

Write-Host "CUDA profile"
Write-Host "Tool: $Tool"
Write-Host "Target: $(Format-DisplayPath $testAppFull)"
Write-Host "DLLs: $(Format-DisplayPath $dllDirFull)"
Write-Host "Workdir: $(Format-DisplayPath $workingDirectoryFull)"
if ($Tool -eq "ncu") {
    $kernelFilter = if ($KernelName) { $KernelName } elseif ($KernelId) { $KernelId } else { "all" }
    Write-Host "Filter: kernel=$kernelFilter, skip=$LaunchSkip, count=$LaunchCount"
}
else {
    Write-Host "Trace: $NsysTrace, stats=$Stats"
}
Write-Host "Outputs:"
foreach ($path in $expectedFiles) {
    Write-Host "  $(Format-DisplayPath $path)"
}

if ($DryRun -or $ShowCommand) {
    Write-Host "Profiler command:"
    Write-Host (Format-CommandLine -Command $profilerExe -Arguments $profilerArgs)
    if ($Assembly) {
        Write-Host "Assembly command:"
        Write-Host (Format-CommandLine -Command $profilerExe -Arguments $assemblyArgs)
    }
}

if ($DryRun) {
    exit 0
}

New-Item -ItemType Directory -Force -Path $resultParent | Out-Null

$oldPath = $env:PATH

try {
    $testAppDir = Split-Path -Parent $testAppFull
    $env:PATH = (@($dllDirFull, $testAppDir) + $oldPath) -join [System.IO.Path]::PathSeparator

    Push-Location $workingDirectoryFull
    try {
        Invoke-ExternalCommand -Command $profilerExe -Arguments $profilerArgs -FailureMessage "$Tool failed"
        if ($Assembly) {
            Invoke-ExternalCommand -Command $profilerExe -Arguments $assemblyArgs -FailureMessage "$Tool assembly report failed"
        }
        Write-Host "CUDA profile complete."
    }
    finally {
        Pop-Location
    }
}
finally {
    $env:PATH = $oldPath
}
