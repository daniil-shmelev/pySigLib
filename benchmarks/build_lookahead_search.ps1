# Copyright 2026 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

# Run from the repo root in an x64 VS developer shell after building cpsig.
param([string]$Build = 'out/build/windows-cpu', [string]$Output = 'output/lookahead_search')
$ErrorActionPreference = 'Stop'
$target = (Resolve-Path "$Build/siglib/cpsig").Path
$dest = (New-Item -ItemType Directory -Force $Output).FullName
$source = (Resolve-Path 'benchmarks/lookahead_search_bridge.cpp').Path
$compile = Get-Content -Encoding Unicode "$target/cpsig_avx2.dir/Release/cpsig_avx2.tlog/CL.command.1.tlog"
$entry = [array]::FindIndex($compile, [Predicate[string]]{ param($s) $s -match '^\^.*\\CP_SIG_LASSO_IMPL\.CPP$' })
if ($entry -lt 0) { throw 'Missing AVX2 compiler flags. Build the Release cpsig target first.' }
$flags = ($compile[$entry + 1] -replace ' /FI"[^"]*" .*$', '') -replace ' /(Yu|Fp|Fo|Fd)"[^"]*"', ''
$flags + " /Fo`"$dest/bridge.obj`" /Fd`"$dest/compile.pdb`" `"$source`"" | Set-Content -Encoding ASCII "$dest/compile.rsp"
$link = @(Get-Content -Encoding Unicode "$target/cpsig.dir/Release/cpsig.tlog/link.command.1.tlog" | Select-Object -Skip 1)
$link[0] = $link[0] -replace '/(OUT|PDB|MAP|IMPLIB):"[^"]*"', ''
$link = $link | Where-Object { $_ -notmatch 'CPSIG_AVX2\.DIR\\RELEASE\\CP_SIG_LASSO_IMPL\.OBJ' }
$link += "`"$dest/bridge.obj`" /OUT:`"$dest/bridge.dll`" /PDB:`"$dest/bridge.pdb`" /IMPLIB:`"$dest/bridge.lib`""
$link | Set-Content -Encoding ASCII "$dest/link.rsp"
Push-Location $target
try {
    & cl "@$dest/compile.rsp"
    if ($LASTEXITCODE) { throw "Compile failed ($LASTEXITCODE)" }
    & link "@$dest/link.rsp"
    if ($LASTEXITCODE) { throw "Link failed ($LASTEXITCODE)" }
} finally { Pop-Location }
