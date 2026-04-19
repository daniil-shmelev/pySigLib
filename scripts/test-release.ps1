$pypi = Read-Host -Prompt "Install from production PyPI? [y/n] (n = TestPyPI)"
$useTestPyPI = -not ($pypi -match '^[yY]')

$cuda = Read-Host -Prompt "Install with CUDA support? [y/n]"
$installCuda = ($cuda -match '^[yY]')

python -m pip uninstall -y pysiglib pysiglib-cuda

$packages = @('pysiglib')
if ($installCuda) { $packages += 'pysiglib-cuda' }

$indexArgs = @()
if ($useTestPyPI) {
    $indexArgs = @('--index-url', 'https://test.pypi.org/simple/',
                   '--extra-index-url', 'https://pypi.org/simple/')
}

python -m pip install --pre $indexArgs $packages

$repoDir = Join-Path $env:TEMP "pysiglib-rc-tests"
if (Test-Path $repoDir) { Remove-Item -Recurse -Force $repoDir }
git clone --depth 1 --branch main https://github.com/daniil-shmelev/pySigLib.git $repoDir
python -m pip install pytest
python -m pytest (Join-Path $repoDir "tests") -v
Read-Host -Prompt "Press Enter to close"
