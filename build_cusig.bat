@echo off

set "NVCC_EXE=%CUDA_PATH%\bin\nvcc.exe"
set "VS_PATH0=%~1"
set "VS_PATH=%~2"
set "CL_EXE=%VS_PATH%\bin\HostX64\x64\cl.exe"
set "LINK_EXE=%VS_PATH%\bin\HostX64\x64\link.exe"

set "SIGLIB_DIR=%cd%\siglib"

@echo on
REM set env variables for 64b c++
call "%VS_PATH0%\Auxiliary\Build\vcvars64.bat"

setlocal enabledelayedexpansion

REM set current dir
CD "%SIGLIB_DIR%\cusig"

@echo *** Detecting supported GPU architectures ***

REM Query nvcc for supported architectures (available since CUDA 11.5)
set "NVCC_GENCODE="
set "LAST_ARCH="
set "ARCHS_TMP=%TEMP%\nvcc_archs_%RANDOM%.txt"

"%NVCC_EXE%" --list-gpu-arch 2>nul | findstr /b "compute_" > "!ARCHS_TMP!"

for /f "tokens=2 delims=_" %%a in (!ARCHS_TMP!) do (
    set "NVCC_GENCODE=!NVCC_GENCODE! -gencode=arch=compute_%%a,code=sm_%%a"
    set "LAST_ARCH=%%a"
)

del "!ARCHS_TMP!" 2>nul

if not defined LAST_ARCH (
    @echo Error: failed to detect supported GPU architectures from nvcc.
    exit /b 1
)

REM Add PTX for the highest architecture (forward compatibility)
set "NVCC_GENCODE=!NVCC_GENCODE! -gencode=arch=compute_!LAST_ARCH!,code=compute_!LAST_ARCH!"

@echo PTX forward-compat: compute_!LAST_ARCH!


@echo build pch

md x64 2>nul
cd x64
md Release 2>nul

CD "%SIGLIB_DIR%\cusig"

"%CL_EXE%" /c /I"%CUDA_PATH%\include" /Zi /nologo /W3 /WX- /diagnostics:column /sdl /O2 /Oi /GL /D NDEBUG /D CUSIG_EXPORTS /D _WINDOWS /D _USRDLL /D _WINDLL /D _UNICODE /D UNICODE /Gm- /EHsc /MT /GS /Gy /fp:precise /Zc:wchar_t /Zc:forScope /Zc:inline /std:c++20 /permissive- /Yc"cupch.h" /Fp"x64\Release\cusig.pch" /Fo"x64\Release\\" /Fd"x64\Release\vc143.pdb" /external:W3 /Gd /TP /FC /errorReport:prompt cupch.cpp


@echo *** Compiling cuda files with nvcc ***

set NVCC_ARGS= !NVCC_GENCODE! --use-local-env -ccbin "%VS_PATH%\bin\HostX64\x64" -x cu -rdc=true  -I"%CUDA_PATH%\include"    --keep-dir x64\Release  -maxrregcount=0   --machine 64 --compile -cudart static -lineinfo --std c++17  -DNDEBUG -DCUSIG_EXPORTS -D_WINDOWS -D_USRDLL -D_WINDLL -D_UNICODE -DUNICODE -Xcompiler "/EHsc /W3 /nologo /O2 /FS   /MT " -Xcompiler "/Fdx64\Release\vc143.pdb" --dopt on

"%NVCC_EXE%" !NVCC_ARGS! -o "%SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.cu.obj" "%SIGLIB_DIR%\cusig\cu_sig_kernel.cu"

"%NVCC_EXE%" !NVCC_ARGS! -o "%SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.h.obj" "%SIGLIB_DIR%\cusig\cu_sig_kernel.h"

"%NVCC_EXE%" !NVCC_ARGS! -o "%SIGLIB_DIR%\cusig\x64\Release\cu_path_transforms.cu.obj" "%SIGLIB_DIR%\cusig\cu_path_transforms.cu"

"%NVCC_EXE%" !NVCC_ARGS! -o "%SIGLIB_DIR%\cusig\x64\Release\cu_tensor_poly.cu.obj" "%SIGLIB_DIR%\cusig\cu_tensor_poly.cu"

"%NVCC_EXE%" !NVCC_ARGS! -o "%SIGLIB_DIR%\cusig\x64\Release\cu_signature.cu.obj" "%SIGLIB_DIR%\cusig\cu_signature.cu"

"%NVCC_EXE%" !NVCC_ARGS! -o "%SIGLIB_DIR%\cusig\x64\Release\cu_log_signature.cu.obj" "%SIGLIB_DIR%\cusig\cu_log_signature.cu"

"%NVCC_EXE%" !NVCC_ARGS! -o "%SIGLIB_DIR%\cusig\x64\Release\cu_log_sig_cache.cu.obj" "%SIGLIB_DIR%\cusig\cu_log_sig_cache.cu"

"%NVCC_EXE%" !NVCC_ARGS! -o "%SIGLIB_DIR%\cusig\x64\Release\cu_sig_coef.cu.obj" "%SIGLIB_DIR%\cusig\cu_sig_coef.cu"


@echo ---------------------------------------------------------------------------------------
@echo link cuda obj files
REM link cuda obj files
"%NVCC_EXE%" -dlink  -o x64\Release\cusig.device-link.obj -Xcompiler "/EHsc /W3 /nologo /O2   /MT " -Xcompiler "/Fdx64\Release\vc143.pdb" -L"%CUDA_PATH%\bin\crt" -L"%CUDA_PATH%\lib\x64" kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib cudart.lib cudadevrt.lib  !NVCC_GENCODE!  "%SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.h.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_path_transforms.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_tensor_poly.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_signature.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_log_signature.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_log_sig_cache.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_sig_coef.cu.obj"


@echo ---------------------------------------------------------------------------------------
@echo link.exe

REM link final

CD "%SIGLIB_DIR%"

md x64 2>nul
cd x64
md Release 2>nul

CD "%SIGLIB_DIR%\cusig"

set STD_LIBS=kernel32.lib

"%LINK_EXE%" /ERRORREPORT:PROMPT /OUT:"%SIGLIB_DIR%\x64\Release\cusig.dll" /NOLOGO /LIBPATH:"%CUDA_PATH%\lib\x64" %STD_LIBS% cudart.lib cudadevrt.lib /MANIFEST /MANIFESTUAC:NO /manifest:embed /DEBUG /PDB:"%SIGLIB_DIR%\x64\Release\cusig.pdb" /SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF /LTCG:incremental /LTCGOUT:"x64\Release\cusig.iobj" /TLBID:1 /DYNAMICBASE /NXCOMPAT /IMPLIB:"%SIGLIB_DIR%\x64\Release\cusig.lib" /MACHINE:X64 /DLL "%SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.h.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_path_transforms.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_tensor_poly.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_signature.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_log_signature.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_log_sig_cache.cu.obj" "%SIGLIB_DIR%\cusig\x64\Release\cu_sig_coef.cu.obj" x64\Release\cupch.obj  "x64\Release\cusig.device-link.obj"
@echo ---------------------------------------------------------------------------------------

endlocal
