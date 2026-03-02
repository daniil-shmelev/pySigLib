@echo off

set CUDA_PATH="%CUDA_PATH%"

set NVCC_EXE=%CUDA_PATH%\bin\nvcc.exe
set VS_PATH0=%1
set VS_PATH=%2
set CL_EXE=%VS_PATH%\bin\HostX64\x64\cl.exe
set LINK_EXE=%VS_PATH%\bin\HostX64\x64\link.exe

set SIGLIB_DIR="%cd%\siglib"

@echo on
REM set env variables for 64b c++ 
call %VS_PATH0%\Auxiliary\Build\vcvars64.bat

REM set current dir
rem U:
CD %SIGLIB_DIR%\cusig


@echo build pch

md x64
cd x64
md Release

CD %SIGLIB_DIR%\cusig

%CL_EXE% /c /I%CUDA_PATH%\include /Zi /nologo /W3 /WX- /diagnostics:column /sdl /O2 /Oi /GL /D NDEBUG /D CUSIG_EXPORTS /D _WINDOWS /D _USRDLL /D _WINDLL /D _UNICODE /D UNICODE /Gm- /EHsc /MT /GS /Gy /fp:precise /Zc:wchar_t /Zc:forScope /Zc:inline /std:c++20 /permissive- /Yc"cupch.h" /Fp"x64\Release\cusig.pch" /Fo"x64\Release\\" /Fd"x64\Release\vc143.pdb" /external:W3 /Gd /TP /FC /errorReport:prompt cupch.cpp

# @echo *** Compiling .cpp files with cl.exe ***

# %CL_EXE% /c /I%CUDA_PATH%\include /Zi /nologo /W3 /WX- /diagnostics:column /sdl /O2 /Oi /GL /D NDEBUG /D CUSIG_EXPORTS /D _WINDOWS /D _USRDLL /D _WINDLL /D _UNICODE /D UNICODE /Gm- /EHsc /MT /GS /Gy  /fp:precise /Zc:wchar_t /Zc:forScope /Zc:inline /std:c++20 /permissive- /Yu"cupch.h" /Fp"x64\Release\cusig.pch" /Fo"x64\Release\\" /Fd"x64\Release\vc143.pdb" /external:W3 /Gd /TP /FC /errorReport:prompt dllmain.cpp





@echo *** Compiling cuda files with nvcc ***

set NVCC_GENCODE= -arch=sm_50 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75
set NVCC_ARGS= %NVCC_GENCODE% --use-local-env -ccbin %VS_PATH%\bin\HostX64\x64 -x cu -rdc=true  -I%CUDA_PATH%\include    --keep-dir x64\Release  -maxrregcount=0   --machine 64 --compile -cudart static -lineinfo   -DNDEBUG -DCUSIG_EXPORTS -D_WINDOWS -D_USRDLL -D_WINDLL -D_UNICODE -DUNICODE -Xcompiler "/EHsc /W3 /nologo /O2 /FS   /MT " -Xcompiler "/Fdx64\Release\vc143.pdb" --dopt on

%NVCC_EXE% %NVCC_ARGS% -o %SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.cu.obj %SIGLIB_DIR%\cusig\cu_sig_kernel.cu

%NVCC_EXE% %NVCC_ARGS% -o %SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.h.obj %SIGLIB_DIR%\cusig\cu_sig_kernel.h

%NVCC_EXE% %NVCC_ARGS% -o %SIGLIB_DIR%\cusig\x64\Release\cu_path_transforms.cu.obj %SIGLIB_DIR%\cusig\cu_path_transforms.cu

%NVCC_EXE% %NVCC_ARGS% -o %SIGLIB_DIR%\cusig\x64\Release\cu_signature.cu.obj %SIGLIB_DIR%\cusig\cu_signature.cu



@echo ---------------------------------------------------------------------------------------
@echo link cuda obj files
REM link cuda obj files 
%NVCC_EXE% -dlink  -o x64\Release\cusig.device-link.obj -Xcompiler "/EHsc /W3 /nologo /O2   /MT " -Xcompiler "/Fdx64\Release\vc143.pdb" -L%CUDA_PATH%\bin\crt -L%CUDA_PATH%\lib\x64 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib cudart.lib cudadevrt.lib  %NVCC_GENCODE%  %SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.h.obj %SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.cu.obj %SIGLIB_DIR%\cusig\x64\Release\cu_path_transforms.cu.obj %SIGLIB_DIR%\cusig\x64\Release\cu_signature.cu.obj

rem pause



@echo ---------------------------------------------------------------------------------------
@echo link.exe

REM link final 

CD %SIGLIB_DIR%

md x64
cd x64
md Release

CD %SIGLIB_DIR%\cusig

set STD_LIBS=kernel32.lib

%LINK_EXE% /ERRORREPORT:PROMPT /OUT:%SIGLIB_DIR%\x64\Release\cusig.dll /NOLOGO /LIBPATH:%CUDA_PATH%\lib\x64 %STD_LIBS% cudart.lib cudadevrt.lib /MANIFEST /MANIFESTUAC:NO /manifest:embed /DEBUG /PDB:%SIGLIB_DIR%\x64\Release\cusig.pdb /SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF /LTCG:incremental /LTCGOUT:"x64\Release\cusig.iobj" /TLBID:1 /DYNAMICBASE /NXCOMPAT /IMPLIB:%SIGLIB_DIR%\x64\Release\cusig.lib /MACHINE:X64 /DLL %SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.h.obj %SIGLIB_DIR%\cusig\x64\Release\cu_sig_kernel.cu.obj %SIGLIB_DIR%\cusig\x64\Release\cu_path_transforms.cu.obj %SIGLIB_DIR%\cusig\x64\Release\cu_signature.cu.obj x64\Release\cupch.obj  "x64\Release\cusig.device-link.obj"
@echo ---------------------------------------------------------------------------------------

rem pause

