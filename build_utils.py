# Copyright 2025 Daniil Shmelev
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

"""
This file contains functions used for building pySigLib on Windows, Linus and MacOS.
"""

import zipfile
import subprocess
import traceback
import shutil
import os
import platform

import requests

B2_VERSION = '5.4.2'

ZIP_FOLDERNAME = 'b2-' + B2_VERSION
ZIP_FILENAME = ZIP_FOLDERNAME + '.zip'
B2_URL = 'https://github.com/bfgroup/b2/releases/download/' + B2_VERSION + '/b2-' + B2_VERSION + '.zip'

def _run(cmd, log_file, shell = False, check = True):
    # On Windows under non-cmd shells (Git Bash / MSYS2), two issues arise:
    # 1. .bat/.cmd files that call other .bat files by bare name fail because
    #    cmd.exe doesn't resolve .bat files in the current working directory.
    # 2. Executables with forward slashes (e.g. "x64/Release/foo.exe") fail
    #    because CreateProcess doesn't resolve them.
    env = None
    cwd = None
    cmd = list(cmd)  # avoid mutating caller's list
    if platform.system() == 'Windows':
        cmd[0] = os.path.abspath(cmd[0])

        if not shell and cmd[0].endswith(('.bat', '.cmd')):
            bat_dir = os.path.dirname(cmd[0])
            cmd = ['cmd.exe', '/c'] + cmd
            cwd = bat_dir
            env = os.environ.copy()
            # Add the bat's directory and subdirectories (up to 3 levels) to
            # PATH so nested "call <name>.bat" invocations resolve correctly.
            extra_dirs = [bat_dir]
            for root, dirs, files in os.walk(bat_dir):
                depth = root.replace(bat_dir, '').count(os.sep)
                if depth >= 3:
                    dirs.clear()
                    continue
                if any(f.endswith(('.bat', '.cmd')) for f in files):
                    extra_dirs.append(root)
            env['PATH'] = os.pathsep.join(extra_dirs) + os.pathsep + env.get('PATH', '')

    cmd_str = ' '.join(cmd)
    try:
        log_file.write("\n" + "=" * 10 + " Running Command " + "=" * 10 + "\n")
        log_file.write(cmd_str + "\n\n")
        output = subprocess.run(cmd, capture_output=True, check=check, text=True, shell = shell, env=env, cwd=cwd)
        log_file.write(output.stdout)
        log_file.write(output.stderr)
        return output
    except subprocess.CalledProcessError as e:
        log_file.write("\n" + "=" * 10 + " Exception occurred " + "=" * 10 + "\n")
        log_file.write("Exception occured whilst processing the command:\n\n")
        log_file.write(cmd_str + "\n\n")
        log_file.write(repr(e.stdout))
        log_file.write(repr(e.stderr))
        raise e
    except Exception as e:
        log_file.write("\n" + "=" * 10 + " Exception occurred " + "=" * 10 + "\n")
        log_file.write("Exception occured whilst processing the command:\n\n")
        log_file.write(cmd_str + "\n\n")
        traceback.print_exc(file=log_file)
        raise e

def get_paths(log_file):
    if 'CUDA_PATH' not in os.environ:
        raise RuntimeError("Error while compiling pysiglib: CUDA_PATH environment variable not set")

    cuda_path = os.environ['CUDA_PATH']

    vctoolsinstalldir = get_msvc_path(log_file)
    cl_path = os.path.join(vctoolsinstalldir, 'bin', 'HostX64', 'x64')
    os.environ["PATH"] += os.pathsep + cl_path

    idx = vctoolsinstalldir.find("VC")
    path = vctoolsinstalldir[:idx]

    output = _run([os.path.join(path, 'Common7', 'Tools', 'VsDevCmd.bat'), '&&', 'set'], log_file, shell = True)

    log_file.write(output.stdout + output.stderr)
    output = output.stdout
    start = output.find('INCLUDE') + 8
    end = output[start:].find('\n')
    include = output[start: start + end]

    dir_ = os.getcwd()
    return dir_, vctoolsinstalldir, cl_path, cuda_path, include

def get_b2(system, log_file):
    response = requests.get(B2_URL, timeout=(5, 60), stream=True)
    with open(ZIP_FILENAME, 'wb') as f:
        f.write(response.content)

    os.makedirs('.', exist_ok=True)

    with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
        zip_ref.extractall('.')

    os.chdir(ZIP_FOLDERNAME)
    if system == 'Windows':
        _run([".\\bootstrap.bat"], log_file)
    elif system in ('Linux', 'Darwin'):
        _run(["chmod", "-R", "755", "."], log_file)
        _run(["./bootstrap.sh"], log_file)
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + system + "' in get_b2()")

    os.chdir(r'..')

    os.chdir(ZIP_FOLDERNAME)
    _run(["./b2", "install", "--prefix=../b2"], log_file)
    os.chdir(r'..')

    if os.path.isfile(ZIP_FILENAME):
        os.remove(ZIP_FILENAME)

    if os.path.isdir(ZIP_FOLDERNAME):
        shutil.rmtree(ZIP_FOLDERNAME)


def build_cpsig(system, log_file):
    os.chdir(r'siglib')
    if system == 'Windows':
        _run(["../b2/b2", "--toolset=msvc", "--build-type=complete", "architecture=x86", "address-model=64", "release"], log_file)
    elif system == 'Linux':
        _run(["chmod", "755", "../b2"], log_file)
        _run(["../b2/bin/b2", "--toolset=gcc", "--build-type=complete", "architecture=x86", "address-model=64", "release"], log_file)
    elif system == 'Darwin':
        _run(["chmod", "755", "../b2"], log_file)
        _run(["../b2/bin/b2", "--build-type=complete", "release"], log_file)
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + system + "' in build_cpsig()")
    os.chdir(r'..')

def build_cusig(system, log_file):
    if system == 'Windows':
        _, vctoolsinstalldir, _, _, _ = get_paths(log_file)
        vc0 = vctoolsinstalldir[:vctoolsinstalldir.find(r'\Tools')]
        _run(["build_cusig.bat", vc0, vctoolsinstalldir], log_file)
    elif system == 'Linux':
        _run(["bash", "build_cusig.sh"], log_file)
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + system + "' in build_cusig()")

def get_msvc_path(log_file):
    os.chdir('siglib')
    output = _run(["../b2/b2", "toolset=msvc", "--debug-configuration", "-n"], log_file)
    os.chdir('..')
    output = output.stdout
    log_file.write(output + "\n")
    idx = output.find("[msvc-cfg] msvc-")
    output = output[idx:]
    start = output.find("'") + 1
    end = output.find("bin") - 1

    if idx == -1 or start == 0 or end == -2:
        raise RuntimeError("Error while compiling pysiglib: MSVC not found")

    return output[start: end]

def get_vec_info(system, log_file):
    if system == "Darwin":
        return ["neon"]

    os.chdir('avx_info')

    file_path = "jamroot.jam"
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "w") as file:
        file.write(
    """
exe avx_info : avx_info.cpp ;
install dist : avx_info :
   <variant>release:<location>x64/Release
   ;
"""
)
    if system == "Windows":
        _run(["../b2/b2", "release"], log_file)
        output = _run(["x64/Release/avx_info.exe"], log_file, check=False)
    elif system in ("Linux", "Darwin"):
        _run(["../b2/bin/b2", "release"], log_file)
        output = _run(["x64/Release/avx_info"], log_file, check=False)
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + system + "' in get_avx_info()")

    avx_instr_sets = ['avx', 'avx2', 'avx512f', 'avx512pf', 'avx512er', 'avx512cd']
    instructions = []

    rc = output.returncode
    for instr_ in avx_instr_sets:
        if rc & 1:
            instructions.append(instr_)
        rc = rc >> 1

    log_file.write("\nFound supported instruction sets: " + repr(instructions) + "\n")
    print("Found supported instruction sets: ", instructions)

    os.chdir('..')
    return instructions

def make_jamfiles(system, instructions, log_file):
    #siglib/Jamroot.jam
    file_path = "siglib/Jamroot.jam"
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "w") as file:
        file.write(
    """
build-project cpsig ;
install dist : cpsig ./cpsig/cpsig.h :
   <variant>release:<location>x64/Release
   ;
"""
)

    # siglib/cpsig/Jamfile.jam
    file_path = "siglib/cpsig/Jamfile.jam"
    if os.path.exists(file_path):
        os.remove(file_path)

    # Get a list of cpp files to compile
    cpp_files = os.listdir("siglib/cpsig")
    cpp_files.remove("cp_unit_tests.cpp")
    cpp_files = [x for x in cpp_files if x[-4:] == ".cpp"]
    cpp_files_str = ' '.join(cpp_files)

    # Get VEC info
    sys_vec_instr = "neon" if system == "Darwin" else "avx2"
    if sys_vec_instr in instructions:
        define_vec = '<define>VEC'
        log_file.write("\n" + sys_vec_instr + " supported, defining macro VEC in cpsig\n")
        print(sys_vec_instr + " supported, defining macro VEC in cpsig")
    else:
        define_vec = ''

    if system=="Windows":
        toolset = '<toolset>msvc:<cxxflags>"'

        if 'avx512f' in instructions:
            toolset += '/arch:AVX512'
        elif 'avx2' in instructions:
            toolset += '/arch:AVX2'
        elif 'avx' in instructions:
            toolset += '/arch:AVX'

        toolset += ' /Qvec-report:2"'

    elif system in ("Linux", "Darwin"):
        toolset = '<toolset>gcc:<cxxflags>"-march=native -ftree-vectorize -fopt-info-vec-missed"'
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + system + "' in make_jamfiles()")

    with open(file_path, "w") as file:
        file.write(
    f"""
lib cpsig : {cpp_files_str}
        : <define>CPSIG_EXPORTS {define_vec} <cxxstd>20 <threading>multi
        {toolset}
        : <variant>release
        ;
"""
        )
