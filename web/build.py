"""Build the browser receiver using the same portable native source files."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emsdk', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/web')
    args = parser.parse_args()
    env = os.environ.copy()
    if args.emsdk:
        compiler = args.emsdk.resolve() / 'upstream/emscripten'
        env['PATH'] = str(compiler) + os.pathsep + env['PATH']
    emcmake = shutil.which('emcmake', path=env['PATH'])
    if not emcmake:
        raise SystemExit('Activate Emscripten 4.0.15 or pass --emsdk PATH')
    build = ROOT / 'build/web-wasm'
    subprocess.run([emcmake, 'cmake', '-S', str(ROOT/'web'), '-B', str(build),
                    '-DCMAKE_BUILD_TYPE=Release'], env=env, check=True)
    subprocess.run(['cmake', '--build', str(build), '--parallel', '4'], env=env, check=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT/'web/src', out, dirs_exist_ok=True)
    (out/'vendor').mkdir(exist_ok=True)
    for retired in ('player.js', 'vendor/mpegts.js', 'vendor/mpegts-LICENSE.txt'):
        (out / retired).unlink(missing_ok=True)
    fflate = ROOT/'web/node_modules/fflate'
    if not (fflate/'esm/browser.js').is_file():
        raise SystemExit('Run npm ci --prefix web before building the browser app')
    shutil.copy2(fflate/'esm/browser.js', out/'vendor/fflate.js')
    shutil.copy2(fflate/'LICENSE', out/'vendor/fflate-LICENSE.txt')
    (out/'wasm').mkdir(exist_ok=True)
    for name in ('ci8_resample','c3780_extract','deinterleave_qam64','ldpc_bch_decode'):
        for ext in ('mjs','wasm'):
            shutil.copy2(build/f'{name}.{ext}', out/'wasm')
    shutil.copytree(ROOT/'python/dtmb/data', out/'data', dirs_exist_ok=True)
    sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    cmake = (ROOT/'core/cpp/CMakeLists.txt').read_text(encoding='utf-8')
    core_version = re.search(r'project\(dtmb_core VERSION (\S+)', cmake).group(1)
    dirty = bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=ROOT, text=True).strip())
    source_hash = hashlib.sha256()
    for path in sorted((ROOT/'core/cpp').rglob('*')):
        if path.suffix in {'.cpp','.hpp','.h'}:
            source_hash.update(path.relative_to(ROOT).as_posix().encode())
            source_hash.update(path.read_bytes())
    web_hash = hashlib.sha256(Path(__file__).read_bytes())
    for path in sorted((ROOT/'web/src').rglob('*')):
        if path.is_file():
            web_hash.update(path.relative_to(ROOT/'web/src').as_posix().encode())
            web_hash.update(path.read_bytes())
    web_revision = web_hash.hexdigest()
    script_path = re.compile(r"(['\"])(\./[^'\"\n?]+\.js)\1")
    for path in sorted(out.glob('*.js')) + [out/'index.html']:
        content = script_path.sub(lambda match: f'{match[1]}{match[2]}?v={web_revision}{match[1]}',
                                  path.read_text(encoding='utf-8'))
        if path.name == 'index.html':
            content = content.replace('href="./style.css"', f'href="./style.css?v={web_revision}"')
        path.write_text(content, encoding='utf-8')
    (out/'build.json').write_text(json.dumps({'commit':sha,'dirty':dirty,'core':'dtmb-sdr','coreVersion':core_version,'coreSourceSha256':source_hash.hexdigest(),
        'webSourceSha256':web_revision,
        'loaders':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'wasm').glob('*.mjs')},
        'wasm':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'wasm').glob('*.wasm')}}), encoding='utf-8')
    (out/'.nojekyll').touch()
    print(f'Web app built: {out}')

if __name__ == '__main__':
    main()
